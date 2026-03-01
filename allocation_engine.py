# allocation_engine.py
"""
Fixed Income Allocation Engine — Washington State
==================================================

Universe   : MUB (WA muni ETF)  ·  BIL (1-month T-bills)  ·  HYG (BB-rated HY)
Tax model  : WA (no state income tax); federal TEY / after-tax daily-return adjustment
Signal     : ICE BofA BB OAS z-score — FRED primary / IBKR ETF proxy fallback
Optimizers : Sortino  ·  Omega ratio (Keating-Shadwick 2002)  ·  Omega-Sortino composite

All three optimizers implement the same PortfolioOptimizer protocol, so they are
drop-in replacements.  Swap optimizer_type in build_engine() to compare.

Quick start
-----------
    from allocation_engine import build_engine, BacktestEngine, print_performance_report

    engine = build_engine(optimizer_type="omega_sortino")   # connects to IBKR TWS/Gateway
    weights = engine.target_weights()                        # live weights as of today
    print(weights)

    bt = BacktestEngine(engine).run(start="2020-01-01")      # walk-forward backtest
    print_performance_report(bt)

Dependencies
------------
    pip install numpy pandas scipy pandas-datareader ib_insync
    # ib_async is optional (only if you use IBAsyncAdapter)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

TRADING_DAYS: int = 252


# ============================================================
# Section 1: Performance Metrics  (numpy-only, no pandas in hot paths)
# ============================================================

def sortino_np(r: np.ndarray, mar: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualized Sortino ratio.  r is a 1-D daily return array."""
    if len(r) < 2:
        return -np.inf
    excess = r - mar
    downside = np.where(excess < 0.0, excess, 0.0)
    dd_std = downside.std(ddof=1)
    if dd_std <= 0.0 or not np.isfinite(dd_std):
        return np.inf if r.mean() > mar else -np.inf
    return float((r.mean() - mar) * periods / (dd_std * math.sqrt(periods)))


def omega_np(r: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega ratio (Keating & Shadwick 2002).

    Ω(τ) = E[max(r − τ, 0)] / E[max(τ − r, 0)]

    threshold should be a *daily* figure (e.g. 0.0 or rf_daily).
    Returns +inf when the denominator is zero and gains > 0.
    """
    gains = np.maximum(r - threshold, 0.0).mean()
    losses = np.maximum(threshold - r, 0.0).mean()
    if losses <= 1e-15:
        return np.inf if gains > 1e-15 else 1.0
    return float(gains / losses)


def max_drawdown_np(equity: np.ndarray) -> float:
    """Maximum drawdown from a 1-based cumulative equity curve."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(dd.min())


def sharpe_np(r: np.ndarray, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    vol = r.std(ddof=1)
    if vol <= 0.0 or not np.isfinite(vol):
        return np.nan
    return float((r.mean() - rf) * periods / (vol * math.sqrt(periods)))


def calmar_np(r: np.ndarray, periods: int = TRADING_DAYS) -> float:
    equity = np.cumprod(1.0 + r)
    mdd = abs(max_drawdown_np(equity))
    if mdd < 1e-12:
        return np.inf
    ann_ret = equity[-1] ** (periods / max(len(r), 1)) - 1.0
    return float(ann_ret / mdd)


def performance_summary(
    r: np.ndarray,
    periods: int = TRADING_DAYS,
    mar: float = 0.0,
    omega_threshold: float = 0.0,
) -> Dict[str, float]:
    """Compute all key metrics in one pass over the return array."""
    equity = np.cumprod(1.0 + r)
    ann_ret = equity[-1] ** (periods / max(len(r), 1)) - 1.0
    return {
        "ann_return":   float(ann_ret),
        "ann_vol":      float(r.std(ddof=1) * math.sqrt(periods)),
        "sharpe":       sharpe_np(r, rf=mar, periods=periods),
        "sortino":      sortino_np(r, mar=mar, periods=periods),
        "omega":        omega_np(r, threshold=omega_threshold),
        "max_drawdown": max_drawdown_np(equity),
        "calmar":       calmar_np(r, periods=periods),
    }


# ============================================================
# Section 2: Weight Utilities
# ============================================================

def simplex_project(w: np.ndarray) -> np.ndarray:
    """
    Project w onto the probability simplex: sum(w)=1, w≥0.
    Algorithm: Duchi et al. (2008).
    """
    w = np.asarray(w, dtype=float)
    if w.min() >= 0.0 and abs(w.sum() - 1.0) < 1e-10:
        return w
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    candidates = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1.0))[0]
    if len(candidates) == 0:
        w = np.clip(w, 0.0, None)
        s = w.sum()
        return w / s if s > 0.0 else np.ones_like(w) / len(w)
    rho = candidates[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(w - theta, 0.0)


def clip_and_normalize(
    w: np.ndarray,
    wmin: np.ndarray,
    wmax: np.ndarray,
    allow_short: bool = False,
) -> np.ndarray:
    """Clip to [wmin, wmax] then renormalize so weights sum to 1."""
    w = np.clip(w, wmin, wmax)
    if not allow_short:
        return simplex_project(w)
    s = w.sum()
    return w / s if abs(s) > 1e-12 else np.ones_like(w) / len(w)


# ============================================================
# Section 3: IBKR Data Adapters
# ============================================================

@runtime_checkable
class IBKRDataAdapter(Protocol):
    def hist_prices(
        self,
        symbols: List[str],
        end_dt: Optional[str] = None,
        duration: str = "3 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """Return DataFrame indexed by date, columns=symbols, values=adjusted close."""
        ...


@dataclass
class IBSyncAdapter:
    """
    Synchronous IBKR adapter using ib_insync.

    Requires TWS or IB Gateway running with API connections enabled.
    """
    host: str = "127.0.0.1"
    port: int = 7497        # 7497=TWS paper, 7496=TWS live, 4002=Gateway paper, 4001=Gateway live
    client_id: int = 12
    timeout: float = 20.0  # connection timeout in seconds
    request_delay: float = 0.25  # seconds between requests (rate limiting)

    def hist_prices(
        self,
        symbols: List[str],
        end_dt: Optional[str] = None,
        duration: str = "3 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        from ib_insync import IB, Stock, util  # type: ignore

        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
        try:
            series_map: Dict[str, pd.Series] = {}
            for sym in symbols:
                contract = Stock(sym, "SMART", "USD")
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt or "",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=1,
                )
                df = util.df(bars)
                if df is None or df.empty:
                    raise RuntimeError(f"No historical data returned for {sym!r}")
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                series_map[sym] = df["close"].rename(sym)
                logger.debug("IBSync: fetched %d bars for %s", len(df), sym)
                time.sleep(self.request_delay)
            return pd.concat(series_map.values(), axis=1).sort_index()
        finally:
            ib.disconnect()


@dataclass
class IBAsyncAdapter:
    """
    Asynchronous IBKR adapter using ib_async.

    Exposes the same synchronous interface by running asyncio internally.
    Note: calling from inside an already-running event loop (e.g. Jupyter)
    will raise RuntimeError — use IBSyncAdapter in that case.
    """
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 13
    request_delay: float = 0.25

    def hist_prices(
        self,
        symbols: List[str],
        end_dt: Optional[str] = None,
        duration: str = "3 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        return asyncio.run(
            self._fetch_all(symbols, end_dt, duration, bar_size, what_to_show, use_rth)
        )

    async def _fetch_all(
        self,
        symbols: List[str],
        end_dt: Optional[str],
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ) -> pd.DataFrame:
        from ib_async import IB, Stock  # type: ignore

        ib = IB()
        await ib.connectAsync(self.host, self.port, clientId=self.client_id)
        try:
            series_map: Dict[str, pd.Series] = {}
            for sym in symbols:
                contract = Stock(sym, "SMART", "USD")
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_dt or "",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=1,
                )
                df = pd.DataFrame(bars)
                if df.empty:
                    raise RuntimeError(f"No historical data returned for {sym!r}")
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                series_map[sym] = df["close"].rename(sym)
                logger.debug("IBAsync: fetched %d bars for %s", len(df), sym)
                await asyncio.sleep(self.request_delay)
            return pd.concat(series_map.values(), axis=1).sort_index()
        finally:
            # Some ib_async versions use disconnect(), others disconnectAsync().
            try:
                await ib.disconnectAsync()
            except AttributeError:
                ib.disconnect()


# ============================================================
# Section 4: Spread Signal Providers
# ============================================================

@runtime_checkable
class SpreadSignalProvider(Protocol):
    def get_bb_spread(self, start: Optional[pd.Timestamp] = None) -> pd.Series:
        """Return a daily BB/HY OAS series.  Units must be consistent (% or bps)."""
        ...


@dataclass
class FRED_BB_OAS_Provider:
    """
    ICE BofA BB US High Yield OAS from FRED (best quality signal).

    FRED series 'BAMLH0A1HYBB' is reported in percentage points
    (e.g. 2.50 = 250 bps).  Weekly frequency, forward-filled to daily.

    Requires: pip install pandas-datareader
    """
    fred_series: str = "BAMLH0A1HYBB"

    def get_bb_spread(self, start: Optional[pd.Timestamp] = None) -> pd.Series:
        from pandas_datareader import data as pdr  # type: ignore

        logger.info("Fetching FRED %s (start=%s)", self.fred_series, start)
        raw = pdr.DataReader(self.fred_series, "fred", start=start)
        s = raw[self.fred_series].rename("bb_oas").dropna()
        logger.debug("FRED: got %d BB OAS observations", len(s))
        return s


@dataclass
class IBKR_ETF_SpreadProxy_Provider:
    """
    IBKR-only fallback: momentum-based HY spread regime proxy.

    Method
    ------
    1. Fetch HY ETF (HYG) and short-bill ETF (BIL) price history.
    2. Compute log-price difference (credit spread proxy).
    3. Negative rolling momentum → signal for spread widening.

    Proxy respects the `start` parameter by estimating the required IBKR duration.
    This is NOT a true OAS — use only when FRED is unavailable.
    """
    data: IBKRDataAdapter
    hy_etf: str = "HYG"
    bill_etf: str = "BIL"
    momentum_window: int = 21  # trading-day momentum rolling window
    _WARMUP_DAYS: int = 400    # extra buffer days for rolling warmup

    def get_bb_spread(self, start: Optional[pd.Timestamp] = None) -> pd.Series:
        if start is None:
            duration = "5 Y"
        else:
            days_needed = (pd.Timestamp.today() - start).days + self._WARMUP_DAYS
            years = max(1, math.ceil(days_needed / 365))
            duration = f"{min(years, 10)} Y"

        logger.info(
            "Fetching IBKR ETF spread proxy (%s/%s), duration=%s",
            self.hy_etf, self.bill_etf, duration,
        )
        px = self.data.hist_prices([self.hy_etf, self.bill_etf], duration=duration)
        px = px.dropna()
        rel = np.log(px[self.hy_etf]) - np.log(px[self.bill_etf])
        # Negative momentum in relative returns → wider spreads proxy
        proxy = (-rel.diff().rolling(self.momentum_window).sum()).rename("bb_spread_proxy")
        if start is not None:
            proxy = proxy.loc[start:]
        return proxy.dropna()


# ============================================================
# Section 5: Tax Model — Washington State
# ============================================================

@dataclass
class TaxModelWA:
    """
    Washington State investor tax model.

    WA levies no state income tax.  Only federal rates apply.

    federal_marginal_rate      : your marginal federal income tax rate
    net_investment_income_tax  : 3.8% NIIT if MAGI exceeds IRS threshold, else 0.0

    Usage with daily return series
    --------------------------------
    Munis    → tax_equivalent_return(muni_rets)     raises return by 1/(1−t)
    Taxable  → after_tax_return(taxable_rets)        scales return by (1−t)

    Mixing these in the optimizer makes the Sortino / Omega objective comparable
    across the three sleeves on an after-tax basis.
    """
    federal_marginal_rate: float = 0.32
    net_investment_income_tax: float = 0.0  # set to 0.038 if NIIT applies

    @property
    def effective_rate(self) -> float:
        """Combined federal + NIIT rate, clamped to [0, 0.60]."""
        return float(np.clip(self.federal_marginal_rate + self.net_investment_income_tax, 0.0, 0.60))

    def tax_equivalent_return(self, muni_ret: pd.Series) -> pd.Series:
        """Gross up muni daily returns: TE = r_muni / (1 − t)."""
        return muni_ret / (1.0 - self.effective_rate)

    def after_tax_return(self, taxable_ret: pd.Series) -> pd.Series:
        """Scale down taxable daily returns: AT = r × (1 − t)."""
        return taxable_ret * (1.0 - self.effective_rate)


# ============================================================
# Section 6: Regime Detection
# ============================================================

class RegimeState(Enum):
    RISK_ON  = "risk_on"
    NEUTRAL  = "neutral"
    RISK_OFF = "risk_off"


@dataclass
class RegimeConfig:
    z_lookback: int = 252          # rolling window for z-score (trading days)
    widen_threshold: float = 0.75  # z > this  →  RISK_OFF
    tight_threshold: float = -0.75 # z < this  →  RISK_ON


@dataclass
class RegimeDetector:
    """Classifies spread regime via rolling z-score of a spread series."""
    cfg: RegimeConfig = field(default_factory=RegimeConfig)

    def z_score_series(self, spread: pd.Series) -> pd.Series:
        lb = self.cfg.z_lookback
        m = spread.rolling(lb, min_periods=lb // 2).mean()
        s = spread.rolling(lb, min_periods=lb // 2).std(ddof=1)
        return ((spread - m) / s.replace(0.0, np.nan)).rename("bb_zscore")

    def state_at(self, z: float) -> RegimeState:
        if not np.isfinite(z):
            return RegimeState.NEUTRAL
        if z >= self.cfg.widen_threshold:
            return RegimeState.RISK_OFF
        if z <= self.cfg.tight_threshold:
            return RegimeState.RISK_ON
        return RegimeState.NEUTRAL

    def classify_series(self, spread: pd.Series) -> pd.Series:
        """Return a string Series of RegimeState values aligned to spread.index."""
        return (
            self.z_score_series(spread)
            .map(self.state_at)
            .map(lambda s: s.value)
            .rename("regime")
        )

    def current_state(self, spread: pd.Series) -> Tuple[RegimeState, float]:
        """(state, last_z) from the most recent spread observation."""
        zs = self.z_score_series(spread).dropna()
        if zs.empty:
            return RegimeState.NEUTRAL, 0.0
        z = float(zs.iloc[-1])
        return self.state_at(z), z


# ============================================================
# Section 7: Optimizer Framework
# ============================================================

@dataclass
class OptimizerResult:
    weights: np.ndarray
    success: bool
    message: str
    objective_value: float
    n_iter: int


class PortfolioOptimizer(Protocol):
    """
    Interface for all portfolio optimizers.

    Parameters
    ----------
    returns  : T × N numpy array of daily returns (tax-adjusted)
    w0       : N-vector starting weights
    wmin     : N-vector lower bounds
    wmax     : N-vector upper bounds (may be regime-adjusted by the engine)
    w_prev   : N-vector previous weights (for turnover penalty; defaults to w0)
    """
    def optimize(
        self,
        returns: np.ndarray,
        w0: np.ndarray,
        wmin: np.ndarray,
        wmax: np.ndarray,
        w_prev: Optional[np.ndarray] = None,
    ) -> OptimizerResult: ...


# ---------------------------------------------------------------------------
# Sortino Optimizer
# ---------------------------------------------------------------------------

@dataclass
class SortinoOptimizerConfig:
    mar_daily: float = 0.0         # minimum acceptable daily return
    turnover_penalty: float = 0.05 # λ · Σ|w − w_prev|
    allow_short: bool = False
    max_iter: int = 500
    ftol: float = 1e-10
    min_history: int = 63          # minimum observations before optimizing


@dataclass
class SortinoOptimizer:
    """Maximize the annualized Sortino ratio subject to linear constraints."""
    cfg: SortinoOptimizerConfig = field(default_factory=SortinoOptimizerConfig)

    def optimize(
        self,
        returns: np.ndarray,
        w0: np.ndarray,
        wmin: np.ndarray,
        wmax: np.ndarray,
        w_prev: Optional[np.ndarray] = None,
    ) -> OptimizerResult:
        if len(returns) < self.cfg.min_history:
            return OptimizerResult(w0, False, "Insufficient history", np.nan, 0)

        R = returns
        _mar = self.cfg.mar_daily
        _tp  = self.cfg.turnover_penalty
        _wp  = w_prev if w_prev is not None else w0.copy()

        def objective(w: np.ndarray) -> float:
            sr = sortino_np(R @ w, mar=_mar)
            val = -sr if np.isfinite(sr) else 1e6
            if _tp > 0.0:
                val += _tp * float(np.abs(w - _wp).sum())
            return val

        res = minimize(
            objective, x0=w0, method="SLSQP",
            bounds=[(float(wmin[i]), float(wmax[i])) for i in range(len(w0))],
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": self.cfg.max_iter, "ftol": self.cfg.ftol, "disp": False},
        )
        if not res.success:
            logger.warning("SortinoOptimizer: %s", res.message)

        w = clip_and_normalize(
            res.x if res.success else w0, wmin, wmax, self.cfg.allow_short
        )
        return OptimizerResult(w, res.success, res.message, float(res.fun), res.nit)


# ---------------------------------------------------------------------------
# Omega Optimizer
# ---------------------------------------------------------------------------

@dataclass
class OmegaOptimizerConfig:
    threshold_daily: float = 0.0   # τ — daily return threshold
    turnover_penalty: float = 0.05
    allow_short: bool = False
    max_iter: int = 500
    ftol: float = 1e-10
    min_history: int = 63


@dataclass
class OmegaOptimizer:
    """
    Maximize the Omega ratio (Keating & Shadwick 2002).

    Ω(τ) = E[max(r−τ, 0)] / E[max(τ−r, 0)]

    Omega captures the full return distribution without assuming normality:
    it rewards upside and penalizes downside independently.  This makes it
    well-suited to fixed-income return distributions, which are left-skewed
    with occasional large drawdowns.
    """
    cfg: OmegaOptimizerConfig = field(default_factory=OmegaOptimizerConfig)

    def optimize(
        self,
        returns: np.ndarray,
        w0: np.ndarray,
        wmin: np.ndarray,
        wmax: np.ndarray,
        w_prev: Optional[np.ndarray] = None,
    ) -> OptimizerResult:
        if len(returns) < self.cfg.min_history:
            return OptimizerResult(w0, False, "Insufficient history", np.nan, 0)

        R   = returns
        _tau = self.cfg.threshold_daily
        _tp  = self.cfg.turnover_penalty
        _wp  = w_prev if w_prev is not None else w0.copy()

        def objective(w: np.ndarray) -> float:
            om = omega_np(R @ w, threshold=_tau)
            if om == np.inf:
                val = -1e6
            elif np.isfinite(om):
                val = -om
            else:
                val = 1e6
            if _tp > 0.0:
                val += _tp * float(np.abs(w - _wp).sum())
            return val

        res = minimize(
            objective, x0=w0, method="SLSQP",
            bounds=[(float(wmin[i]), float(wmax[i])) for i in range(len(w0))],
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": self.cfg.max_iter, "ftol": self.cfg.ftol, "disp": False},
        )
        if not res.success:
            logger.warning("OmegaOptimizer: %s", res.message)

        w = clip_and_normalize(
            res.x if res.success else w0, wmin, wmax, self.cfg.allow_short
        )
        return OptimizerResult(w, res.success, res.message, float(res.fun), res.nit)


# ---------------------------------------------------------------------------
# Omega-Sortino Composite Optimizer  (recommended default)
# ---------------------------------------------------------------------------

@dataclass
class OmegaSortinoConfig:
    """
    Composite objective: α · Ω(τ) + (1−α) · Sortino(MAR)

    Each metric is normalized by its value at the initial weights before
    blending, so neither term dominates due to scale differences.

    omega_weight=0.5 gives equal emphasis.  Tilt toward 1.0 to lean on
    the full-distribution view; toward 0.0 for pure downside-deviation focus.
    """
    omega_weight: float = 0.5        # α ∈ [0, 1]
    threshold_daily: float = 0.0     # τ for Omega
    mar_daily: float = 0.0           # MAR for Sortino
    turnover_penalty: float = 0.05
    allow_short: bool = False
    max_iter: int = 600
    ftol: float = 1e-10
    min_history: int = 63


@dataclass
class OmegaSortinoOptimizer:
    """
    Composite Omega-Sortino optimizer.

    Rationale: Omega captures the full shape of the return distribution
    (rewarding upside; penalizing downside independently), while Sortino
    focuses specifically on below-MAR deviation.  Together they provide
    a robust objective for a tax-aware fixed-income portfolio where the
    return distribution is non-normal and downside protection matters most.
    """
    cfg: OmegaSortinoConfig = field(default_factory=OmegaSortinoConfig)

    def optimize(
        self,
        returns: np.ndarray,
        w0: np.ndarray,
        wmin: np.ndarray,
        wmax: np.ndarray,
        w_prev: Optional[np.ndarray] = None,
    ) -> OptimizerResult:
        if len(returns) < self.cfg.min_history:
            return OptimizerResult(w0, False, "Insufficient history", np.nan, 0)

        R    = returns
        _tau = self.cfg.threshold_daily
        _mar = self.cfg.mar_daily
        _a   = self.cfg.omega_weight
        _tp  = self.cfg.turnover_penalty
        _wp  = w_prev if w_prev is not None else w0.copy()

        # Scale factors computed at starting weights (cheap, done once)
        port0     = R @ w0
        _om0_abs  = max(abs(omega_np(port0, _tau)), 1e-6)
        _sr0_abs  = max(abs(sortino_np(port0, _mar)), 1e-6)

        def objective(w: np.ndarray) -> float:
            port = R @ w
            om = omega_np(port, _tau)
            sr = sortino_np(port, _mar)
            om_n = (om / _om0_abs) if np.isfinite(om) and om != np.inf else (-1e3 if om != np.inf else 1e3)
            sr_n = (sr / _sr0_abs) if np.isfinite(sr) else -1e3
            val = -(_a * om_n + (1.0 - _a) * sr_n)
            if _tp > 0.0:
                val += _tp * float(np.abs(w - _wp).sum())
            return float(val)

        res = minimize(
            objective, x0=w0, method="SLSQP",
            bounds=[(float(wmin[i]), float(wmax[i])) for i in range(len(w0))],
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"maxiter": self.cfg.max_iter, "ftol": self.cfg.ftol, "disp": False},
        )
        if not res.success:
            logger.warning("OmegaSortinoOptimizer: %s", res.message)

        w = clip_and_normalize(
            res.x if res.success else w0, wmin, wmax, self.cfg.allow_short
        )
        return OptimizerResult(w, res.success, res.message, float(res.fun), res.nit)


# ============================================================
# Section 8: Universe & Engine Configuration
# ============================================================

@dataclass
class EngineUniverse:
    muni:  str = "MUB"   # WA-friendly muni ETF (consider VTEB; MUB is national but liquid)
    bills: str = "BIL"   # 1–3 month T-bills
    hy_bb: str = "HYG"   # BB-rated HY (HYG skews more BB than JNK)

    @property
    def symbols(self) -> List[str]:
        return [self.muni, self.bills, self.hy_bb]

    @property
    def asset_names(self) -> List[str]:
        return ["muni", "bills", "hy"]

    @property
    def symbol_to_name(self) -> Dict[str, str]:
        return {self.muni: "muni", self.bills: "bills", self.hy_bb: "hy"}


@dataclass
class HYCapConfig:
    """
    Defines how the HY weight cap shifts with spread regime.

    risk_off_scale   : wmax_hy × scale in RISK_OFF (e.g. 0.35 → 35% of normal cap)
    risk_on_ceiling  : absolute HY cap in RISK_ON; intentionally set *above* the
                       default wmax_hy so the optimizer has room to buy the dip.
    """
    risk_off_scale:  float = 0.35   # 0.60 × 0.35 = 0.21 max HY in risk-off
    risk_on_ceiling: float = 0.75   # absolute max HY in risk-on (above default 0.60)

    def adjusted_cap(self, default_cap: float, state: RegimeState) -> float:
        if state == RegimeState.RISK_OFF:
            return default_cap * self.risk_off_scale
        if state == RegimeState.RISK_ON:
            return self.risk_on_ceiling   # can exceed default_cap by design
        return default_cap


@dataclass
class AllocationEngineConfig:
    # ---- Data ----
    duration:  str  = "5 Y"
    bar_size:  str  = "1 day"
    use_rth:   bool = True

    # ---- Tax ----
    tax: TaxModelWA = field(default_factory=TaxModelWA)

    # ---- Regime ----
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    hy_cap: HYCapConfig  = field(default_factory=HYCapConfig)

    # ---- Rebalance ----
    rebalance_freq: str = "ME"   # "ME" = month-end (pandas ≥ 2.2)

    # ---- Static weight bounds (HY cap further adjusted by HYCapConfig) ----
    wmin: Dict[str, float] = field(default_factory=lambda: {
        "muni": 0.05, "bills": 0.05, "hy": 0.00
    })
    wmax: Dict[str, float] = field(default_factory=lambda: {
        "muni": 0.80, "bills": 0.90, "hy": 0.60
    })

    def bounds_arrays(
        self,
        regime_state: RegimeState = RegimeState.NEUTRAL,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (wmin_arr, wmax_arr) with regime-adjusted HY cap."""
        wmin = np.array([self.wmin["muni"], self.wmin["bills"], self.wmin["hy"]], float)
        wmax = np.array([self.wmax["muni"], self.wmax["bills"], self.wmax["hy"]], float)
        raw_cap = self.hy_cap.adjusted_cap(self.wmax["hy"], regime_state)
        wmax[2] = float(np.clip(raw_cap, wmin[2], 1.0))
        return wmin, wmax


# ============================================================
# Section 9: Allocation Engine
# ============================================================

@dataclass
class AllocationEngine:
    """
    Main orchestrator.

    Wires together:
        data adapter → spread signal → tax adjustment → regime detection
        → bounds construction → optimizer → target weights

    The `optimizer` field accepts any object satisfying PortfolioOptimizer:
        SortinoOptimizer, OmegaOptimizer, OmegaSortinoOptimizer, or custom.
    """
    data:             IBKRDataAdapter
    spread_provider:  SpreadSignalProvider
    optimizer:        PortfolioOptimizer
    universe:         EngineUniverse         = field(default_factory=EngineUniverse)
    cfg:              AllocationEngineConfig  = field(default_factory=AllocationEngineConfig)

    # ---- Internal helpers ------------------------------------------------

    def _prices(self) -> pd.DataFrame:
        px = self.data.hist_prices(
            self.universe.symbols,
            duration=self.cfg.duration,
            bar_size=self.cfg.bar_size,
            use_rth=self.cfg.use_rth,
        )
        # Rename symbols → canonical names using explicit mapping (no positional guessing)
        return px.dropna().rename(columns=self.universe.symbol_to_name)

    def _returns(self, px: pd.DataFrame) -> pd.DataFrame:
        return px.pct_change().dropna()

    def _tax_adjusted(self, rets: pd.DataFrame) -> pd.DataFrame:
        """Apply WA tax adjustments: munis grossed up, taxable scaled down."""
        return pd.concat([
            self.cfg.tax.tax_equivalent_return(rets["muni"]).rename("muni"),
            self.cfg.tax.after_tax_return(rets["bills"]).rename("bills"),
            self.cfg.tax.after_tax_return(rets["hy"]).rename("hy"),
        ], axis=1).dropna()

    def _fetch_spread(self, rets_start: pd.Timestamp) -> pd.Series:
        """Fetch spread with extra buffer for z-score rolling warmup."""
        buffer = pd.Timedelta(days=self.cfg.regime.z_lookback + 120)
        return self.spread_provider.get_bb_spread(start=rets_start - buffer)

    def _w_to_dict(self, w: np.ndarray) -> Dict[str, float]:
        return {name: float(w[i]) for i, name in enumerate(self.universe.asset_names)}

    def _w_to_vec(self, d: Dict[str, float]) -> np.ndarray:
        return np.array([d.get(n, 0.0) for n in self.universe.asset_names], float)

    # ---- Public API -------------------------------------------------------

    def target_weights(
        self,
        w_prev: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute target allocation weights as of the latest available data.

        Parameters
        ----------
        w_prev : previous allocation dict (for turnover penalty).
                 Keys: "muni", "bills", "hy".  If None, equal weights used.

        Returns
        -------
        dict with keys "muni", "bills", "hy" that sum to 1.0.
        """
        px       = self._prices()
        rets     = self._returns(px)
        rets_adj = self._tax_adjusted(rets)

        spread   = self._fetch_spread(rets_adj.index.min())
        detector = RegimeDetector(cfg=self.cfg.regime)
        # Align spread to return index for regime lookup
        spread_aligned = spread.reindex(rets_adj.index, method="ffill").dropna()
        state, z = detector.current_state(spread_aligned)

        logger.info("Spread regime: %s  (z-score=%.2f)", state.value, z)

        wmin, wmax = self.cfg.bounds_arrays(state)
        n  = len(self.universe.asset_names)
        w0 = self._w_to_vec(w_prev) if w_prev else np.ones(n) / n

        result = self.optimizer.optimize(
            returns=rets_adj.values,
            w0=w0,
            wmin=wmin,
            wmax=wmax,
            w_prev=w0.copy(),
        )
        if not result.success:
            logger.warning(
                "Optimizer did not converge (%s). Regime=%s. Using w0.",
                result.message, state.value,
            )
        return self._w_to_dict(result.weights)

    def regime_snapshot(self) -> Dict[str, Union[str, float]]:
        """Quick regime check (fetches prices + spread, no optimization)."""
        px       = self._prices()
        rets     = self._returns(px)
        rets_adj = self._tax_adjusted(rets)
        spread   = self._fetch_spread(rets_adj.index.min())
        detector = RegimeDetector(cfg=self.cfg.regime)
        spread_al = spread.reindex(rets_adj.index, method="ffill").dropna()
        state, z  = detector.current_state(spread_al)
        wmin, wmax = self.cfg.bounds_arrays(state)
        return {
            "regime":    state.value,
            "zscore":    round(z, 3),
            "hy_wmax":   round(float(wmax[2]), 4),
        }


# ============================================================
# Section 10: Backtest Engine
# ============================================================

@dataclass
class BacktestResult:
    weights:         pd.DataFrame       # columns: w_muni, w_bills, w_hy
    port_returns:    pd.Series          # daily portfolio returns (tax-adjusted)
    port_nav:        pd.Series          # NAV starting at 1.0
    regime:          pd.Series          # daily regime label
    zscore:          pd.Series          # daily spread z-score
    rebalance_dates: pd.DatetimeIndex
    summary:         Dict[str, float]   # performance metrics

    def __repr__(self) -> str:
        s = self.summary
        return (
            f"BacktestResult("
            f"ann_ret={s.get('ann_return', np.nan):.1%}, "
            f"sortino={s.get('sortino', np.nan):.2f}, "
            f"omega={s.get('omega', np.nan):.2f}, "
            f"mdd={s.get('max_drawdown', np.nan):.1%})"
        )


@dataclass
class BacktestEngine:
    """
    Walk-forward monthly rebalance backtest.

    Design choices
    --------------
    - Spread series is fetched ONCE before the loop (not once per rebalance).
    - Z-score series is computed ONCE on the full spread history.
    - Regime lookup per date is an O(1) index operation.
    - Optimizer runs on an expanding window of returns up to (not including) each
      rebalance date to avoid look-ahead bias.
    - First rebalance fires only after `min_history` daily observations have
      accumulated (optimizer warmup guard).
    """
    engine: AllocationEngine

    def run(
        self,
        start:           Optional[str]            = None,
        end:             Optional[str]             = None,
        initial_weights: Optional[Dict[str, float]] = None,
        min_history:     int                       = 252,
    ) -> BacktestResult:
        """
        Parameters
        ----------
        start           : backtest start date (e.g. "2020-01-01")
        end             : backtest end date
        initial_weights : starting weight dict; defaults to equal weight
        min_history     : minimum observations before the first rebalance fires
        """
        eng = self.engine
        cfg = eng.cfg

        # 1. Fetch all data up front ----------------------------------------
        px       = eng._prices()
        rets     = eng._returns(px)
        rets_adj = eng._tax_adjusted(rets)

        # 2. Fetch spread once (with warmup buffer) -------------------------
        spread_raw = eng._fetch_spread(rets_adj.index.min())

        # 3. Align spread to return calendar; forward-fill gaps (weekends, holidays)
        full_cal   = pd.date_range(rets_adj.index.min(), rets_adj.index.max(), freq="B")
        spread_cal = spread_raw.reindex(full_cal, method="ffill")
        spread_ret = spread_cal.reindex(rets_adj.index, method="ffill")

        # 4. Pre-compute full z-score series (rolling, no look-ahead)
        detector = RegimeDetector(cfg=cfg.regime)
        zs_full  = detector.z_score_series(spread_ret)   # aligned to rets_adj.index

        # 5. Slice backtest window ------------------------------------------
        if start:
            rets_adj = rets_adj.loc[pd.to_datetime(start):]
            zs_full  = zs_full.loc[pd.to_datetime(start):]
        if end:
            rets_adj = rets_adj.loc[:pd.to_datetime(end)]
            zs_full  = zs_full.loc[:pd.to_datetime(end)]

        # 6. Rebalance dates (month-end intersected with actual trading days)
        rb_dates = set(
            rets_adj.resample(cfg.rebalance_freq).last().index
            .intersection(rets_adj.index)
        )

        # 7. Initialize ----------------------------------------------------
        n      = len(eng.universe.asset_names)
        w_cur  = initial_weights or {k: 1.0 / n for k in eng.universe.asset_names}
        w_vec  = eng._w_to_vec(w_cur)

        weights_hist:     List[np.ndarray]   = []
        regime_hist:      List[str]          = []
        zscore_hist:      List[float]        = []
        port_rets_list:   List[float]        = []
        rebalanced_dates: List[pd.Timestamp] = []

        idx_list    = rets_adj.index.tolist()
        R_full      = rets_adj.values       # pre-extract numpy array

        # 8. Walk-forward loop ----------------------------------------------
        for i, dt in enumerate(idx_list):
            # Regime as of today (backward-looking z-score, no look-ahead)
            z_today = float(zs_full.get(dt, np.nan))
            if not np.isfinite(z_today):
                z_today = 0.0
            state = detector.state_at(z_today)

            regime_hist.append(state.value)
            zscore_hist.append(z_today)

            # Rebalance?
            if dt in rb_dates and i >= min_history:
                wmin, wmax = cfg.bounds_arrays(state)
                R_window   = R_full[:i]          # expanding window, no today's return

                result = eng.optimizer.optimize(
                    returns=R_window,
                    w0=w_vec,
                    wmin=wmin,
                    wmax=wmax,
                    w_prev=w_vec,
                )
                if result.success:
                    w_vec = result.weights
                    logger.debug(
                        "Rebalanced %s | regime=%s | muni=%.1f%% bills=%.1f%% hy=%.1f%%",
                        dt.date(), state.value,
                        w_vec[0] * 100, w_vec[1] * 100, w_vec[2] * 100,
                    )
                else:
                    logger.warning(
                        "Rebalance at %s failed (%s). Keeping current weights.",
                        dt.date(), result.message,
                    )
                rebalanced_dates.append(dt)

            weights_hist.append(w_vec.copy())
            port_rets_list.append(float(R_full[i] @ w_vec))

        # 9. Assemble output -----------------------------------------------
        idx      = rets_adj.index
        w_df     = pd.DataFrame(weights_hist, index=idx, columns=["w_muni", "w_bills", "w_hy"])
        port_ret = pd.Series(port_rets_list, index=idx, name="port_ret")
        port_nav = (1.0 + port_ret).cumprod().rename("port_nav")
        regime_s = pd.Series(regime_hist, index=idx, name="regime")
        zscore_s = pd.Series(zscore_hist, index=idx, name="bb_zscore")

        summary = performance_summary(port_ret.values, mar=0.0, omega_threshold=0.0)
        summary["rebalance_count"] = float(len(rebalanced_dates))

        return BacktestResult(
            weights=w_df,
            port_returns=port_ret,
            port_nav=port_nav,
            regime=regime_s,
            zscore=zscore_s,
            rebalance_dates=pd.DatetimeIndex(rebalanced_dates),
            summary=summary,
        )


# ============================================================
# Section 11: Reporting
# ============================================================

def print_performance_report(result: BacktestResult, title: str = "Backtest") -> None:
    s  = result.summary
    w  = result.weights
    rc = result.regime.value_counts(normalize=True)

    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}")
    print(f"  Period       : {result.port_returns.index[0].date()} → "
          f"{result.port_returns.index[-1].date()}")
    print(f"  Trading days : {len(result.port_returns)}")
    print(f"{'─' * 62}")
    print(f"  Ann Return   : {s.get('ann_return',   np.nan):>8.2%}")
    print(f"  Ann Vol      : {s.get('ann_vol',       np.nan):>8.2%}")
    print(f"  Sharpe       : {s.get('sharpe',        np.nan):>8.2f}")
    print(f"  Sortino      : {s.get('sortino',       np.nan):>8.2f}")
    print(f"  Omega        : {s.get('omega',         np.nan):>8.2f}")
    print(f"  Max Drawdown : {s.get('max_drawdown',  np.nan):>8.2%}")
    print(f"  Calmar       : {s.get('calmar',        np.nan):>8.2f}")
    print(f"  Rebalances   : {int(s.get('rebalance_count', 0)):>8d}")
    print(f"{'─' * 62}")
    print(f"  Mean Weights (over full period)")
    print(f"    Muni       : {w['w_muni'].mean():>7.1%}")
    print(f"    Bills      : {w['w_bills'].mean():>7.1%}")
    print(f"    HY         : {w['w_hy'].mean():>7.1%}")
    print(f"{'─' * 62}")
    print(f"  Regime Distribution")
    for label in ["risk_on", "neutral", "risk_off"]:
        print(f"    {label:<11}: {rc.get(label, 0.0):>6.1%}")
    print(f"{'═' * 62}\n")


def compare_optimizers(
    engine_factory,
    start: str = "2020-01-01",
    end:   Optional[str] = None,
) -> Dict[str, BacktestResult]:
    """
    Run all three optimizers on the same universe and print a comparison.

    Parameters
    ----------
    engine_factory : callable(optimizer_type) → AllocationEngine
    start, end     : backtest window
    """
    results: Dict[str, BacktestResult] = {}
    for opt_type in ("sortino", "omega", "omega_sortino"):
        eng = engine_factory(opt_type)
        bt  = BacktestEngine(eng).run(start=start, end=end)
        results[opt_type] = bt
        print_performance_report(bt, title=f"Optimizer: {opt_type.upper()}")
    return results


# ============================================================
# Section 12: Factory & Example Usage
# ============================================================

def build_engine(
    optimizer_type:        str   = "omega_sortino",
    federal_marginal_rate: float = 0.32,
    niit:                  float = 0.0,
    adapter:               str   = "sync",
    host:                  str   = "127.0.0.1",
    port:                  int   = 7497,
    client_id:             int   = 12,
    use_fred_spread:       bool  = True,
    omega_weight:          float = 0.5,
    turnover_penalty:      float = 0.05,
    duration:              str   = "5 Y",
) -> AllocationEngine:
    """
    Factory function — builds a fully wired AllocationEngine.

    Parameters
    ----------
    optimizer_type        : "sortino" | "omega" | "omega_sortino"
    federal_marginal_rate : your federal marginal income tax rate (e.g. 0.32)
    niit                  : 3.8% Net Investment Income Tax if applicable, else 0.0
    adapter               : "sync" (ib_insync)  or  "async" (ib_async)
    host / port / client_id : IBKR TWS or Gateway connection params
    use_fred_spread       : True → FRED OAS (best quality); False → IBKR ETF proxy
    omega_weight          : α blend in OmegaSortino (0=pure Sortino, 1=pure Omega)
    turnover_penalty      : λ on L1 turnover (raise to reduce trading frequency)
    duration              : IBKR history duration string (e.g. "5 Y", "3 Y")
    """
    # --- Data adapter ---
    if adapter == "sync":
        data: IBKRDataAdapter = IBSyncAdapter(host=host, port=port, client_id=client_id)
    elif adapter == "async":
        data = IBAsyncAdapter(host=host, port=port, client_id=client_id)
    else:
        raise ValueError(f"adapter must be 'sync' or 'async', got {adapter!r}")

    # --- Spread provider ---
    if use_fred_spread:
        spread: SpreadSignalProvider = FRED_BB_OAS_Provider()
    else:
        spread = IBKR_ETF_SpreadProxy_Provider(data=data)

    # --- Tax model ---
    tax = TaxModelWA(
        federal_marginal_rate=federal_marginal_rate,
        net_investment_income_tax=niit,
    )

    # --- Optimizer ---
    _tp = turnover_penalty
    if optimizer_type == "sortino":
        opt: PortfolioOptimizer = SortinoOptimizer(
            SortinoOptimizerConfig(mar_daily=0.0, turnover_penalty=_tp)
        )
    elif optimizer_type == "omega":
        opt = OmegaOptimizer(
            OmegaOptimizerConfig(threshold_daily=0.0, turnover_penalty=_tp)
        )
    elif optimizer_type == "omega_sortino":
        opt = OmegaSortinoOptimizer(
            OmegaSortinoConfig(
                omega_weight=omega_weight,
                threshold_daily=0.0,
                mar_daily=0.0,
                turnover_penalty=_tp,
            )
        )
    else:
        raise ValueError(
            f"optimizer_type must be 'sortino'|'omega'|'omega_sortino', got {optimizer_type!r}"
        )

    # --- Engine config ---
    cfg = AllocationEngineConfig(
        duration=duration,
        bar_size="1 day",
        use_rth=True,
        tax=tax,
        regime=RegimeConfig(z_lookback=252, widen_threshold=0.75, tight_threshold=-0.75),
        hy_cap=HYCapConfig(risk_off_scale=0.35, risk_on_ceiling=0.75),
        rebalance_freq="ME",
        wmin={"muni": 0.05, "bills": 0.05, "hy": 0.00},
        wmax={"muni": 0.80, "bills": 0.90, "hy": 0.60},
    )

    return AllocationEngine(
        data=data,
        spread_provider=spread,
        optimizer=opt,
        universe=EngineUniverse(muni="MUB", bills="BIL", hy_bb="HYG"),
        cfg=cfg,
    )


# ============================================================
# Section 13: Entry Point
# ============================================================

def _example_live() -> None:
    """Compute today's target weights and print a regime snapshot."""
    engine = build_engine(
        optimizer_type="omega_sortino",
        federal_marginal_rate=0.32,
        niit=0.038,   # set 0.0 if MAGI below NIIT threshold
        adapter="sync",
        host="127.0.0.1",
        port=7497,
        client_id=12,
        use_fred_spread=True,
    )

    snapshot = engine.regime_snapshot()
    print(f"\nRegime snapshot : {snapshot}")

    weights = engine.target_weights(w_prev={"muni": 0.40, "bills": 0.30, "hy": 0.30})
    print(f"Target weights  : { {k: f'{v:.1%}' for k, v in weights.items()} }")


def _example_backtest() -> BacktestResult:
    """3-year walk-forward backtest, all three optimizers."""
    def _factory(opt_type: str) -> AllocationEngine:
        return build_engine(
            optimizer_type=opt_type,
            federal_marginal_rate=0.32,
            use_fred_spread=True,
        )

    results = compare_optimizers(_factory, start="2020-01-01", end="2022-12-31")
    return results["omega_sortino"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _example_live()
