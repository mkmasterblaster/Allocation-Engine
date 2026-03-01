# allocation_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple
import math
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ============================================================
# Utilities
# ============================================================

def _to_series(x, name=None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, name=name)

def annualize_return(daily_mean: float, periods: int = 252) -> float:
    return (1.0 + daily_mean) ** periods - 1.0

def annualize_vol(daily_std: float, periods: int = 252) -> float:
    return daily_std * math.sqrt(periods)

def sortino_ratio(returns: pd.Series, mar: float = 0.0, periods: int = 252) -> float:
    """
    Annualized Sortino ratio using MAR (minimum acceptable return) per-period.
    If you want MAR = risk-free daily rate, pass it in.
    """
    r = returns.dropna()
    if r.empty:
        return np.nan
    downside = (r - mar).clip(upper=0.0)
    downside_std = downside.std(ddof=1)
    if downside_std <= 0 or np.isnan(downside_std):
        return np.nan
    mu = r.mean()
    # Annualize numerator and denominator consistently
    ann_mu_excess = (mu - mar) * periods
    ann_downside = downside_std * math.sqrt(periods)
    return float(ann_mu_excess / ann_downside)

def zscore(x: pd.Series, lookback: int = 252) -> pd.Series:
    m = x.rolling(lookback).mean()
    s = x.rolling(lookback).std(ddof=1)
    return (x - m) / s

def clip_weights(w: np.ndarray, wmin: np.ndarray, wmax: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(w, wmin), wmax)

def simplex_project(w: np.ndarray) -> np.ndarray:
    """Project weights onto simplex sum(w)=1, w>=0 (simple, robust)."""
    w = np.asarray(w, dtype=float)
    if np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-10:
        return w
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        # fallback
        w = np.clip(w, 0, None)
        s = w.sum()
        return w / s if s > 0 else np.ones_like(w) / len(w)
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(w - theta, 0.0)
    return w


# ============================================================
# IBKR Historical Data Adapters
# ============================================================

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
        """
        Return DataFrame indexed by date with columns=symbols containing adjusted close if possible.
        """

@dataclass
class IBSyncAdapter:
    """
    Adapter using ib_insync.
    """
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 12

    def _connect(self):
        from ib_insync import IB
        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id)
        return ib

    def hist_prices(
        self,
        symbols: List[str],
        end_dt: Optional[str] = None,
        duration: str = "3 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        from ib_insync import Stock, util

        ib = self._connect()
        try:
            out = {}
            for sym in symbols:
                c = Stock(sym, "SMART", "USD")
                bars = ib.reqHistoricalData(
                    c,
                    endDateTime=end_dt or "",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=1,
                )
                df = util.df(bars)
                if df.empty:
                    raise RuntimeError(f"No data returned for {sym}")
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                out[sym] = df["close"].rename(sym)
                time.sleep(0.2)  # be kind to the gateway
            px = pd.concat(out.values(), axis=1).sort_index()
            return px
        finally:
            ib.disconnect()


@dataclass
class IBAsyncAdapter:
    """
    Adapter using ib_async (if you prefer).
    """
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 13

    def hist_prices(
        self,
        symbols: List[str],
        end_dt: Optional[str] = None,
        duration: str = "3 Y",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        # ib_async API can vary by version; keep this adapter minimal and easy to patch.
        # If your ib_async wrapper differs, adjust here once and the rest of the engine stays stable.
        import asyncio
        from ib_async import IB, Stock  # type: ignore

        async def _run():
            ib = IB()
            await ib.connectAsync(self.host, self.port, clientId=self.client_id)
            try:
                out = {}
                for sym in symbols:
                    c = Stock(sym, "SMART", "USD")
                    bars = await ib.reqHistoricalDataAsync(
                        c,
                        endDateTime=end_dt or "",
                        durationStr=duration,
                        barSizeSetting=bar_size,
                        whatToShow=what_to_show,
                        useRTH=use_rth,
                        formatDate=1,
                    )
                    df = pd.DataFrame(bars)
                    if df.empty:
                        raise RuntimeError(f"No data returned for {sym}")
                    # expect columns: date, close, ...
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                    out[sym] = df["close"].rename(sym)
                    await asyncio.sleep(0.2)
                px = pd.concat(out.values(), axis=1).sort_index()
                return px
            finally:
                ib.disconnect()

        return asyncio.get_event_loop().run_until_complete(_run())


# ============================================================
# Spread Signal Providers (BB)
# ============================================================

class SpreadSignalProvider(Protocol):
    def get_bb_spread(self, start: Optional[pd.Timestamp] = None) -> pd.Series:
        """Return a daily series of BB spread (e.g., OAS) in % or bps (consistent)."""

@dataclass
class FRED_BB_OAS_Provider:
    """
    Best quality BB spread signal.
    Requires: pandas_datareader
    Uses FRED series: 'BAMLH0A1HYBB' (ICE BofA BB US High Yield OAS)
    """
    fred_series: str = "BAMLH0A1HYBB"

    def get_bb_spread(self, start: Optional[pd.Timestamp] = None) -> pd.Series:
        from pandas_datareader import data as pdr  # type: ignore
        s = pdr.DataReader(self.fred_series, "fred", start=start)
        s = s[self.fred_series].rename("bb_oas")
        s = s.dropna()
        return s

@dataclass
class IBKR_ETF_SpreadProxy_Provider:
    """
    IBKR-only fallback: build a proxy using HY ETF vs short Treasury ETF.

    Proxy idea:
      - Use log price ratio (HY ETF / short-bill ETF) and map to "spread regime"
      - Use rolling z-score of drawdown / relative performance to throttle HY exposure

    NOTE: This is NOT a true OAS; it's a regime proxy. Works surprisingly well as a risk switch.
    """
    data: IBKRDataAdapter
    hy_etf: str = "HYG"   # or "JNK"
    bill_etf: str = "BIL" # or "SHV"

    def get_bb_spread(self, start: Optional[pd.Timestamp] = None) -> pd.Series:
        px = self.data.hist_prices([self.hy_etf, self.bill_etf], duration="5 Y")
        px = px.dropna()
        rel = np.log(px[self.hy_etf]) - np.log(px[self.bill_etf])
        # Turn relative weakness into "wider spreads" proxy
        # Negative rel momentum => wider spreads
        proxy = -rel.diff().rolling(21).sum()
        proxy.name = "bb_spread_proxy"
        return proxy.dropna()


# ============================================================
# Tax model for Munis (WA)
# ============================================================

@dataclass
class TaxModelWA:
    """
    Washington state: no state income tax.
    Adjust munis on federal tax-equivalent basis.

    Inputs should be yields/returns in decimal, e.g. 0.03 for 3%.
    """
    federal_marginal_rate: float = 0.32  # user-set
    # NIIT, AMT, etc can be added if you want:
    net_investment_income_tax: float = 0.0  # e.g. 0.038 for 3.8%

    def effective_federal_rate(self) -> float:
        return min(max(self.federal_marginal_rate + self.net_investment_income_tax, 0.0), 0.60)

    def tax_equivalent_return(self, muni_return: pd.Series) -> pd.Series:
        """
        Convert muni returns to tax-equivalent returns:
          TE = muni / (1 - fed_rate)
        This is a simplification; for total return series it’s a practical approximation.
        """
        rate = self.effective_federal_rate()
        return muni_return / (1.0 - rate)

    def after_tax_return(self, taxable_return: pd.Series) -> pd.Series:
        """
        Convert taxable returns to after-tax returns.
        """
        rate = self.effective_federal_rate()
        return taxable_return * (1.0 - rate)


# ============================================================
# Sortino Optimizer (with constraints + turnover penalty)
# ============================================================

@dataclass
class SortinoOptimizerConfig:
    mar_daily: float = 0.0           # minimum acceptable daily return (e.g. rf_daily)
    downside_lookback: int = 252     # used for estimating downside risk stability
    allow_short: bool = False
    turnover_penalty: float = 0.0    # lambda * sum(|w - w_prev|)
    max_iter: int = 400

def optimize_sortino(
    returns: pd.DataFrame,
    cfg: SortinoOptimizerConfig,
    w0: Optional[np.ndarray] = None,
    w_prev: Optional[np.ndarray] = None,
    wmin: Optional[np.ndarray] = None,
    wmax: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Optimize weights to maximize Sortino ratio on a return matrix.

    We do this by minimizing negative Sortino with penalties and constraints:
      - sum(w)=1
      - bounds
      - optional turnover penalty
    """
    R = returns.dropna()
    if len(R) < max(60, cfg.downside_lookback // 4):
        raise ValueError("Not enough return history to optimize Sortino robustly.")

    n = R.shape[1]
    if w0 is None:
        w0 = np.ones(n) / n
    else:
        w0 = np.asarray(w0, float)

    if w_prev is None:
        w_prev = w0.copy()
    else:
        w_prev = np.asarray(w_prev, float)

    if wmin is None:
        wmin = np.zeros(n) if not cfg.allow_short else -np.ones(n)
    if wmax is None:
        wmax = np.ones(n)

    bounds = [(float(wmin[i]), float(wmax[i])) for i in range(n)]

    mar = cfg.mar_daily

    def portfolio_returns(w: np.ndarray) -> pd.Series:
        return pd.Series(R.values @ w, index=R.index, name="port")

    def objective(w: np.ndarray) -> float:
        w = np.asarray(w, float)

        # Enforce sum=1 softly, plus hard constraint below
        port = portfolio_returns(w)
        sr = sortino_ratio(port, mar=mar)
        if np.isnan(sr) or np.isinf(sr):
            base = 1e6
        else:
            base = -sr

        # Turnover penalty (L1 approx with abs)
        if cfg.turnover_penalty > 0:
            base += cfg.turnover_penalty * np.sum(np.abs(w - w_prev))

        return float(base)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        objective,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": cfg.max_iter, "ftol": 1e-9, "disp": False},
    )

    w = res.x if res.success else w0
    w = clip_weights(w, np.asarray(wmin), np.asarray(wmax))
    # Final ensure sum=1
    if not cfg.allow_short:
        w = simplex_project(w)
    else:
        # normalize to sum 1 even if shorts
        s = w.sum()
        w = w / s if abs(s) > 1e-12 else np.ones_like(w) / len(w)
    return w


# ============================================================
# Omega Framework Hook (optional)
# ============================================================

class OmegaOptimizerHook(Protocol):
    def optimize(self, returns: pd.DataFrame, initial_weights: np.ndarray, **kwargs) -> np.ndarray:
        """
        Implement in your Omega optimizer to accept this engine's prepared returns/features.
        """


# ============================================================
# Allocation Engine
# ============================================================

@dataclass
class EngineUniverse:
    muni: str = "MUB"     # muni ETF placeholder (you can swap to WA-specific muni fund if desired)
    bills_1m: str = "BIL" # 1-3 month T-bill ETF proxy
    hy_bb: str = "HYG"    # HY ETF proxy for BB sleeve (if you use a BB corporate ETF, swap it)

@dataclass
class AllocationEngineConfig:
    # Data
    duration: str = "5 Y"
    bar_size: str = "1 day"
    use_rth: bool = True

    # Tax
    tax: TaxModelWA = TaxModelWA(0.32, 0.0)

    # Signal
    spread_z_lookback: int = 252
    spread_widen_threshold: float = 0.75   # zscore above => risk-off for HY
    spread_tight_threshold: float = -0.75  # zscore below => risk-on for HY
    hy_risk_off_scale: float = 0.35        # multiply HY max weight by this in risk-off
    hy_risk_on_boost: float = 1.20         # multiply HY max weight by this in risk-on (capped by global max)

    # Optimization
    opt: SortinoOptimizerConfig = SortinoOptimizerConfig(mar_daily=0.0, turnover_penalty=0.05)
    rebalance_freq: str = "M"              # monthly

    # Constraints (defaults; tweak as you like)
    wmin: Dict[str, float] = None
    wmax: Dict[str, float] = None

    def __post_init__(self):
        if self.wmin is None:
            self.wmin = {"muni": 0.05, "bills": 0.05, "hy": 0.00}
        if self.wmax is None:
            self.wmax = {"muni": 0.80, "bills": 0.90, "hy": 0.60}

@dataclass
class AllocationEngine:
    data: IBKRDataAdapter
    spread_provider: SpreadSignalProvider
    universe: EngineUniverse = EngineUniverse()
    cfg: AllocationEngineConfig = AllocationEngineConfig()
    omega_hook: Optional[OmegaOptimizerHook] = None

    def _get_prices(self) -> pd.DataFrame:
        syms = [self.universe.muni, self.universe.bills_1m, self.universe.hy_bb]
        px = self.data.hist_prices(
            syms,
            duration=self.cfg.duration,
            bar_size=self.cfg.bar_size,
            use_rth=self.cfg.use_rth,
        )
        px = px.dropna()
        return px

    def _compute_returns(self, px: pd.DataFrame) -> pd.DataFrame:
        rets = px.pct_change().dropna()
        rets.columns = ["muni", "bills", "hy"]
        return rets

    def _apply_tax_adjustments(self, rets: pd.DataFrame) -> pd.DataFrame:
        """
        WA muni: treat muni sleeve as federal tax-exempt. Convert to tax-equivalent return for
        *comparability* in optimization.
        Bills and HY are taxable -> optional: convert to after-tax if you want true after-tax optimization.

        Here we do:
          - muni: tax-equivalent (boost numerator)
          - bills, hy: after-tax (reduce taxable)
        This tends to reflect a WA investor’s true utility better than raw returns.
        """
        muni_te = self.cfg.tax.tax_equivalent_return(rets["muni"])
        bills_at = self.cfg.tax.after_tax_return(rets["bills"])
        hy_at = self.cfg.tax.after_tax_return(rets["hy"])
        out = pd.concat([muni_te.rename("muni"), bills_at.rename("bills"), hy_at.rename("hy")], axis=1)
        return out.dropna()

    def _spread_regime_adjusted_bounds(self, rets_idx: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create dynamic bounds for HY based on BB spread z-score.
        """
        spread = self.spread_provider.get_bb_spread(start=rets_idx.min() - pd.Timedelta(days=400))
        spread = spread.reindex(rets_idx).ffill().dropna()
        zs = zscore(spread, lookback=self.cfg.spread_z_lookback).reindex(rets_idx).ffill()

        # Use last available z-score to set current regime
        z_last = float(zs.iloc[-1]) if len(zs) else 0.0

        wmin = np.array([self.cfg.wmin["muni"], self.cfg.wmin["bills"], self.cfg.wmin["hy"]], dtype=float)
        wmax = np.array([self.cfg.wmax["muni"], self.cfg.wmax["bills"], self.cfg.wmax["hy"]], dtype=float)

        # Adjust HY cap based on regime
        hy_cap = wmax[2]
        if z_last >= self.cfg.spread_widen_threshold:
            hy_cap = hy_cap * self.cfg.hy_risk_off_scale
        elif z_last <= self.cfg.spread_tight_threshold:
            hy_cap = min(hy_cap * self.cfg.hy_risk_on_boost, wmax[2])

        wmax[2] = float(np.clip(hy_cap, wmin[2], self.cfg.wmax["hy"]))
        return wmin, wmax

    def target_weights(
        self,
        w_prev: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        One-shot compute target weights as of latest available data.
        """
        px = self._get_prices()
        rets = self._compute_returns(px)
        rets_adj = self._apply_tax_adjustments(rets)

        # Dynamic bounds based on BB spread regime
        wmin, wmax = self._spread_regime_adjusted_bounds(rets_adj.index)

        w_prev_vec = None
        if w_prev is not None:
            w_prev_vec = np.array([w_prev.get("muni", 0.0), w_prev.get("bills", 0.0), w_prev.get("hy", 0.0)], float)

        # Optimize
        w0 = (w_prev_vec if w_prev_vec is not None else np.ones(3) / 3.0)

        if self.omega_hook is not None:
            # Route into your Omega optimizer if desired
            w = self.omega_hook.optimize(rets_adj, initial_weights=w0, wmin=wmin, wmax=wmax)
            w = np.asarray(w, float)
        else:
            w = optimize_sortino(
                returns=rets_adj,
                cfg=self.cfg.opt,
                w0=w0,
                w_prev=w_prev_vec,
                wmin=wmin,
                wmax=wmax,
            )

        return {"muni": float(w[0]), "bills": float(w[1]), "hy": float(w[2])}

    def backtest_rebalance(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Simple monthly rebalance backtest on adjusted returns, using regime-aware HY caps.
        Returns a DataFrame with weights and portfolio returns.
        """
        px = self._get_prices()
        rets = self._compute_returns(px)
        rets_adj = self._apply_tax_adjustments(rets)

        if start:
            rets_adj = rets_adj.loc[pd.to_datetime(start):]
        if end:
            rets_adj = rets_adj.loc[:pd.to_datetime(end)]

        rb_dates = rets_adj.resample(self.cfg.rebalance_freq).last().index
        rb_dates = rb_dates.intersection(rets_adj.index)

        w_prev = initial_weights or {"muni": 1/3, "bills": 1/3, "hy": 1/3}
        weights_hist = []
        port_rets = []

        for dt in rets_adj.index:
            if dt in rb_dates:
                wmin, wmax = self._spread_regime_adjusted_bounds(rets_adj.loc[:dt].index)
                w_prev_vec = np.array([w_prev["muni"], w_prev["bills"], w_prev["hy"]], float)

                if self.omega_hook is not None:
                    w = self.omega_hook.optimize(rets_adj.loc[:dt], initial_weights=w_prev_vec, wmin=wmin, wmax=wmax)
                    w = np.asarray(w, float)
                else:
                    w = optimize_sortino(
                        returns=rets_adj.loc[:dt],
                        cfg=self.cfg.opt,
                        w0=w_prev_vec,
                        w_prev=w_prev_vec,
                        wmin=wmin,
                        wmax=wmax,
                    )

                w_prev = {"muni": float(w[0]), "bills": float(w[1]), "hy": float(w[2])}

            weights_hist.append([w_prev["muni"], w_prev["bills"], w_prev["hy"]])
            port_rets.append(float(rets_adj.loc[dt].values @ np.array(weights_hist[-1])))

        out = pd.DataFrame(weights_hist, index=rets_adj.index, columns=["w_muni", "w_bills", "w_hy"])
        out["port_ret"] = port_rets
        out["port_nav"] = (1.0 + out["port_ret"]).cumprod()
        return out


# ============================================================
# Example usage
# ============================================================

def example_live_target_weights():
    # Choose ONE data adapter
    data = IBSyncAdapter(host="127.0.0.1", port=7497, client_id=12)

    # Preferred BB spread signal: FRED (best)
    # If you insist on IBKR-only, swap to IBKR_ETF_SpreadProxy_Provider
    spread = FRED_BB_OAS_Provider(fred_series="BAMLH0A1HYBB")

    # Tax model for WA: no state income tax; set your federal marginal
    tax = TaxModelWA(federal_marginal_rate=0.32, net_investment_income_tax=0.0)

    cfg = AllocationEngineConfig(
        duration="5 Y",
        bar_size="1 day",
        use_rth=True,
        tax=tax,
        opt=SortinoOptimizerConfig(
            mar_daily=0.0,          # set to rf_daily if you want
            turnover_penalty=0.05,  # raise to reduce trading
            allow_short=False,
        ),
    )

    engine = AllocationEngine(
        data=data,
        spread_provider=spread,
        universe=EngineUniverse(muni="MUB", bills_1m="BIL", hy_bb="HYG"),
        cfg=cfg,
        omega_hook=None,  # plug your Omega hook here if desired
    )

    w = engine.target_weights(w_prev={"muni": 0.40, "bills": 0.40, "hy": 0.20})
    print("Target weights:", w)


if __name__ == "__main__":
    example_live_target_weights()