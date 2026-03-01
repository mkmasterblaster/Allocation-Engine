Allocation_Engine2-27-2026

The engine is written to allocation_engine.py and all tests pass. Here's what was built:

What's in the new engine
3 plug-and-play optimizers (all implement PortfolioOptimizer protocol)
Optimizer
Objective
Best for
SortinoOptimizer
Maximize annualized Sortino ratio
Downside-deviation focus
OmegaOptimizer
Maximize Omega ratio (Keating-Shadwick 2002)
Full distribution, no normality assumption
OmegaSortinoOptimizer
α·Ω + (1-α)·Sortino, scale-normalized
Recommended default — captures both
Swap with one line: build_engine(optimizer_type="omega").
Architecture improvements over the original
    • hy_risk_on_boost bug fixed → HYCapConfig.risk_on_ceiling=0.75 can now genuinely exceed the neutral cap (0.60), enabling real risk-on positioning
    • Positional column rename fixed → explicit symbol_to_name dict mapping
    • Spread fetched once before the backtest loop, not N times (major perf fix)
    • Z-score series computed once, then indexed per date — O(1) lookup per bar
    • asyncio.run() replaces deprecated get_event_loop()
    • "ME" rebalance freq (pandas 2.2+ compliant)
    • Optimizer failures log a warning with message, then fall back gracefully
Key classes to use
from allocation_engine import build_engine, BacktestEngine, print_performance_report

# Live weights
engine = build_engine(optimizer_type="omega_sortino", federal_marginal_rate=0.32, niit=0.038)
weights = engine.target_weights(w_prev={"muni": 0.40, "bills": 0.30, "hy": 0.30})

# Backtest
bt = BacktestEngine(engine).run(start="2020-01-01")
print_performance_report(bt)

# Compare all three optimizers
compare_optimizers(lambda opt: build_engine(optimizer_type=opt), start="2020-01-01")
