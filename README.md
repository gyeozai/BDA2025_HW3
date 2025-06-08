# BDA2025_HW3

## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

## Part 1
- `eqw`: Equal Weight Portfolio  
- `rp`: Risk Parity Portfolio  
- `mv`: Mean-Variance Portfolio (Markowitz)

### Assignment Score
```bash
python Markowitz.py --score eqw
python Markowitz.py --score rp
python Markowitz.py --score mv
python Markowitz.py --score all
```

### Asset Allocation
- Visualize how asset weights change over time for each strategy.
```bash
python Markowitz.py --allocation eqw
python Markowitz.py --allocation rp
python Markowitz.py --allocation mv
```

### Performance and Evaluation
- Plot cumulative return for mean-variance portfolio
```bash
python Markowitz.py --performance mv
```
- Print performance metrics (Sharpe ratio, drawdown, etc.)
```bash
python Markowitz.py --report mv
```

## Part 2
- `mp`: My Portfolio (custom portfolio designed by you)
- `bmp`: Big My Portfolio (same but with more assets or data)

- My Implementation:
  - Mean-Variance Optimization + Momentum
  - My Configuration that passes the test:
    - `mp`: **lookback=380, gamma=0**
    - `bmp`: **lookback=980, gamma=0**
  - Alternatives that may bring out the better results:
    - Minimum Variance + Momentum
    - Black-Litterman Model + Momentum
    - use Dynamic Gamma
      - Volatility-adjusted Gamma
      - Regime-based Gamma

### Assignment Score
```bash
python Markowitz_2.py --score one    # Sharpe ratio > 1
python Markowitz_2.py --score spy    # Sharpe ratio > SPY benchmark
python Markowitz_2.py --score all
```

### Asset Allocation
```bash
python Markowitz_2.py --allocation mp
python Markowitz_2.py --allocation bmp
```

### Performance and Evaluation
```bash
python Markowitz_2.py --performance mp
python Markowitz_2.py --performance bmp

python Markowitz_2.py --report mp
python Markowitz_2.py --report bmp
```

### Cumulative Result
- Plot cumulative returns: `mp/bmp` vs SPY
```bash
python Markowitz_2.py --cumulative mp
python Markowitz_2.py --cumulative bmp
```


