A Python-based trading system that identifies market regimes and applies a different strategy to each. Rather than using a single rule set across all market conditions, the system first classifies what kind of market it's in, then selects the appropriate strategy for that regime.

How It Works
1. Data Pipeline
Historical OHLCV data is pulled via yfinance. Daily log returns and rolling volatility (20-day standard deviation of returns) are computed and used as the two input features to the model.
2. Regime Detection — Hidden Markov Model
A Gaussian HMM (via hmmlearn) is fit to the return/volatility feature matrix. The model learns three latent states, which are then mapped to bull, bear, and sideways regimes by inspecting the mean return and volatility of each state. The HMM uses the Baum-Welch algorithm (Expectation-Maximization) to estimate transition probabilities and emission parameters, and the Viterbi algorithm to decode the most likely regime sequence given observed data.
3. Regime-Specific Strategies

Bull: Momentum strategy (go long when price is above its 50-day moving average)
Bear: Risk-off (move to cash or take a short position)
Sideways: Mean-reversion (buy when price dips below the lower Bollinger Band, sell when it reverts to the mean)

4. Walk-Forward Validation
The system is evaluated using walk-forward testing rather than a single train/test split. A rolling window trains the HMM on a fixed historical period, generates signals on the following out-of-sample window, then steps forward. This prevents lookahead bias and gives a realistic picture of how the strategy would have performed in live trading.
5. Performance Metrics
Each regime's strategy is evaluated on Sharpe ratio, max drawdown, total return, and win rate. Results are broken down per regime so you can see where the system adds value and where it doesn't.
6. Dashboard
The Streamlit dashboard displays the detected regime over time, equity curves per strategy, and summary performance metrics. Regime labels are color-coded on the price chart so the classification is visually interpretable.

Stack

yfinance — data
hmmlearn — regime detection
pandas / numpy — feature engineering and backtesting
matplotlib / plotly — charting
streamlit — frontend
