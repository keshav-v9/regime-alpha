I built a Python-based trading system that tries to adapt to different market conditions instead of using the same strategy all the time. The main idea is to first identify whether the market is in a bull, bear, or sideways regime, and then use a strategy that makes more sense for that environment.

### How It Works

**1. Data Pipeline**
The system pulls historical OHLCV data using `yfinance`. From the price data, I calculate daily log returns and 20-day rolling volatility, which are used as the main features for identifying market regimes.

**2. Regime Detection**
To classify market conditions, I use a Gaussian Hidden Markov Model through `hmmlearn`. The model finds three underlying market states based on patterns in returns and volatility. After training, I look at the characteristics of each state and label them as bull, bear, or sideways.

Under the hood, the HMM uses the Baum-Welch algorithm to learn parameters such as state transition probabilities, while the Viterbi algorithm is used to determine the most likely sequence of market regimes.

**3. Trading Strategies**
Once the current regime is identified, the system applies a different strategy:

* **Bull market:** Uses momentum and takes a long position when the price is above its 50-day moving average.
* **Bear market:** Takes a risk-off approach by moving to cash or optionally taking a short position.
* **Sideways market:** Uses mean reversion with Bollinger Bands, buying when the price falls below the lower band and exiting when it moves back toward the average.

**4. Walk-Forward Testing**
Instead of training and testing the model on one fixed split of the data, I use walk-forward validation. The HMM is trained on a historical window and then tested on the next unseen period. The window then moves forward and the process repeats.

I chose this approach because it better represents how the system would work in practice and helps avoid accidentally using future information when generating signals.

**5. Performance Evaluation**
I evaluate the strategies using metrics including Sharpe ratio, maximum drawdown, total return, and win rate. I also break the results down by market regime so I can see which strategies perform well under different conditions rather than only looking at overall returns.

**6. Dashboard**
I built a Streamlit dashboard to make the results easier to explore. It shows historical prices alongside the detected market regimes, strategy equity curves, and key performance metrics. The regimes are also highlighted on the price chart so it is easy to see how the model's classifications line up with actual market movements.

### Stack

* Python
* `yfinance` for market data
* `hmmlearn` for the Hidden Markov Model
* `pandas` and `numpy` for data processing and backtesting
* `matplotlib` and `plotly` for visualizations
* `streamlit` for the dashboard
