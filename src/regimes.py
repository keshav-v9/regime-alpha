import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class RegimeDetector:
    def __init__(self, n_regimes: int = 3, n_iter: int = 1000, random_state: int = 42):
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        self._label_map: dict = {}

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        features = df[["log_return", "volatility"]].dropna().values
        self.model.fit(features)
        self._label_map = self._build_label_map(df)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        features = df[["log_return", "volatility"]].dropna().values
        raw = self.model.predict(features)
        labels = pd.Series(raw, index=df.dropna(subset=["log_return", "volatility"]).index)
        return labels.map(self._label_map)

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).predict(df)

    def _build_label_map(self, df: pd.DataFrame) -> dict:
        """Map raw HMM state integers → 'bull' / 'bear' / 'sideways' by avg return."""
        features = df[["log_return", "volatility"]].dropna().values
        raw_states = self.model.predict(features)
        temp = df.dropna(subset=["log_return", "volatility"]).copy()
        temp["_state"] = raw_states
        avg_return = temp.groupby("_state")["log_return"].mean()

        sorted_states = avg_return.sort_values(ascending=False).index.tolist()
        names = ["bull", "sideways", "bear"] if self.n_regimes == 3 else [str(i) for i in range(self.n_regimes)]
        return {state: names[i] for i, state in enumerate(sorted_states)}
