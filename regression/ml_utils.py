import numpy as np
import pickle as pkl
import gzip
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import matplotlib.pyplot as plt

with gzip.open("data.pkl.gz", "rb") as f:   data = pkl.load(f)

features: dict[int: dict[str: float]] = [feature for feature in list(data["features"][8405].keys()) if feature not in ["dcd"]]
time_series: dict[int: dict[str: list[float]]] = [time_serie for time_serie in list(data["time_series"][8405].keys()) if time_serie not in ["alb", "rl"]]

del data

features_indices: dict[str: int] = {feature: indice for indice, feature in enumerate(features + time_series)}
indices_features: dict[str: int] = {indice: feature for indice, feature in enumerate(features + time_series)}

def vectorize_with_nan(data: dict[str: float | list[float]], t: int = 0) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    """
    vector: np.ndarray[float] = np.zeros(shape=(1, len(data)*2))
    
    for feature in features:
        if np.isnan(data[feature]): vector[0, 2*features_indices[feature]], vector[0, 2*features_indices[feature] + 1] = np.nan, 1
        else:   vector[0, 2*features_indices[feature]] = data[feature]

    for time_serie in time_series:
        if np.isnan(data[time_serie][t]): vector[0, 2*features_indices[time_serie]], vector[0, 2*features_indices[time_serie] + 1] = np.nan, 1
        else:   vector[0, 2*features_indices[time_serie]] = data[time_serie][t]

    Y: np.ndarray[float] = np.array([[np.nan_to_num(data["rl"][t]) + np.nan_to_num(data["alb"][t])]])

    return vector, Y


def fill_nan_with_median(X: np.ndarray[float]) -> np.ndarray[float]:
    """
    """

    X_filled = X.copy()
    T = X.shape[0]

    for t in range(T):
        medians = np.nanmedian(X_filled[t], axis=0)
        nan_mask = np.isnan(X_filled[t])
        X_filled[t][nan_mask] = np.take(medians, np.where(nan_mask)[1])

    return X_filled


def horizon(X: np.ndarray[float], Y: np.ndarray[float], h: int = 0) -> np.ndarray[float]:
    """
    """
    if h == 0:
        return X
    
    T, N, _ = X.shape

    Y_lags: np.ndarray[float] = np.zeros((T, N, h), dtype=X.dtype)

    for k in range(1, h + 1):   Y_lags[k:, :, k - 1] = Y[:-k, :, 0]

    return np.concatenate([X, Y_lags], axis=2)



def split_by_individuals(X, Y, test_size, seed=12):
    """
    """
    rng = np.random.default_rng(seed)
    N = X.shape[1]

    idx = np.arange(N)
    rng.shuffle(idx)

    n_test = int(test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train = X[:, train_idx, :]
    X_test  = X[:, test_idx, :]

    Y_train = Y[:, train_idx, :]
    Y_test  = Y[:, test_idx, :]

    return X_train, X_test, Y_train, Y_test


def auto_regressive_pred(model, X, h) -> np.ndarray[float]:
    """
    """
    last: np.ndarray[float] = np.zeros(shape=(1, X.shape[1], h))
    Xc: np.ndarray[float] = X.copy()
    pred: list[np.ndarray[float]] = []

    for t in range(X.shape[0]):
        
        pred_t: np.ndarray[float] = model.predict(
            np.concatenate([Xc[t][None, ...], last], axis=2)
        )

        pred.append(pred_t)

        last: np.ndarray[float] = np.concatenate([pred_t[None, :, None], last[:, :, :-1]], axis=2)

    return np.array(pred)


def plot_pred_true(Y_true, Y_pred, ax=None) -> None:
    """
    """
    if ax is None:  _, ax = plt.subplots()

    t = np.arange(Y_true.shape[0])

    ax.plot(t, Y_true, label="Y vraie", linewidth=2, alpha=0.8)

    ax.plot(t, Y_pred, label="Y prédite", linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlabel("Temps t")
    ax.set_xlabel("Valeur")
    ax.legend()

    return ax


def plot_metrics_bar(RESULTS: dict, split: str = "TEST", title: str | None = None):
    models = list(RESULTS.keys())
    metrics = list(next(iter(RESULTS.values()))[split].keys())

    n_models = len(models)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        1, n_metrics, 
        figsize=(4 * n_metrics, 5), 
        sharey=False
    )

    if n_metrics == 1:
        axes = [axes]  # sécurité si une seule métrique

    x = np.arange(n_models)
    width = 0.6

    for ax, metric in zip(axes, metrics):
        values = [RESULTS[model][split][metric] for model in models]

        ax.bar(x, values, width)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Valeur de la métrique")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()

    return None

class MedianImputer3D(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, strategy: str = "global", fill_value: str | float = 0.0) -> None:
        """
        """
        if strategy not in ("global", "per_series"):    raise ValueError("strategy must be 'global' or 'per_series'")
        self.strategy = strategy
        self.fill_value = fill_value

        return None
    

    def fit(self, X: np.ndarray[float], y = None) -> object:
        """
        """
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError("X must be a 3D array (T, N, D)")

        with np.errstate(all="ignore"):
            
            if self.strategy == "global":   medians = np.nanmedian(X, axis=(0, 1))

            elif self.strategy == "per_series":   medians = np.nanmedian(X, axis=0)

        medians = np.where(np.isnan(medians), self.fill_value, medians)

        self.medians_ = medians

        return self


    def transform(self, X: np.ndarray[float]) -> np.ndarray[float]:
        """
        """
        X = np.asarray(X)
        X_out = X.copy()

        if self.strategy == "global":
            T, N, D = X.shape
            for d in range(D):
                mask = np.isnan(X_out[:, :, d])
                X_out[:, :, d][mask] = self.medians_[d]

        elif self.strategy == "per_series":
            T, N, D = X.shape
            for n in range(N):
                for d in range(D):
                    mask = np.isnan(X_out[:, n, d])
                    X_out[:, n, d][mask] = self.medians_[n, d]

        return X_out


class Flatten3D(BaseEstimator, TransformerMixin):
    """
    Transforme (T, N, D) -> (T*N, D)
    """

    def fit(self, X: np.ndarray[float], y=None) -> object:
        """
        """
        X = np.asarray(X)
        self.T_ = X.shape[0]
        self.N_ = X.shape[1]
        self.D_ = X.shape[2]
        return self

    def transform(self, X: np.ndarray[float]) -> np.ndarray[float]:
        """
        """
        X = np.asarray(X)
        T, N, D = X.shape
        return X.reshape(T*N, D)
