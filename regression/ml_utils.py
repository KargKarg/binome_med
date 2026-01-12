import numpy as np
import pickle as pkl
import gzip

with gzip.open("data.pkl.gz", "rb") as f:   data = pkl.load(f)

features: dict[int: dict[str: float]] = list(data["features"][8405].keys())
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

    Y_lagged: np.ndarray[np.nan | float] = np.full((T, N, h), np.nan)

    for k in range(1, h + 1):   Y_lagged[k:, :, k - 1] = Y[:-k, :, 0]

    return np.concatenate([X, Y_lagged], axis=2)


def split_by_individuals(X, Y, test_size=0.2, seed=12):
    """
    """
    rng = np.random.default_rng(seed)
    N = X.shape[1]

    idx = np.arange(N)
    rng.shuffle(idx)

    n_test = int(test_size * N)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train = X[:, train_idx, :]
    X_test  = X[:, test_idx, :]

    Y_train = Y[:, train_idx, :]
    Y_test  = Y[:, test_idx, :]

    return X_train, X_test, Y_train, Y_test
