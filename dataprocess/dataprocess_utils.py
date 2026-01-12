import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from numpy import nan


def load_all_IDs() -> list[int]:
    """
    Return a $list$ of patients IDs ($int$).
    """

    with open("raw/patients.txt", "r") as IDs:
        
        list_IDs: list[int] = list(map(int, IDs.read().replace(" ", "").split(",")))

    return list_IDs


def load_all_DFs(ftype: str = "raw") -> pd.core.frame.DataFrame:
    """
    Return the $DataFrame$ of the concerned :fic:.
    """
    all_DFs: dict[str: pd.core.frame.DataFrame] = {fic[:-4]: pd.read_csv(f"{ftype}/{fic}", sep=";", encoding="utf-8") for fic in os.listdir("raw") if fic[-3:] == "csv"}
    cols_hd129: pd.core.indexes.base.Index = all_DFs["hd129"].columns
    all_DFs["hd129"]: pd.core.frame.DataFrame = all_DFs["hd129"].merge(all_DFs["s129"], on="ID", how="inner").assign(heure=lambda df: (df["heure"] - df["dadmi"]) * 60).query("heure >= 0")[cols_hd129].rename(columns={"heure": "minadmin"})
    all_DFs["urine"]: pd.core.frame.DataFrame = all_DFs["urine"].query("minadmin >= 0")
    return all_DFs


def forward_fill_values(time_series: dict[int, dict[str, list[float | None]]]) -> None:
    """
    in place modification of :time_series:
    """
    for _, col_dict in time_series.items():
        for _, values in col_dict.items():
            last_value = nan
            for i, val in enumerate(values):
                if val is nan or (isinstance(val, float) and math.isnan(val)):
                    if last_value is not nan:
                        values[i] = last_value
                else:
                    last_value = val
    return None


def load_time_series() -> dict[int: dict[str: list[float]]]:
    """
    """

    IDs: list[int] = load_all_IDs()
    all_DFs: pd.core.frame.DataFrame = load_all_DFs()

    series_dtypes: list[str] = ["adr", "dob", "hd129", "nad", "rl129", "urine"]

    upper: int = int(max([max(all_DFs[dtype]["minadmin"]) for dtype in series_dtypes]))

    time_series: dict[int: dict[str: list[float]]] = {pid: {col: [nan for _ in range(upper + 1)] for col in set().union(*[all_DFs[dtype].columns for dtype in series_dtypes]) if col not in ["ID", "minadmin"] } for pid in IDs}


    for dtype in series_dtypes:

        df_sorted: pd.core.frame.DataFrame = all_DFs[dtype].sort_values(by=["ID", "minadmin"])

        columns: list[str] = list(df_sorted.columns)
        columns.remove("ID")
        columns.remove("minadmin")

        for _, row in df_sorted.iterrows():
            if not row["ID"] in IDs: continue
            for col in columns: time_series[row["ID"]][col][int(row["minadmin"])] = row[col]
    
    forward_fill_values(time_series)

    return time_series


def load_features() -> dict[int: dict[str: float]]:
    """
    """

    IDs: list[int] = load_all_IDs()
    all_DFs: pd.core.frame.DataFrame = load_all_DFs()

    features_dtypes: list[str] = ["pds128", "prehrl126", "s129"]

    C: dict[str: [str]] = {dtype: (with_id := list(all_DFs[dtype].columns), with_id.remove("ID"), with_id)[-1] for dtype in features_dtypes}

    features: dict[int: dict[str: list[float]]] = {
        pid: {
            col: all_DFs[dtype].loc[all_DFs[dtype]["ID"] == pid, col].iloc[0]

            if not all_DFs[dtype].loc[all_DFs[dtype]["ID"] == pid].empty
            else nan

            for dtype in features_dtypes
            for col in C[dtype]
    
        }
        for pid in IDs
    }

    for pid in IDs: features[pid]["sexe"] = 1 if features[pid]["sexe"] == "M" else 0

    return features


def time_series_plot(values: list[float]) -> None:
    """
    """
    plt.figure(figsize=(10, 5))
    plt.plot(values, marker='o')
    plt.xlim(0, len(values))
    plt.xlabel("minadmin")
    plt.ylabel("value")
    plt.grid(True)
    plt.show()
    plt.close()

    return None


def data_plot(data: list[float]) -> None:
    """
    """
    plt.figure()
    plt.hist(data, bins="auto")
    plt.xlabel("Valeurs")
    plt.ylabel("Fréquence")
    plt.title("Histogramme des données")
    plt.show()
    plt.close()

    return None