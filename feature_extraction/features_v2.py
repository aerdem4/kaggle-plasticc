import numpy as np
import pandas as pd
from util import get_fe_argparser, none_or_one


def extract_features_v2(df):
    df["mjd_int"] = df["mjd"].astype(int)

    df = df[df["detected"] == 1].groupby(["object_id", "mjd_int"])["flux"].max().reset_index()
    df["abs_flux"] = np.abs(df["flux"])
    for col in ["first", "last", "deep", "peak"]:
        df[col] = df["flux"].values

    df["mjd_min"] = df["mjd_int"].values
    df["mjd_max"] = df["mjd_int"].values
    max_flux = df.groupby("object_id")["flux"].transform("max")
    df["mjd_peak"] = df["mjd_int"] * (max_flux == df["flux"])
    df["mjd_deep"] = df["mjd_int"] * (df.groupby("object_id")["flux"].transform("min") == df["flux"])

    peak_time = df.groupby("object_id")["mjd_peak"].transform("max")
    period = ((df["mjd_int"] > peak_time) & (df["mjd_int"] < peak_time + 32)).astype(int)
    df["peak_32"] = (none_or_one(period) * df["flux"]) / max_flux

    df = df.groupby("object_id").agg({"abs_flux": "max", "first": "first", "last": "last", "mjd_int": "count",
                                      "peak": lambda ll: np.array(ll).argmax(),
                                      "deep": lambda ll: np.array(ll).argmin(),
                                      "mjd_min": "min", "mjd_max": "max", "mjd_peak": "max", "mjd_deep": "max",
                                      "peak_32": "min"}).reset_index()
    df["first"] /= df["abs_flux"]
    df["last"] /= df["abs_flux"]
    df["peak"] /= df["mjd_int"] - 1
    df["deep"] /= df["mjd_int"] - 1
    df["till_peak"] = df["mjd_peak"] - df["mjd_min"]
    df["after_peak"] = df["mjd_max"] - df["mjd_peak"]
    df["deep_peak"] = df["mjd_peak"] - df["mjd_deep"]

    extracted_features = ["first", "last", "peak", "deep", "till_peak", "after_peak", "deep_peak", "peak_32"]

    return df[["object_id"] + extracted_features]


if __name__ == "__main__":
    args = get_fe_argparser("Generate features on detected flux without passband.")
    lc_df = pd.read_csv(args.light_curve_path)
    extract_features_v2(lc_df).to_csv(args.output_path, index=False)
