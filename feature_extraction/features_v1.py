import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from util import get_fe_argparser, none_or_one


def extract_features_v1(df):
    df["std_flux"] = df["flux"].values
    df["min_flux"] = df["flux"].values
    df["max_flux"] = df["flux"].values
    df["detected_flux"] = df["flux"] * df["detected"]
    df["flux_sign"] = np.sign(df["flux"])
    df["observation_count"] = 1

    df["detected_mjd_max"] = df["mjd"] * none_or_one(df["detected"])
    df["detected_mjd_min"] = df["mjd"] * none_or_one(df["detected"])
    df["fake_flux"] = df["flux"] - np.sign(df["flux"]) * df["flux_err"]

    df["diff"] = df["flux"] - df.groupby(["object_id", "passband"])["flux"].shift(1)
    df["time_diff"] = df.groupby(["object_id", "detected", "flux_sign"])["mjd"].shift(-1) - df["mjd"]
    df["time_diff_pos"] = df["time_diff"] * none_or_one(df["detected"]) * (df["flux_sign"] == 1)
    df["time_diff_neg"] = df["time_diff"] * none_or_one(df["detected"]) * (df["flux_sign"] == -1)

    aggs = {"detected_mjd_max": "max", "detected_mjd_min": "min", "observation_count": "count",
            "flux": "median", "flux_err": "mean",
            "std_flux": "std", "min_flux": "min", "max_flux": "max",
            "detected_flux": "max", "time_diff_pos": "max", "time_diff_neg": "max", "time_diff": "max",
            "fake_flux": kurtosis}

    for i in range(6):
        df["raw_flux" + str(i)] = (df["fake_flux"]) * (df["passband"] == i)
        aggs["raw_flux" + str(i)] = "max"

        df["sn" + str(i)] = np.power(df["flux"] / df["flux_err"], 2) * (df["passband"] == i)
        aggs["sn" + str(i)] = "max"

        df["flux_sn" + str(i)] = df["flux"] * df["sn" + str(i)]
        aggs["flux_sn" + str(i)] = "max"

        df["skew" + str(i)] = (df["fake_flux"]) * ((df["passband"] == i) / (df["passband"] == i).astype(int))
        aggs["skew" + str(i)] = "skew"

        df["f" + str(i)] = (df["flux"]) * (df["passband"] == i)
        aggs["f" + str(i)] = "mean"

        df["d" + str(i)] = (df["detected"]) * (df["passband"] == i)
        aggs["d" + str(i)] = "sum"

        df["dd" + str(i)] = (df["diff"]) * (df["passband"] == i)
        aggs["dd" + str(i)] = "max"

    df = df.groupby("object_id").agg(aggs).reset_index()
    df = df.rename(columns={"detected": "total_detected"})
    df["time_diff_full"] = df["detected_mjd_max"] - df["detected_mjd_min"]
    df["detected_period"] = df["time_diff_full"] / df["total_detected"]
    df["ratio_detected"] = df["total_detected"] / df["observation_count"]
    df["delta_flux"] = df["max_flux"] - df["min_flux"]

    for col in ["sn", "flux_sn", "f", "dd"]:
        total_sum = df[[col + str(i) for i in range(6)]].sum(axis=1)
        for i in range(6):
            df[col + str(i)] /= total_sum

    return df


if __name__ == "__main__":
    args = get_fe_argparser("Generate features with passband.")
    lc_df = pd.read_csv(args.light_curve_path)
    extract_features_v1(lc_df).to_csv(args.output_path, index=False)
