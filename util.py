import numpy as np
import argparse
import pandas as pd
from weight_samples import SampleWeighter


def none_or_one(pd_series):
    return pd_series/pd_series


def safe_log(x):
    return np.log(np.clip(x, 1e-4, None))


def get_hostgal_range(hostgal_photoz):
    return np.clip(hostgal_photoz//0.2, 0, 6)


def get_fe_argparser(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("meta_path", action="store", default="input/test_set_metadata.csv")
    parser.add_argument("light_curve_path", action="store", default="input/test_set.csv")
    parser.add_argument("output_path", action="store", default="features/features_test.csv")
    return parser.parse_args()


def map_classes(df):
    class_list = df["target"].value_counts().index
    class_dict = {}
    for i, c in enumerate(class_list):
        class_dict[c] = i

    df["target"] = df["target"].map(class_dict)
    return df, class_list.tolist()


def prepare_data():
    train_df = pd.read_csv("input/training_set_metadata.csv")
    test_df = pd.read_csv("input/test_set_metadata.csv")

    for feature_file in ["bazin", "features_v1", "features_v2"]:
        train_df = train_df.merge(pd.read_csv("features/train_{}.csv".format(feature_file)),
                                  on="object_id", how="left")
        test_df = test_df.merge(pd.read_csv("features/test_{}.csv".format(feature_file)),
                                on="object_id", how="left")

    hostgal_calc_df = pd.read_csv("features/hostgal_calc.csv")
    train_df = train_df.merge(hostgal_calc_df, on="object_id", how="left")
    test_df = test_df.merge(hostgal_calc_df, on="object_id", how="left")

    train_gal = train_df[train_df["hostgal_photoz"] == 0].copy()
    train_exgal = train_df[train_df["hostgal_photoz"] > 0].copy()
    test_gal = test_df[test_df["hostgal_photoz"] == 0].copy()
    test_exgal = test_df[test_df["hostgal_photoz"] > 0].copy()

    sw = SampleWeighter(train_exgal["hostgal_photoz"], test_exgal["hostgal_photoz"])

    train_gal = sw.calculate_weights(train_gal, True)
    train_exgal = sw.calculate_weights(train_exgal, False)

    train_gal, gal_class_list = map_classes(train_gal)
    train_exgal, exgal_class_list = map_classes(train_exgal)
    return (train_gal, train_exgal, test_gal, test_exgal,
            gal_class_list, exgal_class_list,
            test_df[["object_id", "hostgal_photoz"]])


def is_labeled_as(preds, class_list, label):
    return preds.argmax(axis=1) == np.where(np.array(class_list) == label)[0]


def get_class99_proba(test_df, test_preds, all_classes):
    base = 0.02

    high99 = (get_hostgal_range(test_df["hostgal_photoz"]) == 0)

    low99 = is_labeled_as(test_preds, all_classes, 15)
    for label in [64, 67, 88, 90]:
        low99 = low99 | is_labeled_as(test_preds, all_classes, label)
    class99 = 0.22 - 0.18 * low99 + 0.13 * high99 - base

    not_sure = (test_preds.max(axis=1) < 0.9)
    filt = (test_df["hostgal_photoz"] > 0) & not_sure

    return (base + (class99 * filt).values).reshape(-1, 1)


def submit(test_df, test_preds_gal, test_preds_exgal, gal_class_list, exgal_class_list, sub_file):
    all_classes = gal_class_list.tolist() + exgal_class_list.tolist()

    gal_indices = np.where(test_df["hostgal_photoz"] == 0)[0]
    exgal_indices = np.where(test_df["hostgal_photoz"] >= 0)[0]

    test_preds = np.zeros((test_df.shape[0], len(all_classes)))
    test_preds[gal_indices, :] = np.hstack((np.clip(test_preds_gal, 1e-4, None),
                                            np.zeros((test_preds_gal.shape[0], len(exgal_class_list)))))
    test_preds[exgal_indices, :] = np.hstack((np.zeros((test_preds_exgal.shape[0], len(gal_class_list))),
                                              np.clip(test_preds_exgal, 1e-4, None)))

    estimated99 = get_class99_proba(test_df, test_preds, all_classes)

    sub_df = pd.DataFrame(index=test_df['object_id'], data=np.round(test_preds * (1 - estimated99), 4),
                          columns=['class_%d' % i for i in all_classes])
    sub_df["class_99"] = estimated99

    sub_df.to_csv(sub_file)
