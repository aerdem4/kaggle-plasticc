import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgb
from util import prepare_data, submit
from meta_lr import get_meta_preds

NUM_FOLDS = 5


def evaluate(train_gal, train_exgal, oof_preds_gal, oof_preds_exgal):
    gal_loss = log_loss(train_gal["target"], np.round(oof_preds_gal, 4),
                        sample_weight=train_gal["sample_weight"])
    exgal_loss = log_loss(train_exgal["target"], np.round(oof_preds_exgal, 4),
                          sample_weight=train_exgal["sample_weight"])
    print("Galactic CV: {}".format(gal_loss))
    print("Extragalactic CV: {}".format(exgal_loss))
    print("Overall CV: {}".format((5/16)*gal_loss + (11/16)*exgal_loss))


def train_and_predict(train_df, test_df, features, params):
    oof_preds = np.zeros((len(train_df), params["num_class"]))
    test_preds = np.zeros((len(test_df), params["num_class"]))

    skf = StratifiedKFold(NUM_FOLDS, random_state=4)

    for train_index, val_index in skf.split(train_df, train_df["target"]):
        dev_df, val_df = train_df.iloc[train_index], train_df.iloc[val_index]
        lgb_train = lgb.Dataset(dev_df[features], dev_df["target"], weight=dev_df["sample_weight"])
        lgb_val = lgb.Dataset(val_df[features], val_df["target"], weight=val_df["sample_weight"])

        model = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=[lgb_train, lgb_val],
                          early_stopping_rounds=10, verbose_eval=50)
        oof_preds[val_index, :] = model.predict(val_df[features])

        test_preds += model.predict(test_df[features]) / NUM_FOLDS

    return oof_preds, test_preds


def get_lgb_predictions(train_gal, train_exgal, test_gal, test_exgal):
    bazin = ["A", "B", "tfall", "trise", "cc", "fit_error", "t0_shift"]
    f_flux = ["flux_sn" + str(i) for i in range(6)] + ["sn" + str(i) for i in range(6)]
    f_skew = ["skew" + str(i) for i in range(6)]
    f_f = ["f" + str(i) for i in range(6)]
    f_d = ["d" + str(i) for i in range(6)]
    f_dd = ["dd" + str(i) for i in range(6)]
    v3_features = ['first', 'last', 'peak', 'deep', 'till_peak', 'after_peak', 'deep_peak', 'peak_32']
    peak_time = ["peak_time" + str(i) for i in [0, 1, 4, 5]]

    features_gal = ['mwebv', 'flux', 'flux_err', 'fake_flux',
                    'total_detected', 'ratio_detected', 'observation_count',
                    'std_flux', 'min_flux', 'max_flux', 'delta_flux', 'detected_flux',
                    'time_diff_pos', 'time_diff_neg'] + f_flux + f_skew + f_f + f_d + f_dd + v3_features + bazin

    features_exgal = ['hostgal_photoz', 'hostgal_photoz_err', 'hostgal_calc', 'mwebv', 'fake_flux',
                      'time_diff_pos',
                      'time_diff_neg'] + f_flux + f_skew + f_f + f_d + v3_features + bazin + peak_time

    params_gal = {"objective": "multiclass",
                  "num_class": len(gal_class_list),
                  "min_data_in_leaf": 200,
                  "num_leaves": 5,
                  "feature_fraction": 0.7
                  }

    params_exgal = {"objective": "multiclass",
                    "num_class": len(exgal_class_list),
                    "min_data_in_leaf": 200,
                    "num_leaves": 5,
                    "feature_fraction": 0.7
                    }

    print("GALACTIC MODEL")
    oof_preds_gal, test_preds_gal = train_and_predict(train_gal, test_gal, features_gal, params_gal)
    print("EXTRAGALACTIC MODEL")
    oof_preds_exgal, test_preds_exgal = train_and_predict(train_exgal, test_exgal, features_exgal, params_exgal)

    evaluate(train_gal, train_exgal, oof_preds_gal, oof_preds_exgal)

    return oof_preds_gal, oof_preds_exgal, test_preds_gal, test_preds_exgal


if __name__ == "__main__":
    train_gal, train_exgal, test_gal, test_exgal, gal_class_list, exgal_class_list, test_df = prepare_data()
    oof_preds_gal, oof_preds_exgal, test_preds_gal, test_preds_exgal = get_lgb_predictions(train_gal, train_exgal,
                                                                                           test_gal, test_exgal)

    test_preds_gal = get_meta_preds(train_gal, oof_preds_gal, test_preds_gal, 0.2)
    test_preds_exgal = get_meta_preds(train_exgal, oof_preds_exgal, test_preds_exgal, 0.2)

    submit(test_df, test_preds_gal, test_preds_exgal,
           gal_class_list, exgal_class_list, "submissions/submission_lgb.csv")
