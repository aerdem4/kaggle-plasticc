from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from util import prepare_data, submit
from meta_lr import get_meta_preds

BATCH_SIZE = 256
NUM_FOLDS = 5
NUM_ITERS = 2


def evaluate(train_gal, train_exgal, oof_preds_gal, oof_preds_exgal, ohe_gal, ohe_exgal):
    gal_loss = log_loss(ohe_gal.transform(train_gal["target"].values.reshape(-1, 1)),
                        np.clip(np.round(oof_preds_gal, 4), 1e-4, None),
                        sample_weight=train_gal["sample_weight"])
    exgal_loss = log_loss(ohe_exgal.transform(train_exgal["target"].values.reshape(-1, 1)),
                          np.clip(np.round(oof_preds_exgal, 4), 1e-4, None),
                          sample_weight=train_exgal["sample_weight"])
    print("Galactic CV: {}".format(gal_loss))
    print("Extragalactic CV: {}".format(exgal_loss))
    print("Overall CV: {}".format((5/16)*gal_loss + (11/16)*exgal_loss))


def nn_preprocess(df):
    return np.hstack([df.fillna(-1).values, np.log(np.abs(df) + 0.01).fillna(-1).values])


def clip_outliers(array):
    for i in range(array.shape[1]):
        array[:, i] = np.clip(array[:, i], -2, 2)
    return array


def nn_block(input_layer, size, dropout_rate, activation):
    out_layer = Dense(size, activation=None)(input_layer)
    out_layer = BatchNormalization()(out_layer)
    out_layer = Activation(activation)(out_layer)
    out_layer = Dropout(dropout_rate)(out_layer)
    return out_layer


def train_and_predict(train_df, test_df, features, ohe):
    num_classes = ohe.n_values_[0]
    oof_preds = np.zeros((len(train_df), num_classes))

    ss = StandardScaler()
    ss.fit(nn_preprocess(train_df[features]))

    test_preds = np.zeros((len(test_df), num_classes))
    X_test = nn_preprocess(test_df[features])
    X_test = ss.transform(X_test)
    X_test = clip_outliers(X_test)

    for seed in range(NUM_ITERS):
        print("Iteration", seed)
        skf = StratifiedKFold(NUM_FOLDS, shuffle=True, random_state=seed)

        for train_index, val_index in skf.split(train_df, train_df["target"]):

            dev_df, val_df = train_df.iloc[train_index], train_df.iloc[val_index]
            dev_df["sample_weight"] *= dev_df.shape[0] / dev_df["sample_weight"].sum()
            val_df["sample_weight"] *= val_df.shape[0] / val_df["sample_weight"].sum()

            X_train, y_train = nn_preprocess(dev_df[features]), ohe.transform(dev_df["target"].values.reshape(-1, 1))
            X_val, y_val = nn_preprocess(val_df[features]), ohe.transform(val_df["target"].values.reshape(-1, 1))

            X_train = ss.transform(X_train)
            X_val = ss.transform(X_val)
            X_train = clip_outliers(X_train)
            X_val = clip_outliers(X_val)

            dense_input = Input(shape=(X_train.shape[1],))
            hidden_layer = nn_block(dense_input, 2 ** 10, 0.5, "relu")
            hidden_layer = nn_block(hidden_layer, 2 ** 8, 0.2, "relu")
            hidden_layer = nn_block(hidden_layer, 2 ** 6, 0.1, "relu")
            out = Dense(num_classes, activation="softmax")(hidden_layer)

            model = Model(inputs=[dense_input], outputs=out)
            model.compile(loss="categorical_crossentropy", optimizer="adam")

            print("FOLD")

            early_stopping = EarlyStopping(monitor="val_loss", patience=10)
            best_model_path = "best_model.h5"
            model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

            hist = model.fit([X_train], y_train, batch_size=BATCH_SIZE, epochs=200,
                             sample_weight=dev_df["sample_weight"].values,
                             validation_data=([X_val], y_val, val_df["sample_weight"].values),
                             callbacks=[early_stopping, model_checkpoint], verbose=0)
            model.load_weights(best_model_path)
            print("train loss:", min(hist.history["loss"]))
            print("validation loss:", min(hist.history["val_loss"]))

            oof_preds[val_index, :] += model.predict([X_val], batch_size=BATCH_SIZE) / NUM_ITERS

            test_preds += model.predict([X_test], batch_size=BATCH_SIZE * 100) / (NUM_FOLDS * NUM_ITERS)

    return oof_preds, test_preds


def get_nn_predictions(train_gal, train_exgal, test_gal, test_exgal):
    bazin = ["A", "B", "tfall", "trise", "fit_error"]
    f_flux = ["flux_sn" + str(i) for i in range(6)]
    f_skew = ["skew" + str(i) for i in range(6)]
    f_f = ["f" + str(i) for i in range(6)]
    f_d = ["d" + str(i) for i in range(6)]
    f_dd = ["dd" + str(i) for i in range(6)]

    # use stacking features if they exist
    stack_features_gal = [col for col in train_gal.columns if col.startswith("lgb_pred")]
    stack_features_exgal = [col for col in train_exgal.columns if col.startswith("lgb_pred")]

    features_gal = ["mwebv", "flux", "flux_err", "fake_flux", "first", "last", "peak", "deep",
                    "total_detected", "ratio_detected", "observation_count",
                    "std_flux", "min_flux", "max_flux", "delta_flux", "detected_flux",
                    "time_diff_pos", "time_diff_neg"] + f_flux + f_f + f_d + f_dd + f_skew + bazin + stack_features_gal
    features_gal = features_gal + ["time_diff_full", "detected_period"] + ["raw_flux" + str(i) for i in range(6)]

    features_exgal = ["hostgal_calc", "hostgal_photoz", "hostgal_photoz_err", "mwebv",
                      "fake_flux", "first", "last", "peak", "deep",
                      "total_detected", "ratio_detected", "observation_count",
                      "std_flux", "max_flux", "detected_flux",
                      "time_diff_pos", "time_diff_neg"] + f_flux + f_d + bazin + stack_features_exgal
    features_exgal = features_exgal + ["time_diff_full", "detected_period"] + ["raw_flux" + str(i) for i in range(6)]

    ohe_gal = OneHotEncoder(sparse=False)
    ohe_gal.fit(train_gal["target"].values.reshape(-1, 1))
    ohe_exgal = OneHotEncoder(sparse=False)
    ohe_exgal.fit(train_exgal["target"].values.reshape(-1, 1))

    print("GALACTIC MODEL")
    oof_preds_gal, test_preds_gal = train_and_predict(train_gal, test_gal, features_gal, ohe_gal)
    print("EXTRAGALACTIC MODEL")
    oof_preds_exgal, test_preds_exgal = train_and_predict(train_exgal, test_exgal, features_exgal, ohe_exgal)

    evaluate(train_gal, train_exgal, oof_preds_gal, oof_preds_exgal, ohe_gal, ohe_exgal)

    return oof_preds_gal, oof_preds_exgal, test_preds_gal, test_preds_exgal


if __name__ == "__main__":
    train_gal, train_exgal, test_gal, test_exgal, gal_class_list, exgal_class_list, test_df = prepare_data()
    oof_preds_gal, oof_preds_exgal, test_preds_gal, test_preds_exgal = get_nn_predictions(train_gal, train_exgal,
                                                                                          test_gal, test_exgal)

    test_preds_gal = get_meta_preds(train_gal, oof_preds_gal, test_preds_gal, 0.2)
    test_preds_exgal = get_meta_preds(train_exgal, oof_preds_exgal, test_preds_exgal, 0.2)

    submit(test_df, test_preds_gal, test_preds_exgal,
           gal_class_list, exgal_class_list, "submissions/submission_nn.csv")
