import pandas as pd
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from models.nn import nn_preprocess, nn_block

BATCH_SIZE = 256
NUM_FOLDS = 5
NUM_ITERS = 2


def train_and_predict(train_df, test_df, features):
    oof_preds = np.zeros(len(train_df))

    ss = StandardScaler()
    ss.fit(nn_preprocess(train_df[features]))

    test_preds = np.zeros(len(test_df))
    X_test = nn_preprocess(test_df[features])
    X_test = ss.transform(X_test)

    for seed in range(NUM_ITERS):
        print("Iteration", seed)
        skf = KFold(NUM_FOLDS, shuffle=True, random_state=seed)

        for train_index, val_index in skf.split(train_df, train_df["hostgal_specz"]):

            dev_df, val_df = train_df.iloc[train_index], train_df.iloc[val_index]

            X_train, y_train = nn_preprocess(dev_df[features]), dev_df["hostgal_specz"]
            X_val, y_val = nn_preprocess(val_df[features]), val_df["hostgal_specz"]

            X_train = ss.transform(X_train)
            X_val = ss.transform(X_val)

            dense_input = Input(shape=(X_train.shape[1],))
            hidden_layer = nn_block(dense_input, 2 ** 10, 0.5, "relu")
            hidden_layer = nn_block(hidden_layer, 2 ** 8, 0.2, "relu")
            hidden_layer = nn_block(hidden_layer, 2 ** 6, 0.1, "relu")
            out = Dense(1, activation="linear")(hidden_layer)

            model = Model(inputs=[dense_input], outputs=out)
            model.compile(loss="mse", optimizer="adam")

            print("FOLD")

            early_stopping = EarlyStopping(monitor="val_loss", patience=3)
            best_model_path = "best_model.h5"
            model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

            hist = model.fit([X_train], y_train, batch_size=BATCH_SIZE, epochs=200,
                             validation_data=([X_val], y_val),
                             callbacks=[early_stopping, model_checkpoint], verbose=1)
            model.load_weights(best_model_path)
            print("train loss:", min(hist.history["loss"]))
            print("validation loss:", min(hist.history["val_loss"]))

            oof_preds[val_index] += model.predict([X_val], batch_size=BATCH_SIZE).ravel() / NUM_ITERS

            test_preds += model.predict([X_test], batch_size=BATCH_SIZE * 100).ravel() / (NUM_FOLDS * NUM_ITERS)

    return oof_preds, test_preds


if __name__ == "__main__":
    train_df = pd.read_csv("input/training_set_metadata.csv")
    test_df = pd.read_csv("input/test_set_metadata.csv")

    for feature_file in ["bazin", "features_v1", "features_v2"]:
        train_df = train_df.merge(pd.read_csv("features/train_{}.csv".format(feature_file)),
                                  on="object_id", how="left")
        test_df = test_df.merge(pd.read_csv("features/test_{}.csv".format(feature_file)),
                                on="object_id", how="left")

    full_df = train_df.append(test_df)
    print(full_df.shape)
    full_df = full_df[full_df["hostgal_photoz"] > 0]
    print(full_df.shape)
    train_df = full_df[full_df["hostgal_specz"].notnull()]
    full_df = full_df[full_df["hostgal_specz"].isnull()]
    print(full_df.shape, train_df.shape)

    bazin = ["A", "B", "tfall", "trise", "fit_error"]
    f_flux = ["flux_sn" + str(i) for i in range(6)]
    f_skew = ["skew" + str(i) for i in range(6)]
    f_f = ["f" + str(i) for i in range(6)]
    f_d = ["d" + str(i) for i in range(6)]
    f_dd = ["dd" + str(i) for i in range(6)]

    features = ["hostgal_calc", "hostgal_photoz", "hostgal_photoz_err", "mwebv",
                "fake_flux", "first", "last", "peak", "deep",
                "total_detected", "ratio_detected", "observation_count",
                "std_flux", "max_flux", "detected_flux",
                "time_diff_pos", "time_diff_neg"] + f_flux + f_d + bazin
    features = features + ["time_diff_full", "detected_period"] + ["raw_flux" + str(i) for i in range(6)]

    oof_preds, test_preds = train_and_predict(train_df, full_df, features)

    print(mean_squared_error(train_df["hostgal_specz"], train_df["hostgal_photoz"]))
    print(mean_squared_error(train_df["hostgal_specz"], np.clip(oof_preds, 0.0001, None)))

    print(spearmanr(train_df["hostgal_specz"], train_df["hostgal_photoz"]))
    print(spearmanr(train_df["hostgal_specz"], oof_preds))

    train_df["hostgal_calc"] = oof_preds
    full_df["hostgal_calc"] = test_preds

    full_df = full_df[["object_id", "hostgal_calc"]].append(train_df[["object_id", "hostgal_calc"]])
    full_df.to_csv("features/hostgal_calc.csv", index=False)
