from util import prepare_data, submit
from models.lgb import get_lgb_predictions
from models.nn import get_nn_predictions


if __name__ == "__main__":
    train_gal, train_exgal, test_gal, test_exgal, gal_class_list, exgal_class_list, test_df = prepare_data()
    lgb_oof_gal, lgb_oof_exgal, lgb_test_gal, lgb_test_exgal = get_lgb_predictions(train_gal, train_exgal,
                                                                                   test_gal, test_exgal)
    lgb_gal_preds = []
    for i in range(lgb_oof_gal.shape[1]):
        lgb_gal_preds.append("lgb_pred" + str(i))
        train_gal["lgb_pred" + str(i)] = lgb_oof_gal[:, i]
        test_gal["lgb_pred" + str(i)] = lgb_test_gal[:, i]

    lgb_exgal_preds = []
    for i in range(lgb_oof_exgal.shape[1]):
        lgb_exgal_preds.append("lgb_pred" + str(i))
        train_exgal["lgb_pred" + str(i)] = lgb_oof_exgal[:, i]
        test_exgal["lgb_pred" + str(i)] = lgb_test_exgal[:, i]

    oof_preds_gal, oof_preds_exgal, test_preds_gal, test_preds_exgal = get_nn_predictions(train_gal, train_exgal,
                                                                                          test_gal, test_exgal)

    submit(test_df, test_preds_gal, test_preds_exgal,
           gal_class_list, exgal_class_list, "submissions/stacking.csv")
