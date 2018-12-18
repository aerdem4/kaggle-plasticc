from sklearn.linear_model import LogisticRegression
from util import safe_log


# Example usage: test_preds_exgal = get_meta_preds(train_exgal, oof_preds_exgal, test_preds_exgal, 0.2)
def get_meta_preds(train_df, oof_preds, test_preds, C):
    lr = LogisticRegression(C=C, intercept_scaling=0.1, multi_class="multinomial", solver="lbfgs")
    lr.fit(safe_log(oof_preds), train_df["target"], sample_weight=train_df["sample_weight"])
    return lr.predict_proba(safe_log(test_preds))
