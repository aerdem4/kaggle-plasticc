import numpy as np
from util import safe_log

MIN_PRED = 0.00002


def ensemble_submissions(subs, weights):
    """
    :param subs: list of pandas dataframe submissions
    :param weights: weights in ensembling which sum up to 1.0
    :return: ensemble submission pandas dataframe
    """
    df = subs[0].copy()
    num_submissions = len(subs)

    binary_mask = (subs[0] > 0)

    for i in range(1, num_submissions):
        binary_mask = binary_mask | (subs[i] > 0)

    for col in df.columns:
        df[col] = weights[0] * safe_log(subs[0][col])
        for i in range(1, num_submissions):
            df[col] += weights[i] * safe_log(subs[i][col])
        df[col] = np.exp(df[col]) + MIN_PRED

    df = df * binary_mask
    df = df.div(df.sum(axis=1), axis=0)
    return df
