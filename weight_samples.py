from util import get_hostgal_range


class SampleWeighter:
    # Example usage:
    # >> sw = SampleWeighter(train_exgal["hostgal_photoz"], test_exgal["hostgal_photoz"])
    # >> train_exgal = calculate_weights(train_exgal, False)

    def __init__(self, train_exgal_hp, test_exgal_hp):
        train_exgal_hr = get_hostgal_range(train_exgal_hp)
        test_exgal_hr = get_hostgal_range(test_exgal_hp)

        train_hp_dist = (train_exgal_hr.value_counts() / len(train_exgal_hr)).to_dict()
        test_hp_dist = (test_exgal_hr.value_counts() / len(test_exgal_hr)).to_dict()
        self.weight_list = [test_hp_dist[i] / train_hp_dist[i] for i in range(train_exgal_hr.max() + 1)]

    def calculate_weights(self, df, is_galactic):
        # gives weights so that test set hostgal_photoz distribution is represented
        if is_galactic:
            df["sample_weight"] = 1.0
        else:
            df["sample_weight"] = get_hostgal_range(df["hostgal_range"]).apply(lambda x: self.weight_list[x])

        # gives more weights to non-ddf because they are more common in test set
        df["sample_weight"] *= (2 - df["ddf"])

        # normalizes the weights so that each class has total sum of 100 (effecting training equally)
        df["sample_weight"] *= 100 / df.groupby("target")["sample_weight"].transform("sum")

        # doubles weights for class 15 and class 64
        df["sample_weight"] *= df["target"].apply(lambda x: 1 + (x in {15, 64}))
        return df
