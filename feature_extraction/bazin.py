import multiprocessing
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import numba
from util import get_fe_argparser


NUM_PARTITIONS = 500
LOW_PASSBAND_LIMIT = 3
FEATURES = ["A", "B", "t0", "tfall", "trise", "cc", "fit_error", "status", "t0_shift"]


# bazin, errorfunc and fit_scipy are developed using:
# https://github.com/COINtoolbox/ActSNClass/blob/master/examples/1_fit_LC/fit_lc_parametric.py
@numba.jit(nopython=True)
def bazin(time, low_passband, A, B, t0, tfall, trise, cc):
    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
    return (A * X + B) * (1 - cc * low_passband)


@numba.jit(nopython=True)
def errfunc(params, time, low_passband, flux, weights):
    return abs(flux - bazin(time, low_passband, *params)) * weights


def fit_scipy(time, low_passband, flux, flux_err):
    time -= time[0]
    sn = np.power(flux / flux_err, 2)
    start_point = (sn * flux).argmax()

    t0_init = time[start_point] - time[0]
    amp_init = flux[start_point]
    weights = 1 / (1 + flux_err)
    weights = weights / weights.sum()
    guess = [0, amp_init, t0_init, 40, -5, 0.5]

    result = least_squares(errfunc, guess, args=(time, low_passband, flux, weights), method='lm')
    result.t_shift = t0_init - result.x[2]

    return result


def yield_data(meta_df, lc_df):
    cols = ["object_id", "mjd", "flux", "flux_err", "low_passband"]
    for i in range(NUM_PARTITIONS):
        yield meta_df[(meta_df["object_id"] % NUM_PARTITIONS) == i]["object_id"].values, \
              lc_df[(lc_df["object_id"] % NUM_PARTITIONS) == i][cols]


def get_params(object_id_list, lc_df, result_queue):
    results = {}
    for object_id in object_id_list:
        light_df = lc_df[lc_df["object_id"] == object_id]
        try:
            result = fit_scipy(light_df["mjd"].values, light_df["low_passband"].values,
                               light_df["flux"].values, light_df["flux_err"].values)
            results[object_id] = np.append(result.x, [result.cost, result.status, result.t_shift])
        except Exception as e:
            print(e)
            results[object_id] = None
    result_queue.put(results)


def parallelize(meta_df, df):
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)

    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    for m, d in yield_data(meta_df, df):
        pool.apply_async(get_params, (m, d, result_queue))

    pool.close()
    pool.join()

    return [result_queue.get() for _ in range(NUM_PARTITIONS)]


if __name__ == "__main__":
    args = get_fe_argparser("Fit bazin function and generate features.")

    meta_df = pd.read_csv(args.meta_path)
    lc_df = pd.read_csv(args.light_curve_path)
    lc_df["low_passband"] = (lc_df["passband"] < LOW_PASSBAND_LIMIT).astype(int)

    result_list = parallelize(meta_df, lc_df)
    final_result = {}
    for res in result_list:
        final_result.update(res)

    for index, col in enumerate(FEATURES):
        meta_df[col] = meta_df["object_id"].apply(lambda x: final_result[x][index])

    meta_df[["object_id"] + FEATURES].to_csv(args.output_path, index=False)
