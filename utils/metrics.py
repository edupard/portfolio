import numpy as np
import math
from utils.utils import date_from_timestamp

def get_eq_params(capital, ts, recap = False):
    BEG = date_from_timestamp(ts[0])
    END = date_from_timestamp(ts[-1])
    years = (END - BEG).days / 365
    dd = get_draw_down(capital, recap)
    if recap:
        rets = (capital[1:] - capital[:-1]) / capital[:-1]
        sharpe = get_sharpe_ratio(rets, years)
        y_avg = (capital[-1] - capital[0]) / capital[0] / years
    else:
        rets = capital[1:] - capital[:-1]
        sharpe = get_sharpe_ratio(rets, years)
        y_avg = (capital[-1] - capital[0]) / years
    return dd, sharpe, y_avg

def get_draw_down(c, recap):
    # c == capital in time array
    # recap == flag indicating recapitalization or fixed bet
    def generate_previous_max():
        max = c[0]
        for idx in range(len(c)):
            # update max
            if c[idx] > max:
                max = c[idx]
            yield max

    prev_max = np.fromiter(generate_previous_max(), dtype=np.float64)
    if recap:
        dd_a = (c - prev_max) / prev_max
    else:
        dd_a = c - prev_max

    return np.min(dd_a)

def get_sharpe_ratio(ret, years):
    return math.sqrt(ret.shape[0] / years) * np.mean(ret) / np.std(ret)


def get_avg_yeat_ret(ret, years):
    return np.sum(ret) / years