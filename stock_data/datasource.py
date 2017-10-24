import numpy as np

from utils.utils import get_date_timestamp, date_from_timestamp
from stock_data.config import get_config

_dow = np.arange(1, 8)
_dow_stddev = np.sqrt(np.cov(_dow))
_dow_mean = np.mean(_dow)

_moy = np.arange(1, 13)
_moy_stddev = np.sqrt(np.cov(_moy))
_moy_mean = np.mean(_moy)


def _calc_pct(new, old):
    if old != 0:
        return (new - old) / old
    return 0


_v_pct = np.vectorize(_calc_pct)

_day_of_week_extractor = lambda ts: date_from_timestamp(ts).isoweekday()

_v_day_of_week_extractor = np.vectorize(_day_of_week_extractor)

_month_extractor = lambda ts: date_from_timestamp(ts).month

_v_month_extractor = np.vectorize(_month_extractor)


class DataSource(object):
    def __init__(self, ticker):
        self.ticker = ticker
        print('loading data for %s...' % ticker)
        dump_file_name = get_config().get_dump_file_name(self.ticker)
        data = np.load(dump_file_name)

        self.ts = data['ts']
        self.o = data['o']
        self.h = data['h']
        self.l = data['l']
        self.c = data['c']
        self.v = data['v']
        self.a_o = data['a_o']
        self.a_h = data['a_h']
        self.a_l = data['a_l']
        self.a_c = data['a_c']
        self.a_v = data['a_v']

        # this is not memory efficient, but code is beautiful and easy
        self.prev_a_c = np.roll(self.a_c, 1)
        self.prev_a_c[0] = self.a_c[0]
        self.prev_a_v = np.roll(self.a_v, 1)
        self.prev_a_v[0] = self.a_v[0]

        self.f_o = _v_pct(self.a_o, self.prev_a_c)
        self.f_c = _v_pct(self.a_c, self.prev_a_c)
        self.f_h = _v_pct(self.a_h, self.prev_a_c)
        self.f_l = _v_pct(self.a_l, self.prev_a_c)
        self.f_v = _v_pct(self.a_v, self.prev_a_v)

        px_1d = np.roll(self.a_c, -1)
        px_1d[-1:] = self.a_c[-1]
        self.f_1dy = _v_pct(px_1d, self.a_c)

        px_5d = np.roll(self.a_c, -5)
        px_5d[-5:] = self.a_c[-1]
        self.f_5dy = _v_pct(px_5d, self.a_c)

        self.day_of_week = _v_day_of_week_extractor(self.ts)
        self.month = _v_month_extractor(self.ts)
        self.f_dow = (self.day_of_week - _dow_mean) / _dow_stddev
        self.f_moy = (self.day_of_week - _moy_mean) / _moy_stddev

    def get_o_f(self, b, e):
        return self.f_o[b: e]

    def get_c_f(self, b, e):
        return self.f_c[b: e]

    def get_h_f(self, b, e):
        return self.f_h[b: e]

    def get_l_f(self, b, e):
        return self.f_l[b: e]

    def get_v_f(self, b, e):
        return self.f_v[b: e]

    def get_1dy_f(self, b, e):
        return self.f_1dy[b: e]

    def get_5dy_f(self, b, e):
        return self.f_5dy[b: e]

    def get_dow_f(self, b, e):
        return self.f_dow[b: e]

    def get_moy_f(self, b, e):
        return self.f_moy[b: e]

    def get_data_range(self, beg, end):
        beg_ts = get_date_timestamp(beg)
        end_ts = get_date_timestamp(end)
        beg_idxs = np.nonzero(self.ts >= beg_ts)
        end_idxs = np.nonzero(self.ts <= end_ts)
        beg_idx = None
        if beg_idxs[0].shape[0] > 0:
            beg_idx = beg_idxs[0][0]
        end_idx = None
        if end_idxs[0].shape[0] > 0:
            end_idx = end_idxs[0][-1] + 1
        return beg_idx, end_idx
