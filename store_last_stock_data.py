import pandas as pd
import snp.snp as snp
import numpy as np
import datetime

import utils.utils as utils
import stock_data.config as config
import utils.folders as folders

def convert_to_ib(ticker):
    return ticker.replace('-', ' ')

# read current px, dividends, splits
tod_px_ds = pd.read_csv("data/tod_px_adj/px_tod.csv")
div_ds = pd.read_csv("data/tod_px_adj/dividends.csv")
splits_ds = pd.read_csv("data/tod_px_adj/splits.csv")

# record tickers we can make predictions on
allowed_tickers = []

# get tod ts
today = datetime.date.today()
today_ts = utils.get_date_timestamp(today)

# iterate over snp tickers
tickers = snp.get_snp_tickers()
for ticker in tickers:
    try:
        # get data from temporary folder
        config.get_config().FOLDER = config.TEMP_FOLDER
        # get previous dump
        dump_file_name = config.get_config().get_dump_file_name(ticker)
        data = np.load(dump_file_name)

        ts = data['ts']
        o = data['o']
        h = data['h']
        l = data['l']
        c = data['c']
        v = data['v']
        a_o = data['a_o']
        a_h = data['a_h']
        a_l = data['a_l']
        a_c = data['a_c']
        a_v = data['a_v']

        # calculate ds len
        ds_len = ts.shape[0]

        ib_ticker = convert_to_ib(ticker)

        # get tod px
        tod_px = tod_px_ds[tod_px_ds.ticker == ib_ticker]
        # check weatcher we have tod px
        if tod_px.shape[0] == 1:
            # record ticker as valid
            allowed_tickers.append(ticker)

            # get tod px
            tod_o = tod_px.iloc[0].o
            tod_h = tod_px.iloc[0].h
            tod_l = tod_px.iloc[0].l
            tod_c = tod_px.iloc[0].c
            tod_v = tod_px.iloc[0].v

            # calculate adj factors
            ticker_div = div_ds[div_ds.ticker == ticker]
            div_f = 1.0
            if ticker_div.shape[0] == 1:
                div = ticker_div.iloc[0].dividend
                div_f = (a_c[-1] - div) / a_c[-1]

            ticker_split = splits_ds[splits_ds.ticker == ticker]
            split_f = 1.0
            if ticker_split.shape[0] == 1:
                split = ticker_split.iloc[0].split_factor
                split_f = split

            # crete new arrays

            ts_new = np.zeros([ds_len + 1])
            o_new = np.zeros([ds_len + 1])
            h_new = np.zeros([ds_len + 1])
            l_new = np.zeros([ds_len + 1])
            c_new = np.zeros([ds_len + 1])
            v_new = np.zeros([ds_len + 1])
            a_o_new = np.zeros([ds_len + 1])
            a_h_new = np.zeros([ds_len + 1])
            a_l_new = np.zeros([ds_len + 1])
            a_c_new = np.zeros([ds_len + 1])
            a_v_new = np.zeros([ds_len + 1])

            # fill historical values
            ts_new[0:ds_len] = ts
            o_new[0:ds_len] = o
            h_new[0:ds_len] = h
            l_new[0:ds_len] = l
            c_new[0:ds_len] = c
            v_new[0:ds_len] = v
            a_o_new[0:ds_len] = a_o * div_f * split_f
            a_h_new[0:ds_len] = a_h * div_f * split_f
            a_l_new[0:ds_len] = a_l * div_f * split_f
            a_c_new[0:ds_len] = a_c * div_f * split_f
            a_v_new[0:ds_len] = a_v * split_f

            # fill last day price
            ts_new[ds_len] = today_ts
            o_new[ds_len] = tod_o
            h_new[ds_len] = tod_h
            l_new[ds_len] = tod_l
            c_new[ds_len] = tod_c
            v_new[ds_len] = tod_v
            a_o_new[ds_len] = tod_o
            a_h_new[ds_len] = tod_h
            a_l_new[ds_len] = tod_l
            a_c_new[ds_len] = tod_c
            a_v_new[ds_len] = tod_v

            # swap references
            ts = ts_new
            o = o_new
            h = h_new
            l = l_new
            c = c_new
            v = v_new
            a_o = a_o_new
            a_h = a_h_new
            a_l = a_l_new
            a_c = a_c_new
            a_v = a_v_new

        else:
            print("No prices for %s" % ticker)

        # store data to temporary tod folder
        config.get_config().FOLDER = config.TEMP_TOD_FOLDER
        # get previous dump
        dump_folder_name = config.get_config().get_dir_name(ticker)
        folders.create_dir(dump_folder_name)
        dump_file_name = config.get_config().get_dump_file_name(ticker)
        # save data
        np.savez(dump_file_name,
                 ts=ts,
                 o=o,
                 h=h,
                 l=l,
                 c=c,
                 v=v,
                 a_o=a_o,
                 a_h=a_h,
                 a_l=a_l,
                 a_c=a_c,
                 a_v=a_v,
                 )

    except:
        pass

allowed_ticker_df = pd.DataFrame({'ticker' : allowed_tickers})
allowed_ticker_df.to_csv('data/temp/tod/allowed_tickers.csv', index=False)

