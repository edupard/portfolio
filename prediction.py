import datetime
import numpy as np
import pandas as pd

from net.net_shiva_v1 import NetShivaV1
from stock_data.datasource import DataSource
import net.train_config as train_config
import net.eval_config as eval_config
import net.predict as predict
import snp.snp as snp
import stock_data.config as config
import pickle
import utils.folders as folders
import utils.utils as utils

# store yearly stocks data to temporary folder
PREDICTION_DATE = datetime.datetime.strptime('2017-12-04', '%Y-%m-%d').date()
ONE_WEEK_DATE = datetime.datetime.strptime('2017-11-27', '%Y-%m-%d').date()
ONE_DAY_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d').date()
BM_DATE = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d').date()
OPEN_POS_DATE = datetime.datetime.strptime('2017-12-04', '%Y-%m-%d').date()
HPR_DATE = datetime.datetime.strptime('2017-12-04', '%Y-%m-%d').date()


config.get_config().FOLDER = config.TEMP_TOD_FOLDER
today = datetime.date.today()
year_before = PREDICTION_DATE - datetime.timedelta(days=365)
config.get_config().DATA_BEG = year_before
config.get_config().DATA_END = today

EPOCH = 600

tickers = snp.get_snp_tickers()

dss_map = {}
dss = []
for ticker in tickers:
    try:
        ds = DataSource(ticker)
        dss.append(ds)
        dss_map[ticker] = ds
    except:
        continue

train_config = train_config.get_train_config_bagging()
eval_config = eval_config.get_current_eval_config(year_before, today)
net = NetShivaV1(train_config)

BASE_FOLDER = train_config.DATA_FOLDER

weak_predictors = snp.get_snp_hitorical_components_tickers()
weak_predictions = []
for ticker in snp.get_snp_hitorical_components_tickers():
    try:
        print("Evaluating %s weak predictor..." % ticker)

        train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, ticker)
        net.update_config(train_config)

        net.load_weights(EPOCH)

        weak_prediction = predict.batch_predict(net, dss, train_config, eval_config)
        weak_predictions.append(weak_prediction)

        folder_path, file_path = folders.get_state_file_path(train_config)
        folders.create_dir(folder_path)
    except:
        pass

prediction_matrix = np.zeros([len(dss), len(weak_predictions)])
for i in range(len(weak_predictions)):
    prediction_matrix[:, i] = weak_predictions[i]

strong_predictions = np.mean(prediction_matrix, axis=1)
pred_tickers = []
for ds in dss:
    pred_tickers.append(ds.ticker)

allowed_tickers = pd.read_csv('data/temp/tod/allowed_tickers.csv')

pred_ds = pd.DataFrame({'ticker': pred_tickers, 'prediction': strong_predictions})
mask = pred_ds.ticker.isin(allowed_tickers.ticker)
pred_ds = pred_ds[mask]

ticker_arr = []
prediction_arr = []
class_arr = []
exchange_arr = []
one_week_px_arr = []
one_day_px_arr = []
prediction_px_arr = []
open_pos_px_arr = []
hpr_px_arr = []

one_week_pct_arr = []
one_day_pct_arr = []
hp_pct_arr = []

one_day_v_arr = []
one_week_v_arr = []

exch_map = snp.get_snp_ticker_to_exchange_mapping()


def get_pct(enter_px, exit_px):
    return (exit_px - enter_px) / enter_px


for index, row in pred_ds.iterrows():
    ticker = row.ticker
    prediction = row.prediction
    exchange = exch_map[ticker]
    ds = dss_map[ticker]
    one_week_date_idx, _ = ds.get_data_range(ONE_WEEK_DATE, ONE_WEEK_DATE)
    one_week_px = ds.get_a_c(one_week_date_idx)
    one_day_date_idx, _ = ds.get_data_range(ONE_DAY_DATE, ONE_DAY_DATE)
    one_day_px = ds.get_a_c(one_day_date_idx)
    prediction_date_beg_idx, prediction_date_end_idx = ds.get_data_range(BM_DATE, BM_DATE)
    prediction_px = ds.get_a_c(prediction_date_beg_idx)
    open_pos_date_idx, _ = ds.get_data_range(OPEN_POS_DATE, OPEN_POS_DATE)
    open_pos_px = ds.get_a_c(open_pos_date_idx)
    hpr_pos_date_idx, _ = ds.get_data_range(HPR_DATE, HPR_DATE)
    hpr_px = ds.get_a_c(hpr_pos_date_idx)

    one_day_v = ds.get_a_v(one_day_date_idx)
    one_week_v = np.mean(ds.get_a_v(one_week_date_idx, prediction_date_end_idx))

    one_day_v_arr.append(one_day_v)
    one_week_v_arr.append(one_week_v)

    ticker_arr.append(utils.convert_to_ib(ticker))
    exchange_arr.append(exchange)
    prediction_arr.append(prediction)
    class_arr.append('L' if prediction >= 0 else 'S')
    one_week_px_arr.append(one_week_px)
    one_day_px_arr.append(one_day_px)
    prediction_px_arr.append(prediction_px)
    open_pos_px_arr.append(open_pos_px)
    hpr_px_arr.append(hpr_px)

    one_week_pct_arr.append(get_pct(one_week_px, prediction_px))
    one_day_pct_arr.append(get_pct(one_day_px, prediction_px))
    hp_pct_arr.append(get_pct(open_pos_px, hpr_px))

final_tickers_count = len(ticker_arr)

result_df = pd.DataFrame({
    'ticker': ticker_arr,
    'exchange': exchange_arr,
    'long prob': prediction_arr,
    'class': class_arr,
    '1w': [ONE_WEEK_DATE.strftime('%Y-%m-%d')] * final_tickers_count,
    '1d': [ONE_DAY_DATE.strftime('%Y-%m-%d')] * final_tickers_count,
    '*': [BM_DATE.strftime('%Y-%m-%d')] * final_tickers_count,
    '#': [OPEN_POS_DATE.strftime('%Y-%m-%d')] * final_tickers_count,
    'hp': [HPR_DATE.strftime('%Y-%m-%d')] * final_tickers_count,
    '1w px': one_week_px_arr,
    '1d px': one_day_px_arr,
    '* px': prediction_px_arr,
    '# px': open_pos_px_arr,
    'hp px': hpr_px_arr,
    '1wr pct': one_week_pct_arr,
    '1dr pct': one_day_pct_arr,
    'hpr pct': hp_pct_arr,
    '1d v': one_day_v_arr,
    '1w avg v': one_week_v_arr,
},
    columns=[
        'ticker',
        'exchange',
        'long prob',
        'class',
        '1w',
        '1d',
        '*',
        '#',
        'hp',
        '1w px',
        '1d px',
        '* px',
        '# px',
        'hp px',
        '1wr pct',
        '1dr pct',
        'hpr pct',
        '1d v',
        '1w avg v',
    ]
)

result_df.to_csv('data/prediction.csv', index=False)
