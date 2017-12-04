import datetime

from net.net_shiva import NetShiva
from stock_data.datasource import DataSource
import net.train_config as train_config
import net.eval_config as eval_config
from net.predict import predict
import snp.snp as snp
import stock_data.config as config

#store yearly stocks data to temporary folder
config.get_config().FOLDER = config.TEMP_FOLDER
today = datetime.date.today()
year_before = today - datetime.timedelta(days=365)
config.get_config().DATA_BEG = year_before
config.get_config().DATA_END = today

EPOCH = 600

tickers = snp.get_snp_hitorical_components_tickers()

dss = []
for ticker in tickers:
    try:
        dss.append(DataSource(ticker))
    except:
        continue

train_config = train_config.get_train_config_bagging()
eval_config = eval_config.get_current_eval_config(year_before, today)
net = NetShiva(train_config)

BASE_FOLDER = train_config.DATA_FOLDER

weak_predictors = snp.get_snp_hitorical_components_tickers()
for ticker in tickers:
    try:
        print("Evaluating %s weak predictor..." % ticker)

        train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, ticker)
        net.update_config(train_config)

        net.load_weights(EPOCH)

        avg_loss, predictions_history, last_states = predict(net, dss, train_config, eval_config)
    except:
        pass