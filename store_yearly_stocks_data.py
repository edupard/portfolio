import stock_data.download as download
import snp.snp as snp
import stock_data.config as config
import datetime

#store yearly stocks data to temporary folder
config.get_config().FOLDER = config.TEMP_FOLDER
today = datetime.date.today()
year_before = today - datetime.timedelta(days=365)
config.get_config().DATA_BEG = year_before
config.get_config().DATA_END = today

# download data for current snp components
tickers = snp.get_snp_tickers()
download.download_data(tickers, num_workers=20)