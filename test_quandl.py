from quandl_data.quandl_data import get_vix_tickers
from quandl_data.quandl_data import get_snp_tickers
from quandl_data.quandl_data import get_historical_data
from utils.folders import create_dir

for t, q_id in get_vix_tickers():
    try:
        ds = get_historical_data(q_id)
        folder = "data/futures/VX"
        create_dir(folder)
        ds.to_csv("%s/%s.csv" % (folder, t))
    except:
        continue
    # break

for t, q_id in get_snp_tickers():
    try:
        ds = get_historical_data(q_id)
        folder = "data/futures/ES"
        create_dir(folder)
        ds.to_csv("%s/%s.csv" % (folder, t))
    except:
        continue
    # break


