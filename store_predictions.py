import snp.snp as snp
import net.train_config as train_config
import net.eval_config as eval_config
import utils.folders as folders
import os
import pandas as pd
import utils.utils as utils
import pymongo

train_config = train_config.get_train_config_petri()
eval_config = eval_config.get_eval_config_petri_whole_set()

BASE_FOLDER = train_config.DATA_FOLDER

EPOCH = 600

client = pymongo.MongoClient('localhost',27017)
db = client['predictions']
collection = db['predictions']

for weak_predictor in snp.get_snp_hitorical_components_tickers():
    print("Storing %s weak predictions..." % weak_predictor)

    train_config.DATA_FOLDER = "%s/%s" % (BASE_FOLDER, weak_predictor)
    for ticker in snp.get_snp_hitorical_components_tickers():
        try:
            _, prediction_path = folders.get_prediction_path(train_config, eval_config, ticker, EPOCH)
            if os.path.exists(prediction_path):
                print("Found prediction for %s" % ticker)
                pred_df = pd.read_csv(prediction_path)
                pred_df = pred_df[['prediction','ticker','date','ts']]
                pred_df['weak_predictor'] = weak_predictor
                records = pred_df.to_dict('records')
                collection.insert_many(records)
        except:
            print("Error: can not store %s ticker for %s weak predictor" % (ticker, weak_predictor))
            pass
