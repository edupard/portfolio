import pymongo
import datetime

client = pymongo.MongoClient('localhost',27017)
db = client['predictions']
collection = db['predictions']

collections_map = {}

it = collection.find({})
i = 0
for doc in it:
    s_d = doc["date"]
    p = doc["prediction"]
    w_p = doc["weak_predictor"]
    t = doc["ticker"]
    try:
        if s_d in collections_map:
            write_collection = collections_map[s_d]
        else:
            d = datetime.datetime.strptime(s_d, '%Y-%m-%d').date()
            collection_name = d.strftime('%Y-%m-%d')
            write_collection = db[collection_name]
            collections_map[s_d] = write_collection
        write_collection.insert({
            "prediction": p,
            "weak_predictor" : w_p,
            "ticker" : t
        })
    except:
        print("ERR D: %s WP: %s T: %s" % (s_d, w_p, t))
        pass
    if i % 1000000 == 0:
        print("%.2f" % (i / 1000000))
    i += 1