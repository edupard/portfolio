from capm.capm_v1 import Capm
import numpy as np

npz_file_name = "data/eval/dates/petri/2008-06-16.npz"
data = np.load(npz_file_name)

predictions = data['predictions']
pos_mask = data['pos_mask']

predictions = predictions[pos_mask,:]
abs_predictions = np.abs(predictions)
zeros = np.zeros(abs_predictions.shape)
abs_predictions = abs_predictions - 0.001

cov = np.cov(predictions)
exp = np.mean(predictions, axis=1)

# var = np.zeros(cov.shape)
# for i in range(cov.shape[0]):
#     var[i,i] = cov[i,i]
# cov = var

SELECTION = 50
NUM_PERIODS = 48

capm = Capm(NUM_PERIODS, SELECTION, exp, cov)
capm.init()
w = capm.fit_weights()
debug = 0
