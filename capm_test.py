from capm.capm import Capm
import numpy as np

npz_file_name = "data/eval/dates/petri/2008-06-16.npz"
data = np.load(npz_file_name)

predictions = data['predictions']
pos_mask = data['pos_mask']

predictions = predictions[pos_mask,:]

cov = np.cov(predictions)
exp = np.mean(predictions, axis=1)

SELECTION = 5
abs_exp = np.abs(exp)
sel_idxs = np.argsort(abs_exp)
sel_idxs = sel_idxs[0:SELECTION]
exp = exp[sel_idxs]
cov = cov[sel_idxs,:]
cov = cov[:,sel_idxs]

num_stks = exp.shape[0]


capm = Capm(num_stks, exp, cov)
capm.init()
i = 0
while i<100000:
    if i == 22:
        _debug = 0
    if i == 1000:
        _debug = 0
    w, sharpe, constraint, expectation, variance = capm.get_params()
    if w is None:
        break
    print("Iteration: %d Sharpe: %.2f Constraint: %.6f r: %.4f%% var: %.6f%%" % (i, sharpe, constraint, expectation * 100, variance * 100))
    # print(w)
    capm.fit()
    capm.rescale_weights()
    i += 1