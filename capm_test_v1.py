from capm.capm_v1 import Capm
import numpy as np
import snp.snp as snp

components = snp.get_snp_hitorical_components_tickers()
length = len(components)
idx = components.index('SWK')


exp = np.array([0.05, -0.03])
cov = np.array([[0.05*0.05,0.05*0.03],[0.05*0.03, 0.03 * 0.03]])

# w = np.array([0.375, -0.625])

# portfolio_variance = np.sqrt(np.matmul(np.matmul(w.T, cov), w))
# portfolio_return = np.matmul(exp.T, exp)

# w = [-0.5,-0.5]
# print(cov)

capm = Capm(2, exp, cov)
capm.init()
w = capm.fit_weights()

# i = 0
# exit = False
# while not exit:
#     grads, sharpe, expectation, variance = capm.get_params(w)
#     # grads, sharpe, expectation, variance = capm.get_params(exp, cov, w)
#     print("Iteration: %d Sharpe: %.2f r: %.2f%% var: %.10f%%" % (i, sharpe, expectation * 100, variance * 100))
#     print(w)
#
#     grads = grads[0].reshape([-1])
#     LR = 1.0
#     while True:
#         if LR == 0:
#             exit = True
#             break
#         _w = w - LR * grads
#         constraint = np.sum(np.abs(_w))
#         _w = _w / constraint
#         _grads, _sharpe, _expectation, _variance = capm.get_params(_w)
#         # _grads, _sharpe, _expectation, _variance = capm.get_params(exp, cov, _w)
#         if _sharpe > sharpe:
#             w = _w
#             break
#         LR = LR / 2
#
#     i += 1