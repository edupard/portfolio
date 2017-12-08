from capm.capm import Capm
import snp.snp as snp

wp = snp.get_snp_hitorical_components_tickers()

exp = [0.05, -0.03]
cov = [[0.05*0.05,0.05*0.03],[0.05*0.03, 0.03 * 0.03]]
print(cov)

capm = Capm(2)
capm.init()
i = 0
while i<10000:
    if i == 1161:
        _debug = 0
    w, sharpe, constraint, expectation, variance = capm.get_params(exp, cov)
    if w is None:
        break
    print("Iteration: %d Sharpe: %.2f Constraint: %.6f r: %.2f%% var: %.2f%%" % (i, sharpe, constraint, expectation * 100, variance * 100))
    print(w)
    capm.fit(exp, cov)
    capm.rescale_weights()
    i += 1