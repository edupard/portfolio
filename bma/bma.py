import pymc3 as pm
import pandas as pd
import theano
import numpy as np

class Bma(object):

    def get_prediction_weights(self, predictions, observations, N_SAMPLES = 1000, N_TUNES = 1000, method='stacking'):
        if predictions.shape[1] == 0:
            weak_predictors_num = predictions.shape[0]
            return np.full((weak_predictors_num), 1 / weak_predictors_num)

        sigma_start = np.std(observations)
        aplha_start = 1
        beta_start = 0

        models = []
        traces = []

        for i in range(predictions.shape[0]):
            p = predictions[i,:]

            with pm.Model() as model:
                sigma = pm.HalfNormal('sigma', 0.1, testval=aplha_start)
                alpha = pm.Normal('alpha', mu=1, sd=1, testval=aplha_start)
                beta = pm.Normal('beta', mu=0, sd=1, testval=beta_start)
                mu = alpha * p + beta
                likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=observations)
                trace = pm.sample(N_SAMPLES, tune=N_TUNES)
                models.append(model)
                traces.append(trace)

        compare_ds = pm.compare(traces, models, method=method)

        return compare_ds.weight.sort_index(ascending=True)



    def get_weights(self,  predictions_aapl, predictions_msft, predictions_bac, observations_aapl):
        N_SAMPLES = 1000
        N_TUNES = 1000

        sigma_start = np.std(observations_aapl)
        aplha_start = 1
        beta_start = 0

        # predictions_shared = theano.shared(predictions_aapl)
        predictions = np.stack([predictions_aapl, predictions_msft, predictions_bac])


        with pm.Model() as model:
            sigma = pm.HalfNormal('sigma', 0.1, testval=aplha_start)
            alpha = pm.Normal('alpha', mu=1, sd=1, testval=aplha_start, shape=3)
            beta = pm.Normal('beta', mu=0, sd=1, testval=beta_start, shape=3)
            mu = alpha * predictions + beta
            p = pm.Normal('p', mu=mu, sd=sigma, observed=observations_aapl)
            trace_model = pm.sample(N_SAMPLES, tune=N_TUNES)


        with pm.Model() as model_aapl:
            sigma = pm.HalfNormal('sigma', 0.1, testval=aplha_start)
            alpha = pm.Normal('alpha', mu=1, sd=1, testval=aplha_start)
            beta = pm.Normal('beta', mu=0, sd=1, testval=beta_start)
            mu = alpha * predictions_aapl + beta
            p = pm.Normal('p', mu=mu, sd=sigma, observed=observations_aapl)
            trace_model_aapl = pm.sample(N_SAMPLES, tune=N_TUNES)

        with pm.Model() as model_msft:
            sigma = pm.HalfNormal('sigma', 0.1, testval=aplha_start)
            alpha = pm.Normal('alpha', mu=1, sd=1, testval=aplha_start)
            beta = pm.Normal('beta', mu=0, sd=1, testval=beta_start)
            mu = alpha * predictions_msft + beta
            p = pm.Normal('p', mu=mu, sd=sigma, observed=observations_aapl)
            trace_model_msft = pm.sample(N_SAMPLES, tune=N_TUNES)

        with pm.Model() as model_bac:
            sigma = pm.HalfNormal('sigma', 0.1, testval=aplha_start)
            alpha = pm.Normal('alpha', mu=1, sd=1, testval=aplha_start)
            beta = pm.Normal('beta', mu=0, sd=1, testval=beta_start)
            mu = alpha * predictions_bac + beta
            p = pm.Normal('p', mu=mu, sd=sigma, observed=observations_aapl)
            trace_model_bac = pm.sample(N_SAMPLES, tune=N_TUNES)

        compare_1 = pm.compare([trace_model_aapl, trace_model_msft, trace_model_bac],
                                [model_aapl, model_msft, model_bac],
                                method='pseudo-BMA')
        compare_2 = pm.compare([trace_model_msft, trace_model_bac, trace_model_aapl],
                                [model_msft, model_bac, model_aapl],
                                method='pseudo-BMA')

        compare_3 = pm.compare([trace_model_aapl, trace_model_msft, trace_model_bac],
                               [model_aapl, model_msft, model_bac],
                               method='BB-pseudo-BMA')

        compare_4 = pm.compare([trace_model_aapl, trace_model_msft, trace_model_bac],
                               [model_aapl, model_msft, model_bac],
                               method='stacking')

        compare_5 = pm.compare([trace_model_msft, trace_model_bac],
                               [model_msft, model_bac],
                               method='pseudo-BMA')

        compare_6 = pm.compare([trace_model_aapl, trace_model_msft],
                               [model_aapl, model_msft],
                               method='BB-pseudo-BMA')

        compare_7 = pm.compare([trace_model_aapl, trace_model_msft],
                               [model_aapl, model_msft],
                               method='stacking')


        # pm.traceplot(trace_model)


        d = pd.read_csv('data/milk.csv', sep=';')
        d['neocortex'] = d['neocortex.perc'] / 100
        d.dropna(inplace=True)
        d.shape

        a_start = d['kcal.per.g'].mean()
        sigma_start = d['kcal.per.g'].std()

        mass_shared = theano.shared(np.log(d['mass'].values))
        neocortex_shared = theano.shared(d['neocortex'].values)

        with pm.Model() as m6_11:
            alpha = pm.Normal('alpha', mu=0, sd=10, testval=a_start)
            mu = alpha + 0 * neocortex_shared
            sigma = pm.HalfCauchy('sigma', beta=10, testval=sigma_start)
            kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=d['kcal.per.g'])
            trace_m6_11 = pm.sample(1000, tune=1000)

        pm.traceplot(trace_m6_11)

        with pm.Model() as m6_12:
            alpha = pm.Normal('alpha', mu=0, sd=10, testval=a_start)
            beta = pm.Normal('beta', mu=0, sd=10)
            sigma = pm.HalfCauchy('sigma', beta=10, testval=sigma_start)
            mu = alpha + beta * neocortex_shared
            kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=d['kcal.per.g'])
            trace_m6_12 = pm.sample(1000, tune=1000)

        with pm.Model() as m6_13:
            alpha = pm.Normal('alpha', mu=0, sd=10, testval=a_start)
            beta = pm.Normal('beta', mu=0, sd=10)
            sigma = pm.HalfCauchy('sigma', beta=10, testval=sigma_start)
            mu = alpha + beta * mass_shared
            kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=d['kcal.per.g'])
            trace_m6_13 = pm.sample(1000, tune=1000)

        with pm.Model() as m6_14:
            alpha = pm.Normal('alpha', mu=0, sd=10, testval=a_start)
            beta = pm.Normal('beta', mu=0, sd=10, shape=2)
            sigma = pm.HalfCauchy('sigma', beta=10, testval=sigma_start)
            mu = alpha + beta[0] * mass_shared + beta[1] * neocortex_shared
            kcal = pm.Normal('kcal', mu=mu, sd=sigma, observed=d['kcal.per.g'])
            trace_m6_14 = pm.sample(1000, tune=1000)

        pm.waic(trace_m6_14, m6_14)

        compare_df = pm.compare([trace_m6_11, trace_m6_12, trace_m6_13, trace_m6_14],
                                [m6_11, m6_12, m6_13, m6_14],
                                method='pseudo-BMA')

        compare_df.loc[:, 'model'] = pd.Series(['m6.11', 'm6.12', 'm6.13', 'm6.14'])
        compare_df = compare_df.set_index('model')
        compare_df

        pm.compareplot(compare_df)






        # d = pd.read_csv('data/milk.csv')
        # d.iloc[:, 1:] = d.iloc[:, 1:] - d.iloc[:, 1:].mean()
        # d.head()
        #
        # with pm.Model() as model_0:
        #     alpha = pm.Normal('alpha', mu=0, sd=10)
        #     beta = pm.Normal('beta', mu=0, sd=10)
        #     epsilon = pm.HalfNormal('epsilon', 10)
        #
        #     mu = alpha + beta * d['neocortex']
        #
        #     kcal = pm.Normal('kcal', mu=mu, sd=epsilon, observed=d['kcal.per.g'])
        #     trace_0 = pm.sample(2000)
        #
        #
        #
        # # predictions shape: wp * ts
        # # observations shape: ts
        # with pm.Model() as model:
        #     sd = pm.HalfNormal('epsilon', 0.1)
        #
        #     alpha = pm.Normal('alpha', mu=1, sd=1)
        #     beta = pm.Normal('beta', mu=0, sd=1)
        #     mu = alpha * predictions + beta
        #
        #     p = pm.Lognormal('predictions', mu=mu, sd=sd, observed = observations)
        #
        #     trace_0 = pm.sample(2000)
        #
        #     pm.traceplot(trace_0)
