import pymc3 as pm
import pandas as pd
import theano
import numpy as np
import scipy.stats

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2)
    return y

_prob_extractor = lambda r, mu, std : scipy.stats.norm.pdf(r, mu, std)

_v_prob_extractor = np.vectorize(normpdf)

class BmaAnalytical(object):

    def __init__(self, window_size):
        self.window_size = window_size

    def get_prediction_weights(self, predictions, observations):
        weak_predictors_num = predictions.shape[0]

        if predictions.shape[1] < self.window_size:
            return np.full((weak_predictors_num), 1 / weak_predictors_num)

        sigma = np.std(observations)
        if sigma <= 0.0001:
            return np.full((weak_predictors_num), 1 / weak_predictors_num)
        probs = _v_prob_extractor(predictions, observations, sigma)
        # probs = np.zeros(predictions.shape)
        # for i in range(weak_predictors_num):
        #     probs[i,:] = _v_prob_extractor(predictions[i,:], observations, sigma)

        unnormalized_weights = np.prod(probs, axis=1)
        z = np.sum(unnormalized_weights)
        if (z <= 0):
            return np.full((weak_predictors_num), 1 / weak_predictors_num)
        return unnormalized_weights / z