import numpy as np
import tensorflow as tf
import math


class Capm:
    def __init__(self, num_prediods, selection, exp, cov):
        self.num_prediods = num_prediods
        self.var_mul = math.sqrt(num_prediods)
        self.num_stks = exp.shape[0]

        abs_exp = np.abs(exp)
        sel_idxs = np.argsort(abs_exp)
        selection = min(selection, exp.shape[0])
        self.selection = selection
        self.sel_idxs = sel_idxs[-selection:]

        exp = exp[self.sel_idxs]
        cov = cov[self.sel_idxs, :]
        cov = cov[:, self.sel_idxs]

        self.exp = exp = tf.Variable(initial_value=exp, name='exp', dtype=tf.float64, trainable=False)

        exp = tf.reshape(exp, shape=(selection, 1))

        self.cov = cov = tf.Variable(initial_value=cov, name='cov', dtype=tf.float64, trainable=False)

        self.w = w = tf.placeholder(tf.float64, shape=(selection), name='weights')
        w = tf.reshape(w, shape=(selection, 1))
        self.port_exp = port_exp = tf.matmul(w, exp, transpose_a=True)
        self.port_var = port_var = tf.sqrt(tf.matmul(tf.matmul(w, cov, transpose_a=True), w, name='port_var'))
        self.sharpe = sharpe = tf.reshape(port_exp / port_var, shape=())

        self.loss = loss = -sharpe

        self.grads = tf.gradients(loss, w)

        self.sess = tf.Session()

    def init(self):
        print('initializing weights...')
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_params(self, w):
        feed_dict = {
            self.w: w,
        }
        grads, sharpe, expectation, variance = self.sess.run(
            [self.grads, self.sharpe, self.port_exp, self.port_var],
            feed_dict)
        return grads, sharpe, expectation, variance

    def fit_weights(self):

        def print_params(i, sharpe, expectation, variance):
            e = expectation * 100
            v = variance * 100
            ann_sharpe = sharpe * self.var_mul
            a_e = e * self.num_prediods
            a_v = v * self.var_mul
            print("iteration: %d sharpe: %.5f r: %.3f%% var: %.3f%% ann sharpe: %.5f ann r: %.3f%% ann v: %.3f%%" % (
            i, sharpe, e, v, ann_sharpe, a_e, a_v))

        w = np.array([1 / self.selection] * self.selection)

        CHANGE_BREAK = 0.0001
        i = 0
        LR = 1.0
        _sharpe = None
        while True:
            if _sharpe is not None:
                if (_sharpe - sharpe) / sharpe < CHANGE_BREAK:
                    grads, sharpe, expectation, variance = self.get_params(w)
                    print_params(i, sharpe, expectation, variance)
                    break
            grads, sharpe, expectation, variance = self.get_params(w)

            # print_params(i, sharpe, expectation, variance)
            # print(w)

            grads = grads[0].reshape([-1])

            while True:
                if LR == 0:
                    break
                _w = w - LR * grads
                constraint = np.sum(np.abs(_w))
                _w = _w / constraint
                _grads, _sharpe, _expectation, _variance = self.get_params(_w)
                # _grads, _sharpe, _expectation, _variance = capm.get_params(exp, cov, _w)
                if _sharpe > sharpe:
                    w = _w
                    break
                LR = LR / 2

            i += 1

        weights = np.zeros((self.num_stks))
        weights[self.sel_idxs] = w
        return weights
