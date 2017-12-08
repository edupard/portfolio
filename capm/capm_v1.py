import numpy as np
import tensorflow as tf
import math


class Capm:
    def __init__(self, num_stks, exp, cov):
        self.num_stks = num_stks

        self.exp = exp =tf.Variable(initial_value=exp, name='exp', dtype=tf.float64, trainable=False)

        exp = tf.reshape(exp, shape=(num_stks, 1))

        self.cov = cov = tf.Variable(initial_value=cov, name='cov', dtype=tf.float64, trainable=False)


        self.w = w = tf.placeholder(tf.float64, shape=(num_stks), name='weights')
        w = tf.reshape(w, shape=(num_stks, 1))
        self.port_exp = port_exp = tf.matmul(w, exp, transpose_a=True)
        self.port_var = port_var = tf.matmul(tf.matmul(w, cov, transpose_a=True), w, name='port_var')
        self.sharpe = sharpe = tf.reshape(port_exp / tf.sqrt(port_var), shape=())

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
        w = np.array([1/self.num_stks] * self.num_stks)

        i = 0
        exit = False
        while not exit:
            grads, sharpe, expectation, variance = self.get_params(w)
            # grads, sharpe, expectation, variance = capm.get_params(exp, cov, w)
            print("Iteration: %d Sharpe: %.2f r: %.2f%% var: %.10f%%" % (i, sharpe, expectation * 100, variance * 100))
            print(w)

            grads = grads[0].reshape([-1])
            LR = 1.0
            while True:
                if LR == 0:
                    exit = True
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

        return w