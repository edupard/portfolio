import numpy as np
import tensorflow as tf
import math


class Capm:
    def __init__(self, num_stks):
        self.exp = exp = tf.placeholder(tf.float64, shape=(num_stks), name='exp')
        exp = tf.reshape(exp, shape=(num_stks, 1))
        self.cov = cov = tf.placeholder(tf.float64, shape=(num_stks, num_stks), name='cov')
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

    def get_params(self, exp, cov, w):
        feed_dict = {
            self.exp: exp,
            self.cov: cov,
            self.w: w,
        }
        grads, sharpe, expectation, variance = self.sess.run(
            [self.grads, self.sharpe, self.port_exp, self.port_var],
            feed_dict)
        return grads, sharpe, expectation, variance
