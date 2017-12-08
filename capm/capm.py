import numpy as np
import tensorflow as tf
import math


class Capm:
    def __init__(self, num_stks):
        self.exp = exp = tf.placeholder(tf.float64, shape=(num_stks), name='exp')
        exp = tf.reshape(exp, shape=(num_stks, 1))
        self.cov = cov = tf.placeholder(tf.float64, shape=(num_stks, num_stks), name='cov')

        init_w = np.full((num_stks), -1 / num_stks)
        self.w = w = tf.Variable(initial_value=init_w, name='weights', dtype=tf.float64)
        w = tf.reshape(w, shape=(num_stks, 1))
        self.port_exp = port_exp = tf.matmul(w, exp, transpose_a=True)
        self.port_var = port_var = tf.matmul(tf.matmul(w, cov, transpose_a=True), w, name='port_var')
        self.sharpe = sharpe = tf.reshape(port_exp / tf.sqrt(port_var), shape=())

        self.constraint = constraint = tf.reduce_sum(tf.abs(w))
        self.rescale_op = self.w.assign(self.w / constraint)

        self.loss = loss = -sharpe

        self.optimizer = optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        # self.optimizer = optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        # self.optimizer = optimizer = tf.train.AdamOptimizer()
        # self.gv = gv = optimizer.compute_gradients(loss, var_list=[self.w])
        # self.train = optimizer.apply_gradients(gv)
        self.train = optimizer.minimize(loss)

        self.sess = tf.Session()

    def rescale_weights(self):
        self.sess.run(self.rescale_op)

    def init(self):
        print('initializing weights...')
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_params(self, exp, cov):
        feed_dict = {
            self.exp: exp,
            self.cov: cov
        }
        w, sharpe, constraint, expectation, variance = self.sess.run(
            [self.w, self.sharpe, self.constraint, self.port_exp, self.port_var],
            feed_dict)
        return w, sharpe, constraint, expectation, variance

    def fit(self, exp, cov):
        feed_dict = {
            self.exp: exp,
            self.cov: cov
        }
        _ = self.sess.run(self.train, feed_dict)
