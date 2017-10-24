import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np

from net.train_config import TrainConfig
from utils.folders import create_dir


class NetShiva(object):
    def __init__(self, train_config: TrainConfig):
        self.config = train_config

        self._WEIGHTS_FOLDER_PATH = '%s/weights' % self.config.DATA_FOLDER
        create_dir(self._WEIGHTS_FOLDER_PATH)
        self._WEIGHTS_PATH = '%s/%s' % (self._WEIGHTS_FOLDER_PATH, 'weights')

        print('creating neural network...')
        with tf.Graph().as_default() as graph:
            self.labels = labels = tf.placeholder(tf.float32, [None, None], name='labels')
            self.mask = mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.input = input = tf.placeholder(tf.float32, [None, None, len(train_config.FEATURE_FUNCTIONS)], name='input')
            self.keep_prob = keep_prob = tf.placeholder(tf.float32)

            self.rnn_cell = rnn_cell = rnn.MultiRNNCell(
                [
                    rnn.DropoutWrapper(rnn.LSTMCell(self.config.LSTM_LAYER_SIZE), input_keep_prob=keep_prob)
                    for _ in range(self.config.LSTM_LAYERS)
                ])

            state = ()
            for s in rnn_cell.state_size:
                c = tf.placeholder(tf.float32, [None, s.c])
                h = tf.placeholder(tf.float32, [None, s.h])
                state += (tf.contrib.rnn.LSTMStateTuple(c, h),)
            self.state = state

            # Batch size x time steps x features.
            output, new_state = tf.nn.dynamic_rnn(rnn_cell, input, initial_state=state)
            self.new_state = new_state

            fc_layer_idx = 0
            for num_units in self.config.FC_LAYERS:
                scope_name = 'fc_layer_%d' % fc_layer_idx
                with tf.name_scope(scope_name):
                    output = tf.contrib.layers.fully_connected(output, num_units, activation_fn=tf.nn.relu,
                                                               scope='dense_%d' % fc_layer_idx)
                    output = tf.nn.dropout(output, keep_prob)
                fc_layer_idx += 1

            # final layer to make prediction
            with tf.name_scope('prediction_layer'):
                self.returns = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

            with tf.name_scope('loss'):
                diff = self.returns - tf.expand_dims(labels, 2)
                self.cost = tf.reduce_sum(tf.multiply(tf.square(diff), tf.expand_dims(mask, 2))) / tf.reduce_sum(mask)
                self.optimizer = tf.train.AdamOptimizer()
                self.vars = tf.trainable_variables()
                self.grads_and_vars = self.optimizer.compute_gradients(self.cost, var_list=self.vars)
                self.train = self.optimizer.apply_gradients(self.grads_and_vars)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.sess = tf.Session(graph=graph)

    def zero_state(self, batch_size):
        zero_state = ()
        for s in self.rnn_cell.state_size:
            c = np.zeros((batch_size, s.c))
            h = np.zeros((batch_size, s.h))
            zero_state += (tf.contrib.rnn.LSTMStateTuple(c, h),)
        return zero_state

    def _fill_feed_dict(self, feed_dict, state):
        idx = 0
        for s in state:
            feed_dict[self.state[idx].c] = s.c
            feed_dict[self.state[idx].h] = s.h
            idx += 1

    def eval(self, state, input, labels, mask):
        feed_dict = {self.input: input, self.labels: labels, self.mask: mask, self.keep_prob: 1.0}
        self._fill_feed_dict(feed_dict, state)

        new_state, cost, returns = self.sess.run((self.new_state, self.cost, self.returns), feed_dict)
        return new_state, cost, returns

    def fit(self, state, input, labels, mask):
        feed_dict = {self.input: input, self.labels: labels, self.mask: mask, self.keep_prob: self.config.DROPOUT_KEEP_RATE}
        self._fill_feed_dict(feed_dict, state)
        new_state, cost, returns, _ = self.sess.run((self.new_state, self.cost, self.returns, self.train), feed_dict)
        return new_state, cost, returns

    def save_weights(self, epoch):
        print('saving %d epoch weights' % epoch)
        self.saver.save(self.sess, self._WEIGHTS_PATH, global_step=epoch, write_meta_graph=False)

    def init_weights(self, epoch):
        if epoch != 0:
            print('loading %d epoch weights' % epoch)
            self.saver.restore(self.sess, "%s-%d" % (self._WEIGHTS_PATH, epoch))
        else:
            print('initializing weights...')

            self.sess.run(self.init)
