import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np

from net.train_config import TrainConfig
import utils.folders as folders


class NetSoftmax(object):
    def __init__(self, train_config: TrainConfig):
        self.update_config(train_config)

        print('creating neural network...')
        with tf.Graph().as_default() as graph:
            self.labels = labels = tf.placeholder(tf.float32, [None, None], name='labels')
            self.mask = mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.input = input = tf.placeholder(tf.float32, [None, None, len(train_config.FEATURE_FUNCTIONS)], name='input')

            output = tf.contrib.layers.fully_connected(input, len(train_config.FEATURE_FUNCTIONS), activation_fn=tf.nn.relu,
                                                       scope='dense_0')

            # final layer to make prediction
            with tf.name_scope('prediction_layer'):
                self.weights = weights = tf.nn.softmax(output, 2)
                w = tf.expand_dims(weights, 3)
                i = tf.expand_dims(input, 3)
                r = tf.matmul(w, i, transpose_a=True)
                r = tf.squeeze(r, [3])
                # r = tf.reshape(r, (-1, -1, 1))

                self.returns = r

            with tf.name_scope('loss'):
                diff = self.returns - tf.expand_dims(labels, 2)
                self.sse = sse = tf.reduce_sum(tf.multiply(tf.square(diff), tf.expand_dims(mask, 2)))
                self.cost = sse / tf.reduce_sum(mask)
                self.optimizer = tf.train.AdamOptimizer()
                self.vars = tf.trainable_variables()
                self.grads_and_vars = self.optimizer.compute_gradients(self.cost, var_list=self.vars)
                self.train = self.optimizer.apply_gradients(self.grads_and_vars)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        self.sess = tf.Session(graph=graph)

    def update_config(self, train_config: TrainConfig):
        self.config = train_config

        self._WEIGHTS_FOLDER_PATH, self._WEIGHTS_PREFIX_PATH = folders.get_weights_path(self.config)
        folders.create_dir(self._WEIGHTS_FOLDER_PATH)

    def zero_state(self, batch_size):
        return None

    def eval(self, state, input, labels, mask):
        feed_dict = {self.input: input, self.labels: labels, self.mask: mask}

        sse, returns = self.sess.run((self.sse, self.returns), feed_dict)
        new_state = None
        return new_state, sse, returns

    def fit(self, state, input, labels, mask):
        feed_dict = {self.input: input, self.labels: labels, self.mask: mask}
        sse, returns, _ = self.sess.run((self.sse, self.returns, self.train), feed_dict)
        new_state = None
        return new_state, sse, returns

    def save_weights(self, epoch):
        print('saving %d epoch weights' % epoch)
        self.saver.save(self.sess, self._WEIGHTS_PREFIX_PATH, global_step=epoch, write_meta_graph=False)

    def init_weights(self):
        print('initializing weights...')
        self.sess.run(self.init)

    def load_weights(self, epoch):
        print('loading %d epoch weights' % epoch)
        self.saver.restore(self.sess, "%s-%d" % (self._WEIGHTS_PREFIX_PATH, epoch))
