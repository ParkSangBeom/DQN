import numpy as np
import tensorflow as tf

class DQN:

    def __init__(self, session: tf.Session, input_size: int, output_size: int, name: str="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network();

    def _build_network(self, h_size = 4, l_rate = 0.1):
        
        self._X = tf.placeholder(dtype = tf.float32, shape = [None, self.input_size])
        self._Y = tf.placeholder(dtype = tf.float32, shape = [None, self.output_size])

        with tf.variable_scope(self.net_name):
            W1 = tf.get_variable(name = "W1", shape = [self.input_size, h_size], initializer = tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal(shape = [h_size]), name = "bais1")
            L1 = tf.nn.relu(tf.matmul(self._X, W1) + b1)

            W2 = tf.get_variable(name = "W2", shape = [h_size, self.output_size], initializer = tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal(shape = [self.output_size]), name = "bais2")
            self._Qpred = tf.matmul(L1, W2) + b2

            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)
            optimizer = tf.train.AdamOptimizer(learning_rate = l_rate)
            self._train = optimizer.minimize(self._loss)

    def predict(self, state: np.ndarray) -> np.ndarray:
        x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        feed = { self._X: x_stack, self._Y: y_stack }
        return self.session.run([self._loss, self._train], feed)