import tensorflow as tf

class NN():
    def __init__(self, 
                 x_dim, 
                 y_dim, 
                 hidden_size,
                 params,
                #  init_stddev_1_w, 
                #  init_stddev_1_b, 
                #  init_stddev_2_w,
                 n, 
                 learning_rate):

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.n = n
        self.learning_rate = learning_rate

        self.inputs = tf.placeholder(tf.float64, [None, x_dim], name='inputs')
        self.y_target = tf.placeholder(tf.float64, [None, y_dim], name='target')

        self.layer_1_w = tf.layers.Dense(hidden_size,
                                         activation=tf.nn.tanh,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0., stddev=params["init_stddev_1_w"]),
                                         bias_initializer=tf.random_normal_initializer(
                                             mean=0., stddev=params["init_stddev_1_b"]))

        self.layer_1 = self.layer_1_w.apply(self.inputs)

        self.output_w = tf.layers.Dense(y_dim,
                                        activation=None, 
                                        use_bias=False,
                                        kernel_initializer=tf.random_normal_initializer(
                                            mean=0., stddev=params["init_stddev_2_w"]))

        self.output = self.output_w.apply(self.layer_1)

        # set up loss and optimiser - we'll modify this later with anchoring regularisation
        self.opt_method = tf.train.AdamOptimizer(self.learning_rate)
        self.mse_ = 1/tf.shape(self.inputs, out_type=tf.int64)[0] \
            * tf.reduce_sum(tf.square(self.y_target - self.output))
        self.loss_ = 1/tf.shape(self.inputs, out_type=tf.int64)[0] \
            * tf.reduce_sum(tf.square(self.y_target - self.output))
        self.optimizer = self.opt_method.minimize(self.loss_)

    def get_weights(self, sess):
        '''method to return current params'''

        ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.output_w.kernel]
        w1, b1, w2 = sess.run(ops)

        return w1, b1, w2

    def anchor(self, sess, lambda_anchor):
        '''regularise around initialised parameters'''

        w1, b1, w2 = self.get_weights(sess)

        # get initial params
        self.w1_init, self.b1_init, self.w2_init = w1, b1, w2

        loss_anchor = lambda_anchor[0] * tf.reduce_sum(
            tf.square(self.w1_init - self.layer_1_w.kernel))
        loss_anchor += lambda_anchor[1] * \
            tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))
        loss_anchor += lambda_anchor[2] * tf.reduce_sum(
            tf.square(self.w2_init - self.output_w.kernel))

        # combine with original loss
        self.loss_ = self.loss_ + 1 / \
            tf.shape(self.inputs, out_type=tf.int64)[0] * loss_anchor
        self.optimizer = self.opt_method.minimize(self.loss_)

    def predict(self, x, sess):
        '''predict method'''

        feed = {self.inputs: x}
        y_pred = sess.run(self.output, feed_dict=feed)
        return y_pred
