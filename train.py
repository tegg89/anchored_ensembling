import numpy as np
import tensorflow as tf

class Trainer():
    def __init__(self, X_train, X_val, y_train, base, n_ensembles, data_noise, params, hidden=100):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.base = base
        
        self.n_ensembles = n_ensembles
        self.hidden = hidden
        self.n_epochs = 5
        self.learning_rate = 0.01
        self.data_noise = data_noise

        self.n = X_train.shape[0]
        self.x_dim = X_train.shape[1]
        self.y_dim = y_train.shape[1]

        self.NNs = []
        self.y_prior = []
        self.y_pred = []

        self.params = params
        self.lambda_anchor = self.data_noise / \
            (np.array([self.params["init_stddev_1_w"], 
                       self.params["init_stddev_1_b"], 
                       self.params["init_stddev_2_w"]])**2)

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.init_ensemble_model()
        
    def init_ensemble_model(self):
        for ens in range(0, self.n_ensembles):
            self.NNs.append(
                self.base(self.x_dim, self.y_dim, self.hidden, 
                          self.params, self.n, self.learning_rate))

            # initialise only unitialized variables - stops overwriting ensembles already created
            global_vars = tf.global_variables()
            is_not_initialized = self.sess.run(
                [tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(
                global_vars, is_not_initialized) if not f]
            if len(not_initialized_vars):
                self.sess.run(tf.variables_initializer(not_initialized_vars))

            # do regularisation now that we've created initialisations
            self.NNs[ens].anchor(self.sess, self.lambda_anchor)

            # save their priors
            self.y_prior.append(self.NNs[ens].predict(self.X_val, self.sess))

    def train(self):
        for ens in range(0, self.n_ensembles):

            feed_b = {}
            feed_b[self.NNs[ens].inputs] = self.X_train
            feed_b[self.NNs[ens].y_target] = self.y_train
            print('\nNN:', ens)

            ep_ = 0
            while ep_ < self.n_epochs:
                ep_ += 1
                blank = self.sess.run(self.NNs[ens].optimizer, feed_dict=feed_b)
                if ep_ % (self.n_epochs/5) == 0:
                    loss_mse = self.sess.run(self.NNs[ens].mse_, feed_dict=feed_b)
                    loss_anch = self.sess.run(self.NNs[ens].loss_, feed_dict=feed_b)
                    print('epoch:', ep_, ', mse_', np.round(loss_mse*1e3, 3),
                        ', loss_anch', np.round(loss_anch*1e3, 3))
                    # we're minimising the anchored loss, but it's useful to keep an eye on mse too



    def predict(self):
        # run predictions
        for ens in range(0, self.n_ensembles):
            self.y_pred.append(self.NNs[ens].predict(self.X_val, self.sess))

    def ensemble(self):
        # combine ensembles estimates properly
        y_preds = np.array(self.y_pred)
        y_preds = y_preds[:, :, 0]
        y_pred_mu = np.mean(y_preds, axis=0)
        y_pred_std = np.std(y_preds, axis=0, ddof=1)

        # add on data noise
        y_pred_std = np.sqrt(np.square(y_pred_std) + self.data_noise)

        return y_pred_mu, y_pred_std
