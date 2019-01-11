import numpy as np

from train import Trainer
from model import NN
from utils import plot_priors, plot_pred, plot_data, plot_result

# create some data
X_train = np.atleast_2d([1., 4.5, 5.1, 6., 8., 9.]).T
X_train = X_train/5. - 1
y_train = X_train * np.sin(X_train*5.)  # y = xsin(5x)

# create validation data - here we'll just a 1-d grid
X_val = np.atleast_2d(np.linspace(-3, 3, 100)).T
y_val = np.expand_dims(X_val[:, 0], 1)  # just dummy data

params = {
    "init_stddev_1_w": np.sqrt(10),
    "init_stddev_1_b": np.sqrt(10),  # set these equal
    "init_stddev_2_w": 1.0/np.sqrt(100)  # normal scaling
}

trainer = Trainer(X_train=X_train, X_val=X_val, y_train=y_train,
                  base=NN, n_ensembles=5,
                  data_noise=0.001, params=params)

plot_priors(X_val, trainer.y_prior, n_ensembles=5)

trainer.train()

trainer.predict()

plot_pred(X_train, X_val, y_train, trainer.y_pred, n_ensembles=5)

y_pred_mu, y_pred_std = trainer.ensemble()

plot_result(X_train, X_val, y_train, trainer.y_pred, y_pred_mu, y_pred_std)
