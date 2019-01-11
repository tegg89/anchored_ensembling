import numpy as np
import matplotlib.pyplot as plt

# plot priors
def plot_priors(X_val, y_prior, n_ensembles):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    for ens in range(0, n_ensembles):
        ax.plot(X_val, y_prior[ens], 'k')
    ax.set_xlim(-2.5, 2.5)
    plt.show()

# plot predictions
def plot_pred(X_train, X_val, y_train, y_pred, n_ensembles):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    for ens in range(0, n_ensembles):
        ax.plot(X_val, y_pred[ens], 'k')
    ax.plot(X_train[:, 0], y_train, 'r.',
            markersize=14, markeredgecolor='k', markeredgewidth=0.5)
    ax.set_ylim(-4, 2)
    ax.set_xlim(-2.5, 2.5)

# plot data
def plot_data(X_train, y_train):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(X_train[:, 0], y_train, 'r.',
            markersize=14, markeredgecolor='k', markeredgewidth=0.5)
    ax.set_ylim(-4, 2)
    ax.set_xlim(-2.5, 2.5)
    plt.show()

def plot_result(X_train, X_val, y_train, y_pred, y_pred_mu, y_pred_std):
    # plot predictive distribution
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    ax.plot(X_val, y_pred_mu, 'b-', linewidth=2., label=u'Prediction')
    ax.plot(X_val, y_pred_mu + 2 * y_pred_std, 'b', linewidth=0.5)
    ax.plot(X_val, y_pred_mu - 2 * y_pred_std, 'b', linewidth=0.5)
    ax.plot(X_val, y_pred_mu + 1 * y_pred_std, 'b', linewidth=0.5)
    ax.plot(X_val, y_pred_mu - 1 * y_pred_std, 'b', linewidth=0.5)
    ax.fill(np.concatenate([X_val, X_val[::-1]]),
            np.concatenate([y_pred_mu - 2 * y_pred_std,
                            (y_pred_mu + 2 * y_pred_std)[::-1]]),
            alpha=1, fc='lightskyblue', ec='None')
    ax.fill(np.concatenate([X_val, X_val[::-1]]),
            np.concatenate([y_pred_mu - 1 * y_pred_std,
                            (y_pred_mu + 1 * y_pred_std)[::-1]]),
            alpha=1, fc='deepskyblue', ec='None')

    ax.plot(X_train[:, 0], y_train, 'r.',
            markersize=14, markeredgecolor='k', markeredgewidth=0.5)
    ax.set_ylim(-4, 2)
    ax.set_xlim(-2.5, 2.5)
    plt.show()
