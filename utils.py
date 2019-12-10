# list of helper functions
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def drop_labels(df):
    return df.drop(columns=['subject', 'Activity'])


def convert_categories(row):
    conv_dict = {"LAYING": 'NOT_MOVING', "STANDING": 'NOT_MOVING', "SITTING": 'NOT_MOVING',
                 "WALKING": 'MOVING', "WALKING_UPSTAIRS": "MOVING",
                 "WALKING_DOWNSTAIRS": 'MOVING'}
    return conv_dict[row['Activity']]


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Borrowed From """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_confusion(confusion, plt_name):
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(confusion, cmap="Blues")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, confusion[i][j], ha="center", va="center", color="r")

    plt.xticks([], [])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.yticks([], [])

    fig.tight_layout()
    plt.savefig("plt_name" + ".png")
    plt.show()
