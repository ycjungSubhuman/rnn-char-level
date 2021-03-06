'''
Codes for visualization
'''
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

def save_confusion_matrix(path, label_names, mat):
    """
    Visualize confusion matrix as a .png image

    path            result .png path
    label_names     class label names
    mat             confusion matrix (np.ndarray)
    """
    assert len(mat.shape) == 2
    assert mat.shape[0] == len(label_names)
    assert mat.shape[1] == len(label_names)
    plt.figure("Confusion matrix")
    acc = np.diag(mat).sum() / mat.sum()
    plt.title("Confusion matrix ({})".format(acc))
    conf_dataframe = pd.DataFrame(mat, label_names, label_names)
    sn.heatmap(conf_dataframe, annot=True)
    plt.savefig(path)
    plt.close()

def save_loss_graph(path, iteration, train_loss, val_loss):
    """
    Visualize loss graph as a .png image

    path            result .png path
    iteration       corresponding iteration number
    train_loss      training loss over time
    val_loss        validation loss over time
    """
    plt.figure("Loss progress")
    plt.title("Loss progress")
    plt.plot(iteration, train_loss, label="Train loss")
    plt.plot(iteration, val_loss, label="Validation loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(path)
    plt.close()
