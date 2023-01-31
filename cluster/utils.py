import numpy as np
import matplotlib.pyplot as plt

def make_clusters(
        n: int = 500, 
        m: int = 2, 
        k: int = 3, 
        bounds: tuple = (-10, 10),
        scale: float = 1,
        seed: int = 42) -> (np.ndarray, np.ndarray):
    """
    creates some clustered data

    inputs:
        n: int
            number of observations
        m: int
            number of features
        k: int
            number of clusters
        bounds: tuple
            minimum and maximum bounds for cluster grid
        scale: float
            standard deviation of normal distribution
        seed: int
            random seed

    outputs:
        (np.ndarray, np.ndarray)
            returns a 2D matrix of `n` observations and `m` features that are clustered into `k` groups
            returns a 1D array of `n` size that defines the cluster origin for each observation
    """
    np.random.seed(seed)
    assert k <= n

    labels = np.sort(np.random.randint(0, k, size=n))
    centers = np.random.uniform(bounds[0], bounds[1], size=(k,m))
    mat = np.vstack([
        np.random.normal(
            loc=centers[idx], 
            scale=scale, 
            size=(np.sum(labels==idx), m))
        for idx in np.arange(0, k)])

    return mat, labels


def plot_clusters(
        mat: np.ndarray, 
        labels: np.ndarray, 
        filename: str =None):
    """
    inputs:
        mat: np.ndarray
            a 2D matrix where each row is an observation and each column is a feature
        labels: np.ndarray
            a 1D array where each value represents an integer cluster that an observation belongs to
        filename: str
            an optional value to save a figure to a file
    """

    plt.figure(figsize=(5,5), dpi=200)
    plt.scatter(
        mat[:,0], 
        mat[:,1], 
        c=labels)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_multipanel(
        mat: np.ndarray,
        truth: np.ndarray,
        pred: np.ndarray,
        score: np.ndarray,
        filename: str = None):
    """
    Plots a multipanel figure visualizing the efficiency of truth, prediction, 
    and silhouette scoring on a provided dataset

    inputs:
        mat: np.ndarray
            a 2D matrix where each row is an observation and each column is a feature
        truth: np.ndarray
            a 1D array where each value represents a true integer cluster that an observation belongs to
        pred: np.ndarray
            a 1D array where each value represents a predicted integer cluster than an observation belongs to
        score: np.ndarray
            a 1D array where each value represents a float for the silhouette score of that observation
        filename: str
            an optional value to save a figure to a file
    """

    fig, axs = plt.subplots(1, 3, figsize=(9, 2.7), dpi=200)
    
    cvars = [truth, pred, score]
    names = ["True Cluster Labels", "Predicted Cluster Labels", "Silhouette Scores"]
    cmaps = [None, None, "seismic"]
    for idx, ax in enumerate(axs):
        sub = ax.scatter(
            mat[:,0],
            mat[:,1],
            c=cvars[idx],
            cmap=cmaps[idx],
            s=5)
        ax.set_title(names[idx])
        min_lim = np.min(mat)
        max_lim = np.max(mat)
        ax.set_xlim([min_lim, max_lim])
        ax.set_ylim([min_lim, max_lim])
        if idx == 2:
            plt.colorbar(sub, ax=ax)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

#added by JHB
#check observation/feature matrices
def _inspect_matrix(mat: np.ndarray, n_min=0, n_dims=2):

    #check input type
    if type(mat) != np.ndarray:  # isinstance(mat, np.ndarray): #isinstance does not work
        raise ValueError("Input must be a numpy array.")

    #check matrix dimensionality
    if len(mat.shape) != n_dims:
        raise ValueError(
            f"input matrix had {len(mat.shape)} dimensions.\n Input must be a 2D matrix where the rows are observations and columns are features")

    #check for empty matrices
    if mat.shape[0] == 0:
        raise ValueError(
            "No observations were provided. A non-empty input matrix is required.")

    #check for sufficient number of observations
    if mat.shape[0] < n_min:
        raise ValueError(
            "The number of clusters requested exceeds the number of observations. There must be at least as many observations as clusters")