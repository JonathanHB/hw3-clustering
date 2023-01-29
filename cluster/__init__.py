"""
hw3-clustering
Jonathan Borowsky
this is a necessary docstring
"""


from .kmeans import KMeans
from .silhouette import Silhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

__version__ = "0.1.1"
