import numpy as np
from scipy.spatial.distance import cdist
import cluster
import cluster.utils as utils
import cluster.kmeans as kmeans


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        #check input data structure
        utils._inspect_matrix(X, 0, 2)
        utils._inspect_matrix(y, 0, 1)

        n = X.shape[0]

        #construct distance matrix (very inefficiently)
        distances = np.ndarray([n,n])
        for i, xi in enumerate(X):
            for j, xj in enumerate(X):
                distances[i][j] = np.linalg.norm(xi-xj)

        #calculate silhouette scores
        sscores = []
        for i in range(n):

            #intra-cluster

            #collect distances across the ith row
            intradists = [j for x, j in enumerate(distances[i]) if (y[x] == y[i] and x != i)]

            cluster_size = len(intradists)
            intradist_sum = sum(intradists)

            ai = intradist_sum/(cluster_size-1)

            #inter-cluster

            intercluster_sscores = []
            for cl in np.unique(y):
                #skip this point's own cluster
                if cl != y[i]:
                    #collect distances across the ith row
                    interdists = [j for x, j in enumerate(distances[i]) if y[x] == cl]

                    noncluster_size = len(interdists)
                    interdist_sum = sum(interdists)

                    bi = interdist_sum/noncluster_size

                    si = (bi-ai)/max(ai, bi)

                    intercluster_sscores.append(si)

            sscores.append(min(intercluster_sscores))

        return sscores


        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

#plotting method for graphical inspection
def _clustering_test_plot(k, seed, kmseed):

    cl_in = utils.make_clusters(k=k, seed = seed)
    km = kmeans.KMeans(k, .0005, 90)

    km.fit(cl_in[0], kmseed)

    centroids = km.get_centroids()
    print(f"SME = {km.get_error()}")

    predlabels = km.predict(cl_in[0])

    sil = Silhouette()
    silscore = sil.score(np.array(cl_in[0]), np.array(predlabels))

    plotpoints = cl_in[0]
    truelanels_c = cl_in[1]
    predlabels_c = predlabels

    utils.plot_multipanel(
            plotpoints,
            truelanels_c,
            predlabels_c,
            silscore)

_clustering_test_plot(5, 16, True)

