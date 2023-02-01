import numpy as np
from scipy.spatial.distance import cdist
import cluster
import cluster.utils as utils
import kmeans


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
                distances[i][j] = np.dot(xi-xj, xi-xj)

        #calculate silhouette scores
        sscores = []
        for i in range(n):

            #intra-cluster

            #collect distances across the ith row
            intradists = [j for x, j in enumerate(distances[i]) if (y[x] == y[i] and x != i)]
            # #collect distances down the ith column
            # if i == 0:
            #     intradists2 = []
            # elif i == 1:
            #     if y[0] == y[i]:
            #         intradists2 = [distances[0][i]]
            #     else:
            #         intradists2 = []
            # else:
            #     #print(distances[0:i,i])
            #     intradists2 = [j for x, j in enumerate(distances[0:i,i]) if y[x] == y[i]]
            #
            # intradists = intradists1+intradists2

            #print(len(intradists1))
            print(len(intradists))
            #print(n)
            #assert len(intradists) == n/5-1, "wrong intradists"
            cluster_size = len(intradists)
            intradist_sum = sum(intradists)

            ai = intradist_sum/(cluster_size-1)
            #print(ai)

            #inter-cluster

            #collect distances across the ith row
            interdists = [j for x, j in enumerate(distances[i]) if y[x] != y[i]]
            #collect distances down the ith column
            # if i == 0:
            #     interdists2 = []
            # elif i == 1:
            #     if y[0] != y[i]:
            #         interdists2 = [distances[0][i]]
            #     else:
            #         interdists2 = []
            # else:
            #     interdists2 = [j for x, j in enumerate(distances[0:i,i]) if y[x] != y[i]]

            # if i != 0:
            #     interdists2 = [j for x, j in enumerate(distances[0:i][i]) if y[x] != y[i]]
            # else:
            #     interdists2 = []

            #interdists = interdists1+interdists2

            #print(len(interdists))
            #assert len(interdists) == 4*n/5, "wrong interdists"

            noncluster_size = len(interdists)
            interdist_sum = sum(interdists)

            bi = interdist_sum/noncluster_size
            #print(bi)

            #print(max(ai, bi))
            si = (bi-ai)/max(ai, bi)

            sscores.append(si)

        return sscores



        #print(distances)

        # if np.dot(xi-xj, xi-xj) == 0:
        #     print(i)
        #     print(j+i)
        #     print(xi)
        #     print(xj)

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

