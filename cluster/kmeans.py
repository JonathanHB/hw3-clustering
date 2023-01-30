import numpy as np
import random
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):

        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        #store the cluster centers found by fit()
        self.cluster_centers = []


        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

    def fit(self, mat: np.ndarray):

        #-------------------------input inspection-------------------------

        #verify correct input type/structure
        if type(mat) != np.ndarray:  #isinstance(mat, np.ndarray): #isinstance does not work
            raise ValueError("Input must be a numpy array.")

        if len(mat.shape) != 2:
            raise ValueError(f"input matrix had {len(mat.shape)} dimensions.\n Input must be a 2D matrix where the rows are observations and columns are features")

        if mat.shape[0] < self.k:
            raise ValueError("The number of clusters requested exceeds the number of observations. There must be at least as many observations as clusters")

        #-------------------------miscellaneous-------------------------
        n = mat.shape[0]

        #-------------------------assignment generation-------------------------

        #generate initial assignments matrix and randomize the assignment of each observation
        cluster_assignments = np.zeros([n, self.k])

        #This step is statistically unlikely to be needed for large inputs where k is much smaller than
        #the number of observations (n), but is important for small n or k close to n.
        #If n is small or k is close to n, clustering should not be performed at all (or k should be reduced).
        #It was nevertheless necessary to make the code robust in this case.

        #assign one randomly chosen data point to each cluster so that no cluster is empty
        preassigned_inds = []
        for k in range(self.k):
            a = True
            while a:
                j = np.random.randint(0,n)
                if j not in preassigned_inds:
                    cluster_assignments[j][k] = 1
                    preassigned_inds.append(j)
                    a = False

        #randomly assign all other data points to clusters
        for i in range(n):
            if i not in preassigned_inds:
                cluster_assignments[i][np.random.randint(0,self.k)] = 1

        #-------------------------centroid calculation-------------------------

        print(mat)
        k_eff = self.k

        for i in range(self.max_iter):
            print(i)
            print(cluster_assignments)

            #compute the centroid of each cluster

            for k in range(k_eff):
                # get elements of each cluster
                ke = [e for x, e in enumerate(mat) if cluster_assignments[x][k] == 1]

                #print(ke)

                if len(ke) != 0:
                    self.cluster_centers.append(np.mean(np.stack(ke), axis=0))
                else:
                    #if no points are assigned to a cluster, reduce the cluster number
                    k_eff -= 1


            #reassign each observation to a new cluster

            cluster_assignments = np.zeros([n, k_eff])
            for x, e in enumerate(mat):
                #calculate the distance from the data point to each [new] centroid
                centroid_dists = [np.linalg.norm(e-self.cluster_centers[k]) for k in range(k_eff)]
                #set the assignments variable in the column corresponding to that cluster to 1
                cluster_assignments[x][centroid_dists.index(min(centroid_dists))] = 1

            print(self.cluster_centers)

        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """


km = KMeans(4, .5, 3)
km.fit(np.random.rand(10,2))