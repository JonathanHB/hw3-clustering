import numpy as np
import random
from scipy.spatial.distance import cdist
import cluster
import cluster.utils as cu

class KMeans:

    def _check_numerical(self, x, req_int=True, req_nonzero = True):

        #check datatype

        if req_int:
            if not isinstance(x, int):
                #note that this does not inspect the numerical value of the input, so 1.00 would fail
                raise ValueError("input must be an integer")
        else:
            if not isinstance(x, float):
                raise ValueError("input must be a floating point number")

        #check value
        if req_nonzero:
            if x <= 0:
                raise ValueError("input must be positive")
        else:
            if x < 0:
                raise ValueError("input must be non-negative")


    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):

        #check for valid inputs
        self._check_numerical(k)
        self._check_numerical(tol, req_int=False, req_nonzero=False)
        self._check_numerical(max_iter)

        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        #store the cluster centers found by fit()
        self.cluster_centers = np.zeros([k, 2])
        self.m = 0
        self.fit_mat = np.array(0)
        self.fit_mat_labels = []


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

    #the seed argument is for unit testing
    def fit(self, mat: np.ndarray, seed=False):

        #-------------------------process input-------------------------
        #check for correct input datatype and structure and a reasonable k and n
        cu._inspect_matrix(mat=mat, n_min=self.k, n_dims=2)

        n = mat.shape[0]
        self.m = mat.shape[1]
        self.fit_mat = mat

        #-------------------------assignment generation-------------------------

        #NOTE: this method uses a 2d assignment array with ones and zeros to indicate which cluster each observation belongs to

        #generate initial assignments matrix and randomize the assignment of each observation
        cluster_assignments = np.zeros([n, self.k])

        #This step is statistically unlikely to be needed for large inputs where k is much smaller than
        #the number of observations (n), but is important for small n or k close to n.
        #If n is small or k is close to n, clustering should not be performed at all (or k should be reduced).
        #It was nevertheless necessary to make the code robust in this case.

        #assign one randomly chosen data point to each cluster so that no cluster begins empty
        #if multiple clusters started empty their cluster centers would pile up at the origin and
        # data points near them might subsequently have multiple equally close cluster centers

        # preassigned_inds = []
        # for k in range(self.k):
        #     a = True
        #     while a:
        #         j = np.random.randint(0,n)
        #         if j not in preassigned_inds:
        #             cluster_assignments[j][k] = 1
        #             preassigned_inds.append(j)
        #             a = False

        #randomly assign data points to clusters
        #predictable output for unit testing
        if seed:
            np.random.seed(0)

        for i in range(n):
            #if i not in preassigned_inds:
            cluster_assignments[i][np.random.randint(0,self.k)] = 1

        #calculate initial error; used for tolerance check below
        #self.fit_mat_labels = cluster_assignments

        #-------------------------centroid calculation-------------------------

        #k_eff = self.k

        for i in range(self.max_iter):

            #compute the centroid of each cluster
            #self.cluster_centers = []

            for k in range(self.k):
                # get elements of each cluster
                ke = [e for x, e in enumerate(mat) if cluster_assignments[x][k] == 1]

                #print(ke)

                if len(ke) != 0:
                    cctr = np.mean(np.stack(ke), axis=0)
                    self.cluster_centers[k][0] = cctr[0]
                    self.cluster_centers[k][1] = cctr[1]
                else:
                    #if no points are assigned to a cluster, reduce the cluster number
                    print(f"\nNo points were assigned to cluster {k}; cluster center will not move.")


            #reassign each observation to a new cluster

            #this loop has some redundancy with the contents of get error; the two could be combined into
            # an auxiliary method with some modification to make the code cleaner and faster
            cluster_assignments = np.zeros([n, self.k])
            for x, e in enumerate(mat):
                #calculate the distance from the data point to each [new] centroid
                centroid_dists = [np.dot(e - kc, e - kc) for kc in self.cluster_centers]
                #set the assignments variable in the column corresponding to that cluster to 1
                cluster_assignments[x][centroid_dists.index(min(centroid_dists))] = 1

            self.fit_mat_labels = cluster_assignments

            #check clustering error and compare change therein to tolerance to determine whether to terminate early

            err_i = self.get_error()

            #don't reference the undefined last error in the initial round
            if i == 0:
                self.last_error = err_i
                continue

            delta_error = self.last_error-err_i
            self.last_error = err_i

            if delta_error <= self.tol:
                print(f"\nTerminating after {i} iterations as error change of {delta_error} is below tolerance.")
                return

        print(f"\nTerminating after maximum {self.max_iter} iterations")

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

        cu._inspect_matrix(mat=mat, n_min=self.k, n_dims=2)

        if mat.shape[1] != self.m:
            raise ValueError(f"Input has {mat.shape[1]} observations per feature, but model was fit with {self.m}.\n Fitting and prediction must have the same number of observations per feature.")

        #note that a 1d array of [integer] cluster assignments is used here
        cluster_assignments = []
        for x, e in enumerate(mat):
            # calculate the distance from the data point to each [new] centroid
            centroid_dists = [np.dot(e - k, e - k) for k in self.cluster_centers]
            # set the assignments variable in the column corresponding to that cluster to 1
            cluster_assignments.append(centroid_dists.index(min(centroid_dists)))

        return cluster_assignments

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

        total_sme = 0

        for x, e in enumerate(self.fit_mat):
            # calculate the distance from the data point to each [new] centroid
            centroid_dists = [np.dot(e - k, e - k) for k in self.cluster_centers]
            # set the assignments variable in the column corresponding to that cluster to 1
            total_sme += min(centroid_dists)

        return total_sme/self.fit_mat.shape[0]

        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

    def get_centroids(self) -> np.ndarray:

        return self.cluster_centers

        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

#plotting method for graphical inspection
def _clustering_test_plot(k, seed, kmseed):

    cl_in = cu.make_clusters(k=k, seed = seed)
    km = KMeans(k, .0005, 90)

    km.fit(cl_in[0], kmseed)

    centroids = km.get_centroids()
    print(f"SME = {km.get_error()}")

    predlabels = km.predict(cl_in[0])

    plotpoints = np.concatenate([cl_in[0], centroids])
    truelanels_c = np.concatenate([cl_in[1], [k]*k])
    predlabels_c = np.concatenate([predlabels, [k]*k])

    cu.plot_multipanel(
            plotpoints,
            truelanels_c,
            predlabels_c,
            np.ones(cl_in[0].shape[0]+k))

#_clustering_test_plot(5, 16, True)

