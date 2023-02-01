# Write your k-means unit tests here
import pytest
import numpy as np
import sklearn
from sklearn import cluster as skcl
import cluster.utils as cu
import cluster.kmeans as km

def test_kmean_centroids():

    #parameters for test data
    n=500
    k=5
    seed = 16
    #generate test data
    cl_in = cu.make_clusters(n=n, k=k, seed = seed)[0]

    max_iter = 100
    tol = 0.0001

    #cluster test data with this program's kmeans implementation
    #test_output = _clustering_test(5, cl_in, True)
    kme = km.KMeans(k, tol, max_iter)
    kme.fit(cl_in, True)

    #cluster test data with scipy
    true_output = skcl.KMeans(n_clusters=k, init='random', max_iter=max_iter, tol=tol).fit(cl_in)

    #print(kme.get_error())
    #print(true_output.inertia_/n)

    #check that the fits are equally good
    floating_point_tolerance = 0.0000001
    assert kme.get_error() - true_output.inertia_/n < floating_point_tolerance, f"local method's clustering MSE of {kme.get_error()} does not match scipy's MSE of {true_output.inertia_/n}"

    tcs = true_output.cluster_centers_
    true_sorted = tcs[tcs[:, 0].argsort()]

    pcs = kme.get_centroids()
    pred_sorted = pcs[pcs[:, 0].argsort()]

    for x in range(true_sorted.shape[0]):
        assert abs(true_sorted[x][0] - pred_sorted[x][0]) < floating_point_tolerance
        assert abs(true_sorted[x][1] - pred_sorted[x][1]) < floating_point_tolerance

