import torchvision
import os
import sys
import torch



class KNN:
    def __init__(self):
        pass

    def knn_distance_matrix(test_x, train_x):
        """
        Given a set of testing and training data, compute a matrix of distances between all test points and all training points.

        Args:
            test_x (m x d tensor): Testing data that we want to make predictions for.
            train_x (n x d tensor): Training data that we'll use for making classifications.

        Returns:
            dist_mat (m x n tensor): a matrix D so that D_ij is the distance between the ith test point and the jth training point.

        Hints:
            - You may find it helpful to know that the **squared** Euclidean distance between two vectors x and z, ||x - z||^2, can be computed as (x - z)'(x - z) = x'x - 2*x'z + z'z.
            This formulation is much easier to implement using simple matrix multiplication, elementwise products, and broadcasting.
            - Implementing this function without for loops will be the trickiest part of implementing the algorithm.
        """

        testNorms = (test_x**2).sum(dim= 1, keepdim =True)
        trainNorms = (train_x**2).sum(dim=1).unsqueeze(0)
        crossTerm = test_x @ train_x.t()

        return torch.sqrt(torch.clamp((testNorms + trainNorms  -2*crossTerm), min=0.0))

    def knn_find_k_nearest_neighbors(D, k):
        """
        Given a distance matrix between test and train points, find the k nearest neighbors from the training points for each test point.

        Args:
            D (m x n tensor): a matrix D so that D_ij is the distance between the ith test point and jth training point (e.g., each *row* corresponds to a test point).
            k (int): How many nearest neighbors to find

        Returns:
            knn_inds (m x k tensor): For each test point, the indices of the k nearest neighbors in the training data. In other words, if knn_inds[i, j] = q, then (train_x[q], train_y[q]) is one of the k nearest neighbors to test_x[q].

        Hints:
            - This time, you can get some help from real functions in PyTorch. In particular, you may find the torch.topk function useful (https://pytorch.org/docs/stable/generated/torch.topk.html).
            - This function can be implemented in as little as 1 (relatively short) line of code. If you find yourself struggling, you're probably not using torch.topk as well as you should.
        """
        return torch.topk(D,k, dim = 1, largest= False).indices

    def knn_predict(train_y, knn_inds):
        """
        Given an m x k set of indices of nearest neighbors for each test point, use the training labels and these indices to make a final prediction for each test point.

        Args:
            train_y (n vector): Labels of the training data points
            knn_inds (m x k tensor): For each test point, the indices of the k nearest neighbors in the training data. In other words, if knn_inds[i, j] = q, then (train_x[q], train_y[q]) is one of the k nearest neighbors to test_x[q].

        Returns:
            predictions (m vector): A prediction of the label for each test point.

        Hints:
            - Suppose x is a vector with at least 6 entries and ix is a 2x3 matrix with entries [[0,2,4],[1,3,5]], then x[ix] will return a 2x3 matrix with entries [[x[0], x[2], x[4]],[x[1],x[3],x[5]]].
            - torch.mode (https://pytorch.org/docs/stable/generated/torch.mode.html) will be a pretty helpful function here.
        """
        return torch.mode(train_y[knn_inds], 1).values

    def knn_algorithm(self,test_x, train_x, train_y, k=3):
        """
        Put it all together!

        Args:
            test_x (m x d tensor): Test points to make predictions for
            train_x (n x d tensor): Training data to use for making predictions.
            train_y (n, tensor): Training labels to use for making predictions.
            k (int): k to use in k nearest neighbors

        Returns:
            predictions (m, tensor): Predicted label for each test point.
        """
        # First, flatten the 28x28 images into 784 dimensional vectors. We'll also divide by 255 so that the feature values
        # are in [0,1] instead of [0, 255]. This is important because otherwise the distances will be so large that we'll have
        # floating point error problems!

        train_x_flat = train_x.view(train_x.size(-3), -1).float() / 255.
        test_x_flat = test_x.view(test_x.size(-3), -1).float() / 255.
        disMatrix = self.knn_distance_matrix(test_x_flat, train_x_flat)
        nearestNeighbors = self.knn_find_k_nearest_neighbors(disMatrix, k)

        return self.knn_predict(train_y,nearestNeighbors)



if __name__ == "__main__":
        # Load the MNIST dataset
    mnist_train = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    train_x = mnist_train.data
    train_y = mnist_train.targets

    # MNIST contains digits of all different labels -- let's get just the 0 and 8 images.
    train_x = train_x[torch.logical_or(train_y == 0, train_y == 8)]
    train_y = train_y[torch.logical_or(train_y == 0, train_y == 8)]


    # Load the MNIST test dataset
    mnist_test = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    test_x = mnist_test.data
    test_y = mnist_test.targets

    # MNIST contains digits of all different labels -- let's get just the 0 and 8 images.
    test_x = test_x[torch.logical_or(test_y == 0, test_y == 8)]
    test_y = test_y[torch.logical_or(test_y == 0, test_y == 8)]
    torch.manual_seed(42)
    k = 16
    niters = 500
    knn = KNN()
    mus = knn.kmeans_init(X,k)
    clusters = torch.randint(0,k,(X.size(0),))

    for t in range(niters):
        prev_clusters = clusters
        clusters =  knn.kmeans_assign_clusters(X,mus)
        mus =  knn.kmeans_update_centroids(X,clusters)

        if  knn.kmeans_stopping_criteria(clusters, prev_clusters):
            break
    print(f"Ended after {t} iterations.")