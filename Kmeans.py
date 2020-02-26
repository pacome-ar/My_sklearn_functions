from scipy.spatial import KDTree
import numpy as np

class Kmeans():
    '''Kmeans clustering

    Parameters:
    -----------
    nbclusters: int (defaults 3)
        The number of clusters into which to separate the data
    random_state: int, str, float
        Seed to initalise the internal random RandomState
    maxiter: int (defaults to 10)
        Maximum number of iterations to do in the fit function
    init_start: str (defaults to 'centroids')
        strategy for the initialisation
          if 'centroids': starts by choosing centroids randomly in X
          if 'labels': starts by randomly assigning lables to X
          if 'K++': starts using the kmean++ algorithm
    init_centroids: array like (defaults to None)
        the initial guesses for the new_centroids
        will only be used with init_start = 'centroids'

    Example:
    --------
    >>> import numpy as np
    >>> from Kmeans import Kmeans
    >>> X = np.random.rand(500, 2)
    >>> kmean = Kmeans(nbclusters=3, maxiter=15)
    >>> y = kmean.fit(X)
    '''

    def __init__(self, nbclusters=3, random_state=42, maxiter=10,
                        init_start='centroids', init_centroids=None):
        '''Init function'''
        self.nbclusters = nbclusters
        self._rs = np.random.RandomState(random_state)
        self.maxiter = maxiter
        self.init_start = init_start
        self.init_centroids = init_centroids

    def _pick_within_proba(self, proba):
        '''returns a random number picked with proba'''
        assert np.allclose(np.sum(proba), 1.), 'sum of proba should be 1'
        cumsum = np.cumsum(proba)
        return np.arange(len(cumsum))[cumsum >= self._rs.rand()][0]

    def _init_random_labels(self, X):
        '''returns initial randomly picked classes for the data X'''
        return self._rs.randint(0, self.nbclusters, len(X))

    def _init_random_centroids(self, X):
        '''returns initial randomly picked classes for the data X'''
        if self.init_centroids:
            return init_centroids
        rdm_index = self._rs.choice(range(len(X)),
                                    self.nbclusters,
                                    replace=False)
        return X[rdm_index]

    def _init_kmeanpp(self, X):
        centroids = [X[self._rs.randint(0, len(X)+1)]]
        for i in range(self.nbclusters):
            proba = self._distance_to_centroid(
                        X, np.array(centroids, copy=True)).min(axis=0)
            proba = proba / proba.sum()
            centroids.append(X[self._pick_within_proba(proba)])
        return centroids

    def _distance_to_centroid(self, X, centroids):
        '''returns the array of the distances between X and the centroids
        If X is size n*m and centroids is size k
            returns a n*k array
        '''
        return np.array([((X - x0)**2).sum(axis=1)
                         for x0 in centroids])

    def _get_centroids(self, X, labels):
        '''returns the average coordinates of X for each label in labels'''
        new_centroids = []
        for i in range(self.nbclusters):
            temp = X[labels==i]
            if not len(temp):
                temp = np.zeros((1, X.shape[1]))
            new_centroids.append(temp.mean(axis=0))
        return np.array(new_centroids)

    def _get_clusters(self, X, centroids):
        '''returns the new labels for X given coordinates of centroids'''
        dists = self._distance_to_centroid(X, centroids)
        return np.argmin(dists, axis=0)

    def _get_inertia(self, X, centroids):
        dists = self._distance_to_centroid(X, centroids)
        return dists.min(axis=0).sum()

    def fit(self, X, debug=False):
        '''fit function: trains a kmean solver'''

        if self.init_start == 'centroids':
            centroids = self._init_random_centroids(X)
            labels = self._get_clusters(X, centroids)
        if self.init_start == 'K++':
            centroids = self._init_kmeanpp(X)
            labels = self._get_clusters(X, centroids)
        elif self.init_start == 'labels':
            centroids = self._init_random_centroids(X)
            labels = self._init_random_labels(X)

        centroid_keep, label_keep = [centroids], [labels]

        for i in range(self.maxiter):
            centroids = self._get_centroids(X, labels)
            newlabels = self._get_clusters(X, centroids)

            if debug:
                label_keep.append(labels)
                centroid_keep.append(centroids)

            if np.all(labels == newlabels):
                break
            labels = newlabels

        self.centroids = centroids
        self.labels = self.predict(X)
        self.inertia = self._get_inertia(X, centroids)

        if debug:
            label_keep.append(self.labels)
            self.keep_labels = label_keep
            self.keep_centroids = centroid_keep

        return self.labels

    def predict(self, X):
        '''Use previously trained kmean to predict the classes for X'''
        try:
            self.centroids
        except ValueError:
            raise Exception('can not transform an unfitted kmean')

        return self._get_clusters(X, self.centroids)

#########################################

def test_inertia_with_sklearn(n=200):
    from sklearn.cluster import KMeans as skKmeans
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n, centers=3, n_features=2,
                  random_state=1)
    # sklearn
    sk = skKmeans(n_clusters=3)
    sk.fit(X)

    kmean = Kmeans(nbclusters=3, maxiter=15, random_state=1,
              init_start='centroids')
    kmean.fit(X)

    assert np.allclose(kmean.inertia, sk.inertia_), \
            'inertia test failed, got {} instead of {}'.format(
            kmean.inertia, sk.inertia_)

    print('Inertia test passed')

#########################################

test_inertia_with_sklearn(200)
