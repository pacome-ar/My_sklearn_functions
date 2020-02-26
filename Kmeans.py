import numpy as np

class Kmeans():
    '''Kmeans clustering

    Parameters:
    -----------
    n_clusters: int (defaults 3)
        The number of clusters into which to separate the data
    random_state: int, str, float
        Seed to initalise the internal random RandomState
    maxiter: int (defaults to 100)
        Maximum number of iterations to do in the fit function
    init: str (defaults to 'centroids')
        strategy for the initialisation
          if 'centroids': starts by choosing centroids randomly in X
          if 'labels': starts by randomly assigning lables to X
          if 'K++': starts using the kmean++ algorithm
    n_init: int (defaults to 1)
        number of kmean to train. the final model will be the one with
        the lowest inertia

    Example:
    --------
    >>> import numpy as np
    >>> from Kmeans import Kmeans
    >>> X = np.random.rand(500, 2)
    >>> kmean = Kmeans(n_clusters=3)
    >>> y = kmean.fit(X)
    '''

    def __init__(self, n_clusters=3, random_state=42, maxiter=100,
                        init='centroids', n_init=1):
        '''Init function'''
        self.n_clusters = n_clusters
        self._rs = np.random.RandomState(random_state)
        self.maxiter = maxiter
        self.init = init
        self.n_init = n_init

    def _pick_within_proba(self, proba):
        '''returns a random number picked with proba'''
        assert np.allclose(np.sum(proba), 1.), 'sum of proba should be 1'
        cumsum = np.cumsum(proba)
        return np.arange(len(cumsum))[cumsum >= self._rs.rand()][0]

    def _init_random_labels(self, X):
        '''returns initial randomly picked classes for the data X'''
        return self._rs.randint(0, self.n_clusters, len(X))

    def _init_random_centroids(self, X):
        '''returns initial randomly picked classes for the data X'''
        rdm_index = self._rs.randint(0, len(X), self.n_clusters)
        return X[rdm_index]

    def _init_kmeanpp(self, X):
        centroids = [X[self._rs.randint(0, len(X)+1)]]
        for i in range(self.n_clusters):
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
        for i in range(self.n_clusters):
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
        '''computes the inertia: sum(distance to closest centroid)**2'''
        dists = self._distance_to_centroid(X, centroids)
        return dists.min(axis=0).sum()

    def _init_fit(self, X):
        '''returns the initilisation corresponding to the self.init param'''
        if self.init == 'centroids':
            centroids = self._init_random_centroids(X)
            labels = self._get_clusters(X, centroids)
        elif self.init == 'K++':
            centroids = self._init_kmeanpp(X)
            labels = self._get_clusters(X, centroids)
        elif isinstance(self.init, (list, np.ndarray)):
            centroids = self.init
            labels = self._get_clusters(X, centroids)
        elif self.init == 'labels':
            centroids = self._init_random_centroids(X)
            labels = self._init_random_labels(X)
        return centroids, labels

    def _run_kmean(self, X):
        '''runs one istance of kmean:
        initialise according to init parameter
        then compute centroid / compute label until labels are stable
        returns (centroids, labels, inertia, label_keep, centroid_keep)
        '''
        centroids, labels = self._init_fit(X)
        centroid_keep, label_keep = [centroids], [labels]
        for i in range(self.maxiter):
            centroids = self._get_centroids(X, labels)
            newlabels = self._get_clusters(X, centroids)
            label_keep.append(newlabels)
            centroid_keep.append(centroids)
            if np.all(labels == newlabels):
                break
            labels = newlabels

        inertia = self._get_inertia(X, centroids)
        results = (centroids, labels, inertia, label_keep, centroid_keep)
        return results

    def fit(self, X):
        '''fit function: trains a kmean solver'''
        assert isinstance(X, np.ndarray), 'X must be a numpy array'
        if self.n_init == 1:
            (centroids, labels, inertia,
            label_keep, centroid_keep) = self._run_kmean(X)
        else:
            results = []
            inertias = []
            for i in range(self.n_init):
                result = self._run_kmean(X)
                results.append(result)
                inertias.append(result[2])
            index = np.argmin(inertias)
            (centroids, labels, inertia,
            label_keep, centroid_keep) = results[index]

        self.centroids = centroids
        self.inertia = inertia
        self.labels = labels
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

    kmean = Kmeans(n_clusters=3, maxiter=15, random_state=1,
              init='centroids')
    kmean.fit(X)

    assert np.allclose(kmean.inertia, sk.inertia_), \
            'inertia test failed, got {} instead of {}'.format(
            kmean.inertia, sk.inertia_)

    print('Inertia test passed')

#########################################

test_inertia_with_sklearn(200)
