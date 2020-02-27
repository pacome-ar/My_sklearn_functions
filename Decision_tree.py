import numpy as np

##############################""

def gini(y):
    _, count = np.unique(y, return_counts=True)
    return 1 - ((count / count.sum())**2).sum()

def entropy(y):
    _, count = np.unique(y, return_counts=True)
    return - (count * np.log(count) / np.log(2)).sum()

def make_minimizer(y, i, func):
    N = len(y)
    return i / N * func(y[:i]) + (N - i) / N * func(y[:i])

def single_feature_node_choice(single_feature, y, function=gini):
    args = np.argsort(single_feature)
    N = len(y)
    Xsort = single_feature[args]
    ysort = y[args]
    thresholds, counts = np.unique(Xsort, return_counts=True)

    minimizer = []
    for i in counts.cumsum():
        minimizer.append(make_minimizer(y, i, function))

    return np.array([thresholds, minimizer])

def node_choice(X, y, func=gini):
    '''lots of repeated calculation but important is that it works
    Returns:
    --------
    ret: list
        the feature k and threshold t that minimize the node choice'''
    temp = []
    mins = []
    for k in range(X.shape[1]):
        ts, ms = single_feature_node_choice(X[:,k], y, func)
        args = np.argmin(ms)
        temp.append([k, ts[args]])
        mins.append(ms[args])

    return temp[np.argmin(mins)]

def make_left_mask(X, k, t):
    return X[:, k] <= t

def split(X, y, val):
    if val is None:
        return X, y
    test = make_left_mask(X, *val)
    if test.sum() != len(y):
        leftX, rightX = X[test], X[~test]
        lefty, righty = y[test], y[~test]
        return (leftX, lefty), (rightX, righty)
    else:
        return None

########################################

class Node():
    '''Simple container'''
    def __init__(self, coefs, labels, values, gini, samples, classlabel):
        self.coefs = coefs
        self.labels = labels
        self.values = values
        self.gini = gini
        self.samples = samples
        self.classlabel = classlabel

def make_node(X, y, leaf=False):
    if not len(y):
        return None
    if leaf:
        coefs = None
    else:
        coefs = node_choice(X, y)
    labels, values = np.unique(y, return_counts=True)
    gini_ = gini(y)
    samples = len(y)
    classlabel = labels[np.argmax(values)]

    return Node(coefs, labels, values, gini_, samples, classlabel)

##########################################

def recurse_make_node(X, y, parent=None, depth=0, maxdepth=5, minlen=3):

    if len(X) < minlen or depth > maxdepth:
        return make_node(X, y, True)

    node = make_node(X, y)
    spliting = split(X, y, node.coefs)
    if not spliting:
        return make_node(X, y, True)

    (leftX, lefty), (rightX, righty) = spliting
    lrec = recurse_make_node(leftX, lefty, parent, depth+1, maxdepth, minlen)
    rrec = recurse_make_node(rightX, righty, parent, depth+1, maxdepth, minlen)
    parent = [node, [lrec, rrec]]
    return parent

def make_decision(x, val):
    k, t = val
    return x[k] <= t

def run_tree(tree, x0):
    subtree = tree.copy()
    while True:
        if isinstance(subtree, Node):
            break
        node, subtree = subtree
        test = make_decision(x0, node.coefs)
        if test:
            subtree = subtree[0]
        else:
            subtree = subtree[1]
    return node

def predict(tree, X):
    return [run_tree(tree, x).classlabel for x in X]

#############################################

rs = np.random.RandomState(25)
X = rs.randint(0, 5 ,(10, 3))
y = rs.randint(0, 3, len(X))
x0 = X[0]

tree = recurse_make_node(X, y, maxdepth=15, minlen=1)

ys = predict(tree, X)

list(zip(ys, y))
