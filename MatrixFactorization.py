def split_inbatch(X, m):
    N = len(X)
    redundant = N % m
    shape = N // m
    first_batches = X[:m * shape].reshape(shape, -1)
    last_batch = X[-m:]
    return np.concatenate((first_batches, [last_batch]))

def make_model(Pu, Qi):
    return (Pu * Qi).sum(axis=1)

def grad(theta, r):
    Pu, Qi = theta
    e = r - make_model(*theta)
    dP = -e[:, np.newaxis] * Qi
    dQ = -e[:, np.newaxis] * Pu
    return np.array([dP, dQ])

def gradient_descent(dataset, nb_users, nb_movies, u_label='userId', i_label='movieId', values_label='rating',
                    gamma=0.01, lambda_=1.e-5, nb_epochs=10, batch_size=32, gradfunc=vanilla):
    u_data = dataset[u_label].values
    i_data = dataset[i_label].values
    v_data = dataset[values_label].values
    P = normal(size = (nb_users,k))
    Q = normal(size = (nb_movies,k))
    N = dataset.shape[0]

    print('gradient function', gradfunc)

    for e in range(nb_epochs):
        rmse = np.sqrt(mean_squared_error(v_data, make_model(P[u_data], Q[i_data])))
        print('epoch', e+1, '/', nb_epochs, 'RMSE', rmse)

        # get batches
        rdm = np.random.choice(range(N), N)
        batches = split_inbatch(rdm, batch_size)
        for batch in batches:
            u = u_data[batch]
            i = i_data[batch]
            r_ui = v_data[batch]

            theta = np.array((P[u, :], Q[i, :]))

            Pu, Qi = gradfunc(theta, r_ui)
            P[u] = Pu
            Q[i] = Qi

            if np.any(np.isnan(e_ui)):
                raise
    return P, Q




class Descender():
    def __init__(self, **params):
        self.params = {'gamma':0.9, 'eta':2.e-4, 'eps':1.e-8, 'b1':0.9, 'b2':0.999}
        self.params.update(params)
        self.nu = 0
        self.m = 0

    def make_model(self, Pu, Qi):
        return (Pu * Qi).sum(axis=1)

    def grad(self, theta, r):
        Pu, Qi = theta
        e = (r - self.make_model(*theta)).reshape(-1)
        dP = -e[:, np.newaxis] * Qi
        dQ = -e[:, np.newaxis] * Pu
        return np.array([dP, dQ])

class Vanilla(Descender):
    def __call__(self, theta, r):
        return theta - self.params['eta'] * self.grad(theta, r)

class Momentum(Descender):
    def __call__(self, theta, r):
        self.nu = self.params['gamma'] * self.nu + self.params['eta'] * self.grad(theta, r)
        return theta - self.nu

class NAG(Descender):
    def __call__(self, theta, r):
        self.nu = self.params['gamma'] * self.nu + self.params['eta'] * self.grad(theta - gamma * self.nu, r)
        return theta - self.nu

class Adagrad(Descender):
    def __call__(self, theta, r):
        try:
            self.G
        except AttributeError:
            self.G = np.zeros_like(theta)
        g = self.grad(theta, r)
        theta = self.params['eta'] / (np.sqrt(self.params['eps'] * self.G) * g).sum(axis=-1)
        print(theta.shape)
        self.G += g**2
        return theta

class Adam(Descender):
    def _make_g(self, theta, r):
        return np.mean(self.grad(theta), axis=0)

    def __call__(self,  theta, r):
        eps, b1, b2, eta = (self.params['eps'], self.params['b1'],
                            self.params['b2'], self.params['eta'])
        g = self._make_g(theta, r)
        self.m = b1 * self.m + (1 - b1) * g
        self.nu = b2 * self.nu + (1 - b2) * g**2
        mhat = self.m / (1 - b1)
        nuhat = self.nu / (1 - b2)
        return theta - eta / (sqrt(nuhat) + eps) @ mhat
