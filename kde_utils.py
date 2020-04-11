import numpy as np
from sklearn.neighbors import KernelDensity


class KDEHelper:
    '''
    Gaussian KDE with unbounded data.
    '''

    def __init__(self, data, bandwidth_est='silverman'):
        self.data = data
        if bandwidth_est == 'silverman':
            self.bandwidth = 0.9 \
                * min(
                    data.std(),
                    np.subtract(*np.percentile(data, [75, 25])) / 1.34
                ) \
                * data.size ** (-1 / 5)
        else:
            self.bandwidth = bandwidth_est

        self.fit()

    def fit(self):
        self.model = KernelDensity(
            kernel='gaussian',
            bandwidth=self.bandwidth
        ).fit(self.data.flatten()[:, np.newaxis])

    def sample_and_transform(self, trans_fn=lambda x: 1 / (1 + np.exp(-x)),
            placeholder2=None, size=(1, 1)):

        samples = self.model.sample(n_samples=size[0] * size[1]).reshape(size)

        return trans_fn(samples)
