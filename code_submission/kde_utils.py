import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.neighbors import KernelDensity
from scipy.special import logit

import matplotlib.pyplot as plt


class KDEHelper:
    '''
    Gaussian KDE with unbounded data.
    '''

    def __init__(self, data, bandwidth_est='silverman'):
        # Transform data to the real space
        if np.all(data >= 0) and np.all(data <= 1):
            self.data = logit(data)
        else:
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

    def get_transformed_pdf(self, xs=np.linspace(0, 1, 1002)[1: -1].reshape(-1, 1)):
        pdfs = np.exp(self.model.score_samples(logit(xs))).flatten() \
            / (xs * (1 - xs)).flatten()

        return pdfs


if __name__ == '__main__':
    # Extract the probabilities into a flat np array
    data = pd.read_csv('data.csv', index_col=0)[
        ['ES', 'PSH', 'TH', 'RRH', 'PREV']
    ].to_numpy().flatten()

    # Transform to the inverse sigmoid space and feed into the KDE API
    kde = kde_utils.KDEHelper(np.log(data) - np.log(1 - data))
    samples = kde.sample_and_transform(size=(10000, 1))  # draw 10,000 samples

    # Compare with the original data
    plt.hist(data, bins=50, alpha=0.4, density=True)
    plt.hist(samples, bins=50, alpha=0.4, density=True)
    plt.show()
