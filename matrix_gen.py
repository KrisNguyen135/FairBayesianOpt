import numpy as np


class MatrixGenerator:
    def __init__(self, dist, dist_params, n_agents, n_intvs):
        self.dist = dist
        self.dist_params = dist_params
        self.n_agents = n_agents
        self.n_intvs = n_intvs

    def get_new_matrix(self, sort_rows=False, sort_cols=False):
        '''
        Only one of `sort_rows` and `sort_cols` should be True.
        '''

        matrix = self.dist(
            *self.dist_params,
            size=(self.n_agents, self.n_intvs)
        ).round(5)

        if sort_rows:
            matrix.sort(axis=1)
        elif sort_cols:
            matrix.sort(axis=0)

        return matrix

    def get_matrix_pair(self, other_dist, other_dist_params,
            sort_rows=False, sort_cols=False):
        '''
        Return two matrices generated from two separated distributions
        but the elements are in the same order.
        '''

        # Generate the first matrix
        matrix = self.dist(
            *self.dist_params,
            size=(self.n_agents, self.n_intvs)
        ).round(5)

        if sort_rows:
            matrix.sort(axis=1)
        elif sort_cols:
            matrix.sort(axis=0)

        sorting_indices = np.argsort(matrix, axis=None)
        arranging_indices = np.argsort(sorting_indices)

        # Generate the other matrix
        other_matrix = other_dist(
            *other_dist_params,
            size=(self.n_agents, self.n_intvs)
        ).round(5)

        other_matrix = np.sort(
            other_matrix.flatten()
        )[arranging_indices].reshape((self.n_agents, self.n_intvs))

        return matrix, other_matrix

    def get_matrix_list(self, n_extra_matrices, other_dists, other_dists_params,
            sort_rows=False, sort_cols=False):
        '''
        Return a list of matrices generated from multiple distributions but
        the elements are in the same order across all matrices.
        '''

        # Generate the first matrix
        matrix = self.dist(
            *self.dist_params,
            size=(self.n_agents, self.n_intvs)
        ).round(5)

        if sort_rows:
            matrix.sort(axis=1)
        elif sort_cols:
            matrix.sort(axis=0)

        sorting_indices = np.argsort(matrix, axis=None)
        arranging_indices = np.argsort(sorting_indices)

        matrix_list = [matrix]
        for i in range(n_extra_matrices):
            other_matrix = other_dists[i](
                *other_dists_params[i],
                size=(self.n_agents, self.n_intvs)
            ).round(5)

            other_matrix = np.sort(
                other_matrix.flatten()
            )[arranging_indices].reshape((self.n_agents, self.n_intvs))

            matrix_list.append(other_matrix)

        return matrix_list

    def get_increase_matrix_pair(self, round_=True):
        '''
        Return a pair of matrices with the same elements drawn from `self.dist`.
        The elements in the first are randomly sorted; those in the second are
        row-sorted.
        '''
        matrix = self.dist(
            *self.dist_params,
            size=(self.n_agents, self.n_intvs)
        )
        if round_:
            matrix = matrix.round(5)
        sorted_matrix = matrix.copy()
        sorted_matrix.sort(axis=1)

        increase_matrix = matrix - np.repeat(
            matrix.min(axis=1), self.n_intvs
        ).reshape((self.n_agents, self.n_intvs))
        sorted_increase_matrix = increase_matrix.copy()
        sorted_increase_matrix.sort(axis=1)

        return (matrix, sorted_matrix), (increase_matrix, sorted_increase_matrix)
