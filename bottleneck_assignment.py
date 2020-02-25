import pulp
import pandas as pd
import numpy as np


class BottleneckAssignmentHelper:
    def __init__(self, prob_increases, capacities, init_c_star=None):
        # `prob_increases` should be a n x m matrix
        # `capacities` should be an array with length m
        self.cost_matrix = prob_increases
        self.capacities = capacities

        if init_c_star is None:
            self.c_star = np.concatenate([
                self.cost_matrix.min(axis=0),
                self.cost_matrix.min(axis=1)
            ]).max()
        else:
            self.c_star = init_c_star
        self.c_star_candidates = np.unique(self.cost_matrix)
        self.c_star_candidates = self.c_star_candidates[
            self.c_star_candidates > self.c_star]

    def solve(self, multiplier=10):
        replacement = self.cost_matrix.max().max() * multiplier
        count = 0

        while len(self.c_star_candidates) > 0:
            print(f'c*: {self.c_star:.7f}, count: {count}')

            temp_matrix = self.cost_matrix.copy()
            valid_matrix = np.ones(temp_matrix.shape)

            temp_matrix[temp_matrix > self.c_star] = replacement
            valid_matrix[temp_matrix > self.c_star] = 0

            ### Use Gurobi to find solution
            self.x = pulp.LpVariable.dicts(
                'assignment',
                [(row, col)
                 for row in range(temp_matrix.shape[0])
                 for col in range(temp_matrix.shape[1])
                 if valid_matrix[row, col]],
                cat='Binary'
            )

            self.prob = pulp.LpProblem('bottleneck', pulp.LpMinimize)
            self.prob += pulp.lpSum(
                self.x[key] * self.cost_matrix[key[0], key[1]]
                for key in self.x
            )

            for row in range(temp_matrix.shape[0]):
                self.prob += pulp.lpSum(
                    self.x[(row, col)]
                    for col in range(temp_matrix.shape[1])
                    if valid_matrix[row, col]
                ) == 1

            for col in range(temp_matrix.shape[1]):
                self.prob += pulp.lpSum(
                    self.x[(row, col)]
                    for row in range(temp_matrix.shape[0])
                    if valid_matrix[row, col]
                ) <= self.capacities[col]

            self.prob.solve(solver=pulp.solvers.GUROBI_CMD())
            if pulp.LpStatus[self.prob.status] == 'Optimal':
                assignment_matrix = np.zeros(temp_matrix.shape)
                for key in self.x:
                    if valid_matrix[key[0], key[1]]:
                        assignment_matrix[key[0], key[1]] = self.x[key].varValue

                return assignment_matrix

            self.c_star = self.c_star_candidates[0]
            self.c_star_candidates = self.c_star_candidates[1:]
            count += 1

        return False


class BottleneckAssignmentHelperV2:
    def __init__(self, prob_increases, capacities):
        # `prob_increases` should be a n x m matrix
        # `capacities` should be an array with length m
        self.cost_matrix = prob_increases
        self.capacities = capacities

        self.lower_c_star = np.concatenate([
            self.cost_matrix.min(axis=0),
            self.cost_matrix.min(axis=1)
        ]).max()
        self.c_star_candidates = np.unique(self.cost_matrix)
        self.upper_c_star = self.c_star_candidates[-1]

        self.c_star_candidates = self.c_star_candidates[
            self.c_star_candidates > self.lower_c_star]

    def solve(self, multiplier=10, verbose=False):
        replacement = self.cost_matrix.max().max() * multiplier

        def is_feasible(c_star):
            temp_matrix = self.cost_matrix.copy()
            valid_matrix = np.ones(temp_matrix.shape)

            temp_matrix[temp_matrix > c_star] = replacement
            valid_matrix[temp_matrix > c_star] = 0

            # Use Gurobi to find any feasible solution
            x = pulp.LpVariable.dicts(
                'assignment',
                [(row, col)
                 for row in range(temp_matrix.shape[0])
                 for col in range(temp_matrix.shape[1])
                 if valid_matrix[row, col]],
                 cat='Binary'
            )

            prob = pulp.LpProblem()
            prob += pulp.lpSum(
                x[key] * temp_matrix[key[0], key[1]]
                for key in x
            )

            for row in range(temp_matrix.shape[0]):
                prob += pulp.lpSum(
                    x[(row, col)]
                    for col in range(temp_matrix.shape[1])
                    if valid_matrix[row, col]
                ) == 1

            for col in range(temp_matrix.shape[1]):
                prob += pulp.lpSum(
                    x[(row, col)]
                    for row in range(temp_matrix.shape[0])
                    if valid_matrix[row, col]
                ) <= self.capacities[col]

            prob.solve(solver=pulp.solvers.GUROBI_CMD())
            if pulp.LpStatus[prob.status] == 'Optimal':
                return True

            return False

        if is_feasible(self.lower_c_star):
            return self.lower_c_star
        if not is_feasible(self.upper_c_star):
            return False

        while len(self.c_star_candidates) > 2:
            if verbose:
                print(f'Searching between {self.lower_c_star} and {self.upper_c_star}')

            mid_c_star_id = len(self.c_star_candidates) // 2

            if is_feasible(self.c_star_candidates[mid_c_star_id]):
                self.c_star_candidates = self.c_star_candidates[: mid_c_star_id + 1]
                self.upper_c_star = self.c_star_candidates[-1]
            else:
                self.c_star_candidates = self.c_star_candidates[mid_c_star_id:]
                self.lower_c_star = self.c_star_candidates[0]

        return self.c_star_candidates
