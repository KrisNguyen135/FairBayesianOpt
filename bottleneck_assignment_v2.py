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
