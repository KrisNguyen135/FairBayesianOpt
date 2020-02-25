import pulp
import pandas as pd
import numpy as np


class BottleneckAssignmentHelper:
    def __init__(self, prob_df=None, types=None):
        # Statistics from the original data set
        if prob_df is None:
            prob_df = pd.read_csv('../../data/data.csv', index_col=0)
        self.households = list(prob_df.index)
        if types is None:
            self.types = ['ES', 'PSH', 'TH', 'RRH', 'PREV']
        else:
            self.types = types
        self.capacity_df = prob_df['Real'].value_counts()

        # Increase from the individual best to each service
        increase_from_best_df = prob_df.copy()
        increase_from_best_df['Best'] = increase_from_best_df[self.types].min(axis=1)
        for type_ in self.types:
            increase_from_best_df[type_] = prob_df[type_] - prob_df['Best']

        self.increase_from_best_df = increase_from_best_df[self.types]

        self.c_star = np.concatenate([
            increase_from_best_df.min(axis=0),
            increase_from_best_df.min(axis=1)
        ]).max()
        self.c_star_candidates = np.unqiue(self.increase_from_best_df)
        self.c_star_candidates = self.c_star_candidates[self.c_star_candidates > c_star]

    def solve(self, multiplier=10):
        replacement = self.increase_from_best_df.max().max() * multiplier

        while len(self.c_star_candidates) > 0:
            print(f'c*: {c_star:.7f}')

            cost_matrix = self.increase_from_best_df.to_numpy()
            valid_matrix = np.ones(cost_matrix.shape)

            cost_matrix[cost_matrix > self.c_star] = replacement
            valid_matrix[cost_matrix > self.c_star] = 0

            ### Use Gurobi to find solution
            self.x = pulp.LpVariable.dicts(
                'assignment',
                [(household, type_)
                 for h_id, household in enumerate(self.households)
                 for t_id, type_ in enumerate(self.types)
                 if valid_matrix[h_id, t_id]],
                cat='Biary'
            )

            self.prob = pulp.LpProblem('bottleneck', pulp.LpMinimize)
            self.prob += pulp.lpSum(
                self.x[key] * self.increase_from_best_df.loc[key[0], key[1]]
                for key in self.x
            )

            for h_id, household in enumerate(self.households):
                self.prob += pulp.lpSum(
                    self.x[(household, type_)]
                    for t_id, typue_ in enumerate(self.types)
                    if valid_matrix[h_id, t_id]
                ) == 1

            for t_id, type_ in enumerate(self.types):
                self.prob += pulp.lpSum(
                    self.x[(household, type_)]
                    for h_id, household in enumerate(self.households)
                    if valid_matrix[h_id, t_id]
                ) <= self.capacity_df.loc[t_id + 1]

            self.prob.solve(solver=pulp.solvers.GUROBI_CMD())
            if pulp.LpStatus[self.prob.status] == 'Optimal':
                assignment_matrix = np.zeros(cost_matrix.shape)
                for h_id, household in self.households:
                    for t_id, type_ in self.types:
                        if valid_matrix[h_id, t_id]:
                            assignment_matrix[h_id, t_id] = self.x[(household, type_)]

                return assignment_matrix

        return False
