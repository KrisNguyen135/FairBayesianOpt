import pulp
import pandas as pd
import numpy as np


class BottleneckAssignmentHelper:
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
            self.c_star_candidates >= self.lower_c_star]

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


class BottleneckAssignmentHelperV2:
    def __init__(self, increase_matrix, capacities):
        self.cost_matrix = increase_matrix
        self.n_agents = increase_matrix.shape[0]
        self.n_intvs = increase_matrix.shape[1]
        self.capacities = capacities

        # self.lower_c_star = np.concatenate([
        #     self.cost_matrix.min(axis=0),
        #     self.cost_matrix.min(axis=1)
        # ]).max()
        self.c_star_candidates = np.unique(self.cost_matrix)
        self.lower_c_star = self.c_star_candidates[0]
        self.upper_c_star = self.c_star_candidates[-1]

        self.c_star_candidates = self.c_star_candidates[
            self.c_star_candidates >= self.lower_c_star
        ]

    def is_feasible(self, c_star):
        temp_matrix = self.cost_matrix.copy()
        valid_matrix = np.ones(temp_matrix.shape)

        # temp_matrix[temp_matrix > c_star] = replacement
        valid_matrix[temp_matrix > c_star] = 0

        # Use Gurobi to find a feasible solution
        x = pulp.LpVariable.dicts(
            'assignment',
            [(agent_id, intv_id)
             for agent_id in range(self.n_agents)
             for intv_id in range(self.n_intvs)
             if valid_matrix[agent_id, intv_id]],
             cat='Binary'
        )

        prob = pulp.LpProblem()
        prob += pulp.lpSum(
            x[key] * temp_matrix[key[0], key[1]]
            for key in x
        )

        for agent_id in range(self.n_agents):
            prob += pulp.lpSum(
                x[(agent_id, intv_id)]
                for intv_id in range(self.n_intvs)
                if valid_matrix[agent_id, intv_id]
            ) == 1

        for intv_id in range(self.n_intvs):
            prob += pulp.lpSum(
                x[(agent_id, intv_id)]
                for agent_id in range(self.n_agents)
                if valid_matrix[agent_id, intv_id]
            ) <= self.capacities[intv_id]

        prob.solve(solver=pulp.solvers.GUROBI_CMD())
        if pulp.LpStatus[prob.status] == 'Optimal':
            assignments = []
            for agent_id in range(self.n_agents):
                for intv_id in range(self.n_intvs):
                    if valid_matrix[agent_id, intv_id] \
                            and x[(agent_id, intv_id)].varValue == 1:
                        assignments.append(intv_id)
                        break

            return np.array(assignments)

        return False

    def solve(self, multiplier=10, verbose=False):
        # replacement = self.cost_matrix.max() * multiplier

        potential_assignments = self.is_feasible(self.lower_c_star)
        if potential_assignments is not False:
            return self.lower_c_star, potential_assignments

        if self.is_feasible(self.upper_c_star) is False:
            return False

        while len(self.c_star_candidates) > 2:
            if verbose:
                print(f'Searching between {self.lower_c_star} and {self.upper_c_star}')

            mid_c_star_id = len(self.c_star_candidates) // 2

            if self.is_feasible(self.c_star_candidates[mid_c_star_id]) is not False:
                self.c_star_candidates = self.c_star_candidates[: mid_c_star_id + 1]
                self.upper_c_star = self.c_star_candidates[-1]
            else:
                self.c_star_candidates = self.c_star_candidates[mid_c_star_id:]
                self.lower_c_star = self.c_star_candidates[0]

        return self.c_star_candidates, self.is_feasible(self.c_star_candidates[-1])

    def get_cost_increases(self, assignments, increase_matrix=None):
        if increase_matrix is not None:
            return np.array([
                increase_matrix[agent_id, assignments[agent_id]]
                for agent_id in range(self.n_agents)
            ])

        increases = []
        for agent_id in range(self.n_agents):
            lowest_cost = self.cost_matrix[agent_id, :].min()

            increases.append(
                self.cost_matrix[agent_id, assignments[agent_id]]
                - lowest_cost
            )

        return np.array(increases)

    def get_cost(self, assignments, cost_matrix=None):
        if cost_matrix is None:
            return sum(
                self.cost_matrix[agent_id, assignments[agent_id]]
                for agent_id in range(self.n_agents)
            )

        return sum(
            cost_matrix[agent_id, assignments[agent_id]]
            for agent_id in range(self.n_agents)
        )
