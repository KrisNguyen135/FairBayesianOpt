import pulp
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# plt.style.use('seaborn-darkgrid')

import time


class AssignmentHelper:
    def __init__(self, types=None, name=f'{int(time.time())}', prob_df=None):
        self.name = name

        if prob_df is None:
            prob_df = pd.read_csv('../../data/data.csv', index_col=0)  #.iloc[: 2000, :]
        self.prob_df = prob_df

        self.households = list(prob_df.index)  # [i for i in range(1, self.prob_df.shape[0] + 1)]
        if types is None:
            self.types = ['ES', 'PSH', 'TH', 'RRH', 'PREV']
        else:
            self.types = types

    def update_constraints(self, fairness_constraint=None,
                           fairness_best_constraint=None, capacity_df=None):
        # Variable containing individual assignments
        self.x = pulp.LpVariable.dicts(
            'assignment',
            [(household, type_) for household in self.households
             for type_ in self.types],
            cat='Binary'
        )

        # IP problem
        self.prob = pulp.LpProblem('homelessness', pulp.LpMinimize)

        # Objective function
        self.prob += pulp.lpSum(self.x[(household, type_)] * self.prob_df.loc[household, type_]
                                for household in self.households for type_ in self.types)

        # Assignment constraint
        for household in self.households:
            self.prob += pulp.lpSum(self.x[(household, type_)] for type_ in self.types) == 1

        # Capacity constraint
        if capacity_df is None:
            capacity_df = self.prob_df['Real'].value_counts()  # original constraint
        for type_index, type_ in enumerate(self.types):
            self.prob += pulp.lpSum(
                self.x[(household, type_)] for household in self.households
            ) <= capacity_df.loc[type_index + 1]

        # Fairness constraint
        if fairness_constraint is not None:
            for household in self.households:
                self.prob += pulp.lpSum(
                    self.x[(household, type_)] * self.prob_df.loc[household, type_]
                    for type_ in self.types
                ) <= self.prob_df.loc[
                    household,
                    self.types[self.prob_df.loc[household, 'Real'] - 1]
                ] + fairness_constraint

        # Constraint on the distance from individual best
        if fairness_best_constraint is not None:
            for household in self.households:
                self.prob += pulp.lpSum(
                    self.x[(household, type_)] * self.prob_df.loc[household, type_]
                    for type_ in self.types
                ) <= self.prob_df.loc[household, self.types].min() + fairness_best_constraint

    def ip_solve(self, solver=None):
        if solver is None:
            self.prob.solve(solver=pulp.solvers.GUROBI_CMD())
        else:
            self.prob.solve(solver=solver)

        if pulp.LpStatus[self.prob.status] != 'Optimal':
            # print(f'{self.name}: solution not optimal')
            return False

        sol_df = pd.DataFrame(columns=self.types)
        for household in self.households:
            for type_ in self.types:
                sol_df.loc[household, type_] = self.x[(household, type_)].varValue

        return sol_df

    def compare_reentry(self, assignment_df, name=None, verbose=True):
        if verbose:
            print(f'Total number of reentered households: {self.prob_df["Reentered"].sum()}')

        # BART-estimated
        bart_running_result = 0
        for household in self.households:
            bart_running_result += self.prob_df.loc[
                household, self.types[self.prob_df.loc[household, 'Real'] - 1]
            ]

        if verbose:
            print(f'BART-estimated: {bart_running_result:.4f}')

        # Input assignment
        running_result = 0
        for household in self.households:
            running_result += sum(
                assignment_df.loc[household, type_] * self.prob_df.loc[household, type_]
                for type_ in self.types
            )

        if verbose:
            if name is not None:
                print(f'{name}: {running_result:.4f}')
            else:
                print(f'Input assignment: {running_result:.4f}')

        return self.prob_df['Reentered'].sum(), bart_running_result, running_result

    def compare_prob(self, assignment_df, name=None):
        if name is None:
            name = 'Input assignment'
        prob_compare_df = pd.DataFrame(columns=['Real', name])

        # Compute the DataFrame containing assigned probabilities
        for household in self.households:
            prob_compare_df.loc[household, 'Real'] = self.prob_df.loc[
                household,
                self.types[self.prob_df.loc[household, 'Real'] - 1]
            ]
            prob_compare_df.loc[household, name] = self.prob_df.loc[
                household,
                self.types[np.where(assignment_df.loc[household] == 1)[0][0]]
            ]

        # Difference between assigned and real probabilities
        prob_compare_df['Input_to_real'] = prob_compare_df[name] - prob_compare_df['Real']
        prob_compare_df['Input_to_best'] = prob_compare_df[name] - self.prob_df[self.types].min(axis=1)
        prob_compare_df['Real_to_best'] = prob_compare_df['Input_to_best'] - prob_compare_df['Input_to_real']
        # if include_zero:
        #     plt.hist(prob_compare_df['Diff'], bins=30)
        # else:
        #     plt.hist(prob_compare_df[prob_compare_df['Diff'] != 0]['Diff'], bins=30)
        # plt.show()
        return prob_compare_df


class AssignmentHelperV2:
    def __init__(self, cost_matrix, capacities):
        self.cost_matrix = cost_matrix
        self.capacities = capacities
        self.n_agents = cost_matrix.shape[0]
        self.n_intvs = cost_matrix.shape[1]

        # assert self.n_intvs == len(self.capacities)

    def ip_solve(self):
        # Variables denoting the assignments
        self.x = pulp.LpVariable.dicts(
            'assignment',
            [(agent_id, intv_id)
             for agent_id in range(self.n_agents)
             for intv_id in range(self.n_intvs)],
             cat='Binary'
        )

        # IP problem
        self.prob = pulp.LpProblem()

        # Objective function
        self.prob += pulp.lpSum(
            self.x[(agent_id, intv_id)] * self.cost_matrix[agent_id, intv_id]
            for agent_id in range(self.n_agents)
            for intv_id in range(self.n_intvs)
        )

        # Assignment constraint
        for agent_id in range(self.n_agents):
            self.prob += pulp.lpSum(
                self.x[(agent_id, intv_id)]
                for intv_id in range(self.n_intvs)
            ) == 1

        # Capacity constraint
        for intv_id in range(self.n_intvs):
            self.prob += pulp.lpSum(
                self.x[(agent_id, intv_id)]
                for agent_id in range(self.n_agents)
            ) <= self.capacities[intv_id]

        self.prob.solve(solver=pulp.solvers.GUROBI_CMD())
        if pulp.LpStatus[self.prob.status] != 'Optimal':
            return False

        # Make the assignment array
        assignments = []
        for agent_id in range(self.n_agents):
            for intv_id in range(self.n_intvs):
                if self.x[(agent_id, intv_id)].varValue == 1:
                    assignments.append(intv_id)
                    break

        return np.array(assignments)

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
