import pulp
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

import time


class AssignmentHelper:
    def __init__(self):
        self.prob_df = pd.read_csv('../../data/data.csv', index_col=0)  #.iloc[: 1000, :]

        self.households = [i for i in range(1, self.prob_df.shape[0] + 1)]
        self.types = ['ES', 'PSH', 'TH', 'RRH', 'PREV']

    def update_constraints(self, fairness_constraint=None, capacity_df=None):
        # Variable containing individual assignments
        print('Creating variables')
        self.x = pulp.LpVariable.dicts(
            'assignment',
            [(household, type_) for household in self.households
             for type_ in self.types],
            cat='Binary'
        )

        # IP problem
        print('Setting up problem')
        self.prob = pulp.LpProblem('homelessness', pulp.LpMinimize)

        # Objective function
        print('Setting up objective function')
        self.prob += sum(self.x[(household, type_)] * self.prob_df.loc[household, type_]
                         for household in self.households for type_ in self.types)

        # Assignment constraint
        print('Creating assignment constraint')
        for household in self.households:
            self.prob += sum(self.x[(household, type_)] for type_ in self.types) == 1

        # Capacity constraint
        print('Creating capacity constraint')
        if capacity_df is None:
            capacity_df = self.prob_df['Real'].value_counts()  # original constraint
        for type_index, type_ in enumerate(self.types):
            self.prob += sum(
                self.x[(household, type_)] for household in self.households
            ) <= capacity_df.loc[type_index + 1]

        # Fairness constraint
        if fairness_constraint is not None:
            print('Creating fairness constraint')
            for household in self.households:
                self.prob += sum(
                    self.x[(household, type_)] * self.prob_df.loc[household, type_]
                    for type_ in self.types
                ) <= self.prob_df.loc[
                    household,
                    self.types[self.prob_df.loc[household, 'Real'] - 1]
                ] + fairness_constraint

    def ip_solve(self, solver=None):
        if solver is None:
            self.prob.solve(solver=pulp.solvers.GUROBI_CMD())
        else:
            self.prob.solve(solver=solver)

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

    def visualize_increased_prob(self, assignment_df, name=None, include_zero=False):
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
        prob_compare_df['Diff'] = prob_compare_df[name] - prob_compare_df['Real']
        if include_zero:
            plt.hist(prob_compare_df['Diff'], bins=30)
        else:
            plt.hist(prob_compare_df[prob_compare_df['Diff'] != 0]['Diff'], bins=30)
        plt.show()
