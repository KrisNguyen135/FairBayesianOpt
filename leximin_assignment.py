import pulp
import pandas as pd
import numpy as np


class LeximinAssignmentHelper:
    def __init__(self, cost_matrix, capacities):
        self.cost_matrix = cost_matrix
        self.n_agents = cost_matrix.shape[0]
        self.n_intvs = cost_matrix.shape[1]
        self.capacities = capacities

        self.c_star_candidates = np.unique(self.cost_matrix)
        # self.lower_c_star = self.c_star_candidates[0]
        # self.upper_c_star = self.c_star_candidates[-1]

    def is_feasible(self, c_star, leximin_counts):
        """
        `leximin_counts` is a dictionary of how many variables need to hold a
        common leximin value so that an assignment is feasible.
        `valid_matrix` denotes whether the cost of a cell is larger than the
        first leximin.
        """

        # temp_matrix = self.cost_matrix.copy()

        # Don't allow for the values larger than c* to be in the assignment
        # except for the leximin values (constrained by their resp. counts)
        valid_matrix = np.ones(self.cost_matrix.shape)
        valid_matrix[self.cost_matrix > c_star] = 0
        for leximin in leximin_counts:
            valid_matrix[self.cost_matrix == leximin] = 1
        # print(leximin_counts)
        # print(c_star)
        # print(valid_matrix)

        x = pulp.LpVariable.dicts(
            'assignment',
            [(agent_id, intv_id)
            for agent_id in range(self.n_agents)
            for intv_id in range(self.n_intvs)
            if valid_matrix[agent_id, intv_id]],
            cat='Binary'
        )

        # Objective - just for feasibility
        prob = pulp.LpProblem()
        prob += pulp.lpSum(
            x[key] * self.cost_matrix[key[0], key[1]]
            for key in x
        )

        # Assignment constraints
        for agent_id in range(self.n_agents):
            prob += pulp.lpSum(
                x[(agent_id, intv_id)]
                for intv_id in range(self.n_intvs)
                if valid_matrix[agent_id, intv_id]
            ) == 1

        # Capacity constraints
        for intv_id in range(self.n_intvs):
            prob += pulp.lpSum(
                x[(agent_id, intv_id)]
                for agent_id in range(self.n_agents)
                if valid_matrix[agent_id, intv_id]
            ) <= self.capacities[intv_id]

        # Leximin count constraints
        for leximin in leximin_counts:
            if leximin != c_star:
                prob += pulp.lpSum(
                    x[(agent_id, intv_id)]
                    for agent_id, intv_id in np.argwhere(self.cost_matrix == leximin)
                ) <= leximin_counts[leximin]

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

    def is_feasible_v2(self, c_star, non_unique_leximin_counts, running_capacities):
        """
        `non_unique_leximin_counts` is a dictionary of how many (more than one)
        variables need to hold a common leximin value so that an assignment is
        feasible.
        `running_capacities` is `self.capacities` with subtracted values
        corresponding to variables that have a unique leximin.
        """

        # Don't allow for the values larger than c* to be in the assignment
        # except for the leximin values that occur more than once (constrained
        # by their resp. counts)
        valid_matrix = np.ones(self.cost_matrix.shape)
        valid_matrix[self.cost_matrix > c_star] = 0
        for leximin in non_unique_leximin_counts:
            valid_matrix[self.cost_matrix == leximin] = 1

        x = pulp.LpVariable.dicts(
            'assignment',
            [(agent_id, intv_id)
            for agent_id in range(self.n_agents)
            for intv_id in range(self.n_intvs)
            if valid_matrix[agent_id, intv_id]],
            cat='Binary'
        )

        # Feasibility objective
        prob = pulp.LpProblem()
        prob += pulp.lpSum(
            x[key] * self.cost_matrix[key[0], key[1]]
            fpr key in x
        )

        # Assignment constraints
        for agent_id in range(self.n_agents):
            prob += pulp.lpSum(
                x[(agent_id, intv_id)]
                for intv_id in range(self.n_intvs)
                if valid_matrix[agent_id, intv_id]
            ) == 1

        # Capacity constraints
        for intv_id in range(self.n_intvs):
            prob += pulp.lpSum(
                x[(agent_id, intv_id)]
                for agent_id in range(self.n_agents)
                if valid_matrix[agent_id, intv_id]
            ) <= running_capacities[intv_id]

        # Leximin count constraints
        for leximin in leximin_counts:
            if leximin != c_star:
                prob += pulp.lpSum(
                    x[(agent_id, intv_id)]
                    for agent_id, intv_id in np.argwhere(self.cost_matrix == leximin)
                ) <= leximin_counts[leximin]

        prob.solve(solver=pulp.solvers.GUROBI_CMD())

        if pulp.LpStatus[prob.status] == 'Optimal':
            assignments = np.zeros((self.n_agents,))

            for agent_id in range(self.n_agents):
                for intv_id in range(self.n_intvs):
                    if valid_matrix[agent_id, intv_id] \
                            and x[(agent_id, intv_id)].varValue == 1:
                        assignments[agent_id] = intv_id
                        break

        return assignments

    def solve(self, verbose=False):
        leximin_counts = {}
        agent_count = 0

        # Binary-search for the next leximin between 0 and the current smallest
        # leximin.
        def recur_solve(upper_c_star_id):
            lower_c_star_id = 0

            # Return right away if the lowest c* is feasible or if the highest
            # c* is not.
            potential_assignments = self.is_feasible(
                self.c_star_candidates[0], leximin_counts)
            if potential_assignments is not False:
                return 0, potential_assignments

            if self.is_feasible(
                self.c_star_candidates[upper_c_star_id], leximin_counts
            ) is False:
                return False

            while upper_c_star_id > lower_c_star_id + 1:
                if verbose:
                    print('Searching between:')
                    print(lower_c_star_id, self.c_star_candidates[lower_c_star_id])
                    print(upper_c_star_id, self.c_star_candidates[upper_c_star_id])
                    print('Number of agents assigned:', agent_count)
                    print()

                mid_c_star_id = (lower_c_star_id + upper_c_star_id) // 2

                if self.is_feasible(
                    self.c_star_candidates[mid_c_star_id], leximin_counts
                ) is not False:
                    upper_c_star_id = mid_c_star_id
                else:
                    lower_c_star_id = mid_c_star_id

            # print('Returning leximin id', upper_c_star_id)
            return upper_c_star_id

        # Search for the next leximin sequentially from the current smallest
        # leximin.
        def recur_solve_v2(upper_c_star_id):
            next_c_star_id = upper_c_star_id
            # print(leximin_counts)

            potential_assignments = self.is_feasible(
                self.c_star_candidates[next_c_star_id], leximin_counts)

            while potential_assignments is not False and next_c_star_id >= 0:
                next_c_star_id -= 1

                if verbose:
                    print('Considering')
                    print(next_c_star_id, self.c_star_candidates[next_c_star_id])
                    # print('Number of agents assigned:', agent_count)
                    # print()

                potential_assignments = self.is_feasible(
                    self.c_star_candidates[next_c_star_id], leximin_counts)

            # print('Returning', self.c_star_candidates[next_c_star_id + 1])
            return next_c_star_id + 1

        upper_c_star_id = len(self.c_star_candidates) - 1

        ### for `recur_solve()`
        # while agent_count < self.n_agents:
        #     recur_result = recur_solve(upper_c_star_id)
        #     if isinstance(recur_result, tuple):
        #         return recur_result[1]
        #     elif recur_result is not False:
        #         upper_c_star_id = recur_result
        #
        #     # print(self.c_star_candidates)
        #     next_leximin = self.c_star_candidates[upper_c_star_id]
        #     # print(upper_c_star_id, next_leximin, leximin_counts)
        #     if next_leximin not in leximin_counts:
        #         leximin_counts[next_leximin] = 1
        #     else:
        #         leximin_counts[next_leximin] += 1
        #     agent_count += 1

        ### for `recur_solve_v2()`
        upper_c_star_id = recur_solve(upper_c_star_id)
        agent_count = 1

        while agent_count < self.n_agents:
            if verbose:
                print('Number of agents assigned:', agent_count, '/', self.n_agents)

            next_leximin = self.c_star_candidates[upper_c_star_id]
            if next_leximin not in leximin_counts:
                leximin_counts[next_leximin] = 1
            else:
                leximin_counts[next_leximin] += 1

            upper_c_star_id = recur_solve_v2(upper_c_star_id)
            agent_count += 1

        return self.is_feasible(
            self.c_star_candidates[upper_c_star_id], leximin_counts)

    def solve_v2(self, verbose=False):
        non_unique_leximin_counts = {}
        running_capacities = self.capacities.copy()
        unique_leximin_assignments = np.zeros((self.n_agents,))
        agent_count = 0

        def recur_solve(upper_c_star_id):
            lower_c_star_id = 0

            potential_assignments = self.is_feasible_v2(
                self.c_star_candidates[0],
                non_unique_leximin_counts,
                running_capacities
            )
            if potential_assignments is not False:
                return 0, potential_assignments

            if self.is_feasible_v2(
                self.c_star_candidates[upper_c_star_id],
                non_unique_leximin_counts,
                running_capacities
            ) is False:
                return False

            while upper_c_star_id > lower_c_star_id + 1:
                if verbose:
                    print('Searching between:')
                    pritn(lower_c_star_id, self.c_star_candidates[lower_c_star_id])
                    print(upper_c_star_id, self.c_star_candidates[upper_c_star_id])

                mid_c_star_id = (lower_c_star_id + upper_c_star_id) // 2

                potential_assignments = self.is_feasible_v2(
                    self.c_star_candidates[mid_c_star_id],
                    non_unique_leximin_counts,
                    running_capacities
                )
                if potential_assignments is not False:
                    upper_c_star_id = mid_c_star_id
                else:
                    lower_c_star_id = mid_c_star_id

            if potential_assignments is not False:
                costs = self.get_costs(potential_assignments)
                leximin_agent_id = np.argwhere(
                    costs == self.c_star_candidates[upper_c_star_id]
                )[0][0]

                return upper_c_star_id,

            return upper_c_star_id

        upper_c_star_id = len(self.c_star_candidates) - 1
        next_leximin = self.c_star_candidates[-1]

        while agent_count < self.n_agents:
            old_leximin = next_leximin

            recur_result = recur_solve(upper_c_star_id)
            # TODO: the tuple check has to be changed
            if isinstance(recur_result, tuple):
                return recur_result[1]
            elif recur_result is not False:
                upper_c_star_id = recur_result

            next_leximin = self.c_star_candidates[upper_c_star_id]
            # If the next leximin is equal to the current, increment its count
            # in the count dictionary.
            if next_leximin == old_leximin:
                non_unique_leximin_counts[next_leximin] += 1
            # Else:
            else:
                non_unique_leximin_counts[next_leximin] = 1

                if non_unique_leximin_counts[old_leximin] == 1:
                    del non_unique_leximin_counts[old_leximin]

                    agent_id, intv_id = np.argwhere(self.cost_matrix == leximin)





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

    def get_costs(self, assignments, cost_matrix=None):
        if cost_matrix is None:
            return np.array([
                self.cost_matrix[agent_id, assignments[agent_id]]
                for agent_id in range(self.n_agents)
            ])

        return np.array([
            cost_matrix[agent_id, assignments[agent_id]]
            for agent_id in range(self.n_agents)
        ])

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
