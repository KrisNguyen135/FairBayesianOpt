import pandas as pd
import numpy as np
np.random.seed(1)
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = 20, 13
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 2
import seaborn as sns

import sys
sys.path.insert(0, '../../..')
import assignment
import leximin_assignment
import matrix_gen

from tqdm import tqdm

import warnings
from datetime import datetime


def custom_eff_assignment(cost_matrix, capacities):
    full_cost_matrix = np.repeat(cost_matrix, capacities, axis=1)
    row_ind, col_ind = linear_sum_assignment(full_cost_matrix)
    
    return full_cost_matrix[row_ind, col_ind].sum()


N = 30
N_INTVS = 5
N_EXPERIMENTS = 500
CAPACITIES = np.ones((N_INTVS,), dtype=int) * (N // N_INTVS)

ALPHAS = [0.5, 0.3, 0.1, 0.08, 0.05]
BETAS = [0.5, 0.3, 0.1, 0.08, 0.05]


pof_pop = [[[] for __ in BETAS] for _ in ALPHAS]

for a_id, a in enumerate(ALPHAS):
    for b_id, b in enumerate(BETAS):
        print(a, b, datetime.now())
        
        matrix_generator = matrix_gen.MatrixGenerator(
            np.random.beta, (a, b), N, N_INTVS
        )
        
        success_count = 0
        while success_count < N_EXPERIMENTS:
            (_, sorted_matrix), (__, sorted_increase_matrix) = matrix_generator.get_increase_matrix_pair(
                round_=False
            )


            # Leximin
            lex_assigner = leximin_assignment.LeximinAssignmentHelperV3(
                sorted_increase_matrix, CAPACITIES
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lex_assignments = lex_assigner.solve()
            if lex_assignments is False:
                continue
            lex_cost = lex_assigner.get_cost(
                lex_assignments, cost_matrix=sorted_matrix
            )
            lex_cost_increases = lex_assigner.get_cost_increases(
                lex_assignments, increase_matrix=sorted_increase_matrix
            )
            
            
            # Efficient
            eff_assigner = assignment.AssignmentHelperV2(
                sorted_matrix, CAPACITIES
            )
            eff_assignments = eff_assigner.ip_solve()
            eff_cost = eff_assigner.get_cost(
                eff_assignments, cost_matrix=sorted_matrix
            )
            eff_cost_increases = eff_assigner.get_cost_increases(
                eff_assignments, increase_matrix=sorted_increase_matrix
            )


            if lex_cost_increases.max() > eff_cost_increases.max():
                print(sorted_matrix)
                print(lex_cost_increases.max(), eff_cost_increases.max())
                raise KeyboardInterrupt
            
            if eff_cost == 0:
                pof_pop[a_id][b_id].append(1)
            elif lex_cost / eff_cost > N - 1:
                print(sorted_matrix)
                print(lex_cost, eff_cost)
                raise KeyboardInterrupt
            else:
                pof_pop[a_id][b_id].append(lex_cost / eff_cost)
            success_count += 1
            
            print(a, b, success_count)
        
        print(np.mean(pof_pop[a_id][b_id]))


sns.heatmap(
    np.mean(pof_pop, axis=2),
    annot=True, fmt='.4f',
    xticklabels=BETAS, yticklabels=ALPHAS
)
plt.ylabel('alpha')
plt.xlabel('beta')

plt.show()


sns.heatmap(
    np.mean(pof_pop, axis=2),
    annot=True, fmt='.4f',
    xticklabels=BETAS, yticklabels=ALPHAS
)
plt.ylabel('alpha')
plt.xlabel('beta')

plt.show()


matrix_generator = matrix_gen.MatrixGenerator(
    np.random.beta, (0.01, 0.1), N, N_INTVS
)

while True:
    (_, sorted_matrix), (__, sorted_increase_matrix) = matrix_generator.get_increase_matrix_pair(
        round_=False
    )


    # Efficient
    eff_cost = custom_eff_assignment(sorted_matrix, CAPACITIES)

    # Leximin
    lex_assigner = leximin_assignment.LeximinAssignmentHelperV3(
        sorted_increase_matrix * 1e4, CAPACITIES
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lex_assignments = lex_assigner.solve()
    if lex_assignments is False:
        continue
    lex_cost = lex_assigner.get_cost(
        lex_assignments, cost_matrix=sorted_matrix
    )

    if lex_cost / eff_cost > N - 1:
        print(lex_cost, eff_cost)
        break


sorted_matrix


lex_assignments


lex_assigner.get_cost_increases(
    lex_assignments, increase_matrix=sorted_increase_matrix
)


lex_assigner.get_cost_increases(
    lex_assignments, increase_matrix=sorted_increase_matrix
).max()


full_cost_matrix = np.repeat(sorted_matrix, CAPACITIES, axis=1)
row_ind, col_ind = linear_sum_assignment(full_cost_matrix)

lex_assigner.get_cost_increases(
    col_ind // 5, increase_matrix=sorted_increase_matrix
)


matrix_generator = matrix_gen.MatrixGenerator(
    np.random.beta, (0.01, 0.1), N, N_INTVS
)

count = 0
while True:
    print(count)
    
    (_, sorted_matrix), (__, sorted_increase_matrix) = matrix_generator.get_increase_matrix_pair(
        round_=False
    )


    # Efficient
    eff_cost = custom_eff_assignment(sorted_matrix, CAPACITIES)

    # Leximin
    lex_assigner = leximin_assignment.LeximinAssignmentHelperV3(
        sorted_increase_matrix * 1e10, CAPACITIES
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lex_assignments = lex_assigner.solve()
    if lex_assignments is False:
        continue
    lex_cost = lex_assigner.get_cost(
        lex_assignments, cost_matrix=sorted_matrix
    )

    if lex_cost / eff_cost > N - 1:
        print(lex_cost, eff_cost)
        break
    
    count += 1


sorted_increase_matrix


lex_assigner.get_cost_increases(
    lex_assignments, increase_matrix=sorted_increase_matrix
).max()


full_cost_matrix = np.repeat(sorted_matrix, CAPACITIES, axis=1)
row_ind, col_ind = linear_sum_assignment(full_cost_matrix)

lex_assigner.get_cost_increases(
    col_ind // 5, increase_matrix=sorted_increase_matrix
)


matrix_generator = matrix_gen.MatrixGenerator(
    np.random.beta, (0.01, 0.1), N, N_INTVS
)

count = 0
while True:
    print(count)
    
    (_, sorted_matrix), (__, sorted_increase_matrix) = matrix_generator.get_increase_matrix_pair(
        round_=False
    )


    # Efficient
    # eff_cost = custom_eff_assignment(sorted_matrix, CAPACITIES)
    full_cost_matrix = np.repeat(sorted_matrix, CAPACITIES, axis=1)
    row_ind, col_ind = linear_sum_assignment(full_cost_matrix)
    eff_cost = full_cost_matrix[row_ind, col_ind].sum()

    # Leximin
    lex_assigner = leximin_assignment.LeximinAssignmentHelperV3(
        sorted_increase_matrix * 1e10, CAPACITIES
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lex_assignments = lex_assigner.solve()
    if lex_assignments is False:
        continue
    lex_cost = lex_assigner.get_cost(
        lex_assignments, cost_matrix=sorted_matrix
    )
    lex_cost_increases = lex_assigner.get_cost_increases(
        lex_assignments, increase_matrix=sorted_increase_matrix
    )
    
    eff_cost_increases = lex_assigner.get_cost_increases(
        col_ind // (N // N_INTVS), increase_matrix=sorted_increase_matrix
    )

    if lex_cost_increases.max() > eff_cost_increases.max():
        break
    
    count += 1


lex_cost_increases.max()


eff_cost_increases


matrix_generator = matrix_gen.MatrixGenerator(
    np.random.beta, (0.03, 0.3), N, N_INTVS
)

count = 0
while True:
    print(count)
    
    (_, sorted_matrix), (__, sorted_increase_matrix) = matrix_generator.get_increase_matrix_pair(
        round_=False
    )


    # Efficient
    # eff_cost = custom_eff_assignment(sorted_matrix, CAPACITIES)
    full_cost_matrix = np.repeat(sorted_matrix, CAPACITIES, axis=1)
    row_ind, col_ind = linear_sum_assignment(full_cost_matrix)
    eff_cost = full_cost_matrix[row_ind, col_ind].sum()

    # Leximin
    lex_assigner = leximin_assignment.LeximinAssignmentHelperV3(
        sorted_increase_matrix * 1e10, CAPACITIES
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lex_assignments = lex_assigner.solve()
    if lex_assignments is False:
        continue
    lex_cost = lex_assigner.get_cost(
        lex_assignments, cost_matrix=sorted_matrix
    )
    lex_cost_increases = lex_assigner.get_cost_increases(
        lex_assignments, increase_matrix=sorted_increase_matrix
    )
    
    eff_cost_increases = lex_assigner.get_cost_increases(
        col_ind // (N // N_INTVS), increase_matrix=sorted_increase_matrix
    )

    if lex_cost_increases.max() > eff_cost_increases.max():
        break
    
    count += 1



