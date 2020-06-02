import pandas as pd
import numpy as np
np.random.seed(0)
from scipy.stats import beta as sp_beta

import matplotlib.pyplot as plt
# plt.style.use('seaborn-darkgrid')
plt.rcParams['figure.figsize'] = 20, 13
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 2
import seaborn as sns
import tikzplotlib

import sys
sys.path.insert(0, '../../..')
import assignment
import leximin_assignment
import matrix_gen

from tqdm import tqdm

import warnings


N = 30
N_INTVS = 5
N_EXPERIMENTS = 1000
CAPACITIES = np.ones((N_INTVS,), dtype=int) * (N // N_INTVS)

ALPHAS = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]
BETAS = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8]


pof_pop = [[[] for __ in BETAS] for _ in ALPHAS]

for a_id, a in enumerate(ALPHAS):
    for b_id, b in enumerate(BETAS):
        print(a, b)
        
        matrix_generator = matrix_gen.MatrixGenerator(
            np.random.beta, (a, b), N, N_INTVS
        )
        
        success_count = 0
        while success_count < N_EXPERIMENTS:
            # print(a, b, success_count)
            
            (_, sorted_matrix), __ = matrix_generator.get_increase_matrix_pair(
                round_=False
            )

            _, counts = np.unique(sorted_matrix, return_counts=True)
            if np.any(counts > 1):
                continue


            # Efficient
            eff_assigner = assignment.AssignmentHelperV2(
                sorted_matrix, CAPACITIES
            )
            eff_assignments = eff_assigner.ip_solve()
            eff_cost = eff_assigner.get_cost(
                eff_assignments, cost_matrix=sorted_matrix
            )

            # Leximin
            lex_assigner = leximin_assignment.RowSortedLeximinAssignmentHelper(
                sorted_matrix, CAPACITIES
            )
            lex_assignments = lex_assigner.solve()
            lex_cost = eff_assigner.get_cost(
                lex_assignments, cost_matrix=sorted_matrix
            )


            pof_pop[a_id][b_id].append(lex_cost / eff_cost)
            success_count += 1


sns.heatmap(
    np.mean(pof_pop, axis=2),
    annot=True, fmt='.4f',
    xticklabels=BETAS, yticklabels=ALPHAS
)
plt.ylabel('alpha')
plt.xlabel('beta')

plt.show()


sns.heatmap(
    np.max(pof_pop, axis=2),
    annot=True, fmt='.4f',
    xticklabels=BETAS, yticklabels=ALPHAS
)
plt.ylabel('alpha')
plt.xlabel('beta')

plt.show()


cost_ratio_pop = [[[] for __ in BETAS] for _ in ALPHAS]

for a_id, a in enumerate(ALPHAS):
    print(a)
    for b_id, b in enumerate(BETAS):
        # print(a, b)
        
        matrix_generator = matrix_gen.MatrixGenerator(
            np.random.beta, (a, b), N, N_INTVS
        )
        
        success_count = 0
        while success_count < N_EXPERIMENTS:
            # print(a, b, success_count)
            
            (_, sorted_matrix), (__, sorted_increase_matrix) = matrix_generator.get_increase_matrix_pair(
                round_=False
            )

            # _, counts = np.unique(sorted_matrix, return_counts=True)
            # if np.any(counts > 1):
            #     continue


            # Norm leximin
            lex_n_assigner = leximin_assignment.RowSortedLeximinAssignmentHelper(
                sorted_increase_matrix, CAPACITIES
            )
            lex_n_assignments = lex_n_assigner.solve()
            lex_n_cost = lex_n_assigner.get_cost(
                lex_n_assignments, cost_matrix=sorted_matrix
            )
            
            # Un-norm Leximin
            lex_u_assigner = leximin_assignment.RowSortedLeximinAssignmentHelper(
                sorted_matrix, CAPACITIES
            )
            lex_u_assignments = lex_u_assigner.solve()
            lex_u_cost = lex_u_assigner.get_cost(
                lex_u_assignments, cost_matrix=sorted_matrix
            )

            if lex_n_cost == 0:
                cost_ratio_pop[a_id][b_id].append(1)
            else:
                cost_ratio_pop[a_id][b_id].append(lex_u_cost / lex_n_cost)
            success_count += 1


sns.heatmap(
    np.mean(cost_ratio_pop, axis=2),
    annot=True, fmt='.4f',
    xticklabels=BETAS, yticklabels=ALPHAS
)
plt.ylabel('alpha')
plt.xlabel('beta')

plt.show()


from scipy.optimize import linear_sum_assignment

def custom_eff_assignment(cost_matrix, capacities):
    full_cost_matrix = np.repeat(cost_matrix, capacities, axis=1)
    row_ind, col_ind = linear_sum_assignment(full_cost_matrix)
    
    return (
        col_ind // (cost_matrix.shape[0] // cost_matrix.shape[1]),
        full_cost_matrix[row_ind, col_ind].sum()
    )


lu_cost_pop = [[[] for __ in BETAS] for _ in ALPHAS]
ln_cost_pop = [[[] for __ in BETAS] for _ in ALPHAS]
eff_cost_pop = [[[] for __ in BETAS] for _ in ALPHAS]

for a_id, a in enumerate(ALPHAS):
    # print(a)
    for b_id, b in enumerate(BETAS):
        print(a, b)
        
        matrix_generator = matrix_gen.MatrixGenerator(
            np.random.beta, (a, b), N, N_INTVS
        )
        
        success_count = 0
        while success_count < N_EXPERIMENTS:
            # print(a, b, success_count)
            
            (_, sorted_matrix), (__, sorted_increase_matrix) = matrix_generator.get_increase_matrix_pair(
                round_=False
            )

            # _, counts = np.unique(sorted_matrix, return_counts=True)
            # if np.any(counts > 1):
            #     continue


            # Norm leximin
            lex_n_assigner = leximin_assignment.RowSortedLeximinAssignmentHelper(
                sorted_increase_matrix, CAPACITIES
            )
            lex_n_assignments = lex_n_assigner.solve()
            lex_n_cost = lex_n_assigner.get_cost(
                lex_n_assignments, cost_matrix=sorted_matrix
            )
            
            # Un-norm Leximin
            lex_u_assigner = leximin_assignment.RowSortedLeximinAssignmentHelper(
                sorted_matrix, CAPACITIES
            )
            lex_u_assignments = lex_u_assigner.solve()
            lex_u_cost = lex_u_assigner.get_cost(
                lex_u_assignments, cost_matrix=sorted_matrix
            )
            
            # Efficient
            eff_assignments, eff_cost = custom_eff_assignment(sorted_matrix, CAPACITIES)
            
            if eff_cost == 0:
                lu_cost_pop[a_id][b_id].append(0)
                ln_cost_pop[a_id][b_id].append(0)
                eff_cost_pop[a_id][b_id].append(0)
            else:
                lu_cost_pop[a_id][b_id].append(lex_u_cost)
                ln_cost_pop[a_id][b_id].append(lex_n_cost)
                eff_cost_pop[a_id][b_id].append(eff_cost)

            success_count += 1


lu_costs = np.array(lu_cost_pop)
ln_costs = np.array(ln_cost_pop)
eff_costs = np.array(eff_cost_pop)

lu_costs.shape, ln_costs.shape, eff_costs.shape


np.save('lu_costs', lu_costs)
np.save('ln_costs', ln_costs)
np.save('eff_costs', eff_costs)


lu_costs = np.load('lu_costs.npy')
ln_costs = np.load('ln_costs.npy')
eff_costs = np.load('eff_costs.npy')

lu_costs.shape, ln_costs.shape, eff_costs.shape


lu_pofs = lu_costs / eff_costs
ln_pofs = ln_costs / eff_costs
r = lu_costs / ln_costs


sns.heatmap(
    np.mean(lu_pofs, axis=2),
    annot=True, fmt='.4f',
    xticklabels=BETAS, yticklabels=ALPHAS,
    cmap='Blues_r'
)
plt.ylabel('alpha')
plt.xlabel('beta')

plt.show()


sns.heatmap(
    np.mean(ln_pofs, axis=2),
    annot=True, fmt='.4f',
    xticklabels=BETAS, yticklabels=ALPHAS,
    cmap='Blues_r'
)
plt.ylabel('alpha')
plt.xlabel('beta')

plt.show()


sns.heatmap(
    np.mean(r, axis=2),
    annot=True, fmt='.4f',
    xticklabels=BETAS, yticklabels=ALPHAS,
    cmap='Blues_r'
)
plt.ylabel('alpha')
plt.xlabel('beta')

plt.show()


ax = sns.heatmap(
    np.mean(lu_pofs, axis=2),
    xticklabels=BETAS, yticklabels=ALPHAS,
    cmap='Reds', vmin=1
)
ax.invert_yaxis()
plt.ylabel('alpha')
plt.xlabel('beta')

# plt.show()

tikzplotlib.save('pof_lu_heatmap.tex')


ax = sns.heatmap(
    np.mean(ln_pofs, axis=2),
    xticklabels=BETAS, yticklabels=ALPHAS,
    cmap='Reds', vmin=1
)
ax.invert_yaxis()
plt.ylabel('alpha')
plt.xlabel('beta')

tikzplotlib.save('pof_ln_heatmap.tex')


sns.heatmap(
    np.mean(r, axis=2),
    xticklabels=BETAS, yticklabels=ALPHAS,
    cmap='Blues_r'
)
plt.ylabel('alpha')
plt.xlabel('beta')

plt.show()


avg_lu_pofs = np.mean(lu_pofs, axis=2)
avg_ln_pofs = np.mean(ln_pofs, axis=2)
max_avg_pof = np.max([avg_lu_pofs, avg_ln_pofs])

fig, ax = plt.subplots(
    1, 2, figsize=(20, 7.5), sharex=True, sharey=True,
    gridspec_kw=dict(width_ratios=[4.4, 5.6])
)

sns.heatmap(
    avg_lu_pofs,
    xticklabels=BETAS, yticklabels=ALPHAS,
    label='big', cmap='Blues_r', ax=ax[0],
    cbar=False, vmin=1, vmax=max_avg_pof
)

cs = sns.heatmap(
    avg_ln_pofs,
    xticklabels=BETAS, yticklabels=ALPHAS,
    label='big', cmap='Blues_r', ax=ax[1],
    cbar=True, vmin=1, vmax=max_avg_pof
)

ax[0].set_ylabel('Alpha')
ax[0].set_xlabel('Beta')
ax[1].set_xlabel('Beta');


avg_lu_pofs = np.mean(lu_pofs, axis=2)
avg_ln_pofs = np.mean(ln_pofs, axis=2)
max_avg_pof = np.max([avg_lu_pofs, avg_ln_pofs])

fig, ax = plt.subplots(
    1, 2, figsize=(20, 7.5), sharex=True, sharey=True,
    gridspec_kw=dict(width_ratios=[4.4, 5.6])
)

sns.heatmap(
    avg_lu_pofs,
    xticklabels=BETAS, yticklabels=ALPHAS,
    label='big', ax=ax[0],
    cbar=False, vmin=1, vmax=max_avg_pof
)

cs = sns.heatmap(
    avg_ln_pofs,
    xticklabels=BETAS, yticklabels=ALPHAS,
    label='big', ax=ax[1],
    cbar=True, vmin=1, vmax=max_avg_pof
)

ax[0].set_ylabel('Alpha')
ax[0].set_xlabel('Beta')
ax[1].set_xlabel('Beta');


sns.heatmap(
    np.mean(lu_pofs, axis=2),
    xticklabels=BETAS, yticklabels=ALPHAS,
    cmap='Reds', vmin=1
)
plt.ylabel('alpha')
plt.xlabel('beta')

tikzplotlib.save('pof_lu_heatmap_new.tex', externalize_tables=True)


# lu_costs
# ln_costs
eff_costs.shape


np.percentile(lu_costs[:, 0, :], [0.25, 0.75], axis=1)


pcnts = np.percentile(lu_costs[:, 0, :], [0.95, 0.05], axis=1)
means = np.mean(lu_costs[:, 0, :], axis=1)
errors = pcnts - means
errors[0] *= -1

plt.errorbar(
    ALPHAS, means, yerr=errors,
    capsize=5, capthick=3, marker='^', markersize=10
);


plt.plot(
    ALPHAS, np.mean(lu_costs[:, 0, :], axis=1),
    label=r'$C(L_u)$', marker='^'
)
plt.plot(
    ALPHAS, np.mean(eff_costs[:, 0, :], axis=1),
    label=r'$C(E)$', marker='^'
)

plt.xscale('log')
plt.xticks(ALPHAS, labels=ALPHAS)

plt.legend();


fig, ax = plt.subplots(2, 2, figsize=(20, 16))

for b_id, b in enumerate(BETAS[: 4]):
    ax[b_id // 2][b_id % 2].plot(
        ALPHAS, np.mean(lu_costs[:, b_id, :], axis=1),
        label=r'$C(L_u)$', marker='o'
    )
    ax[b_id // 2][b_id % 2].plot(
        ALPHAS, np.mean(eff_costs[:, b_id, :], axis=1),
        label=r'$C(E)$', marker='^'
    )
    
    ax[b_id // 2][b_id % 2].set_xscale('log')
    ax[b_id // 2][b_id % 2].set_xticks(ALPHAS)
    ax[b_id // 2][b_id % 2].set_xticklabels(ALPHAS)
    
    ax[b_id // 2][b_id % 2].set_title(r'$\beta = $' + str(b))
    ax[b_id // 2][b_id % 2].legend()
    
tikzplotlib.save('avg_cost_growth_alpha.tex')



