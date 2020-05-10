import pandas as pd
import numpy as np
np.random.seed(1)
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# plt.style.use('seaborn-darkgrid')
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = 20, 13
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 2
import seaborn as sns

import sys
sys.path.insert(0, '../../..')
import assignment
import leximin_assignment
import matrix_gen
import kde_utils

from tqdm import tqdm

import warnings


def custom_eff_assignment(cost_matrix, capacities):
    full_cost_matrix = np.repeat(cost_matrix, capacities, axis=1)
    row_ind, col_ind = linear_sum_assignment(full_cost_matrix)
    
    return (
        col_ind // (cost_matrix.shape[0] // cost_matrix.shape[1]),
        full_cost_matrix[row_ind, col_ind].sum()
    )


df = pd.read_csv('../../../data/data.csv', index_col=0)
types = ['ES', 'TH', 'RRH', 'PREV']
df = df[df['Real'] get_ipython().getoutput("= 2]")
df = df.drop(labels='PSH', axis=1)

data = df[types].to_numpy().flatten()
kde = kde_utils.KDEHelper(np.log(data) - np.log(1 - data))


NS = [30, 40, 50, 60, 70, 80, 90, 100]
N_COLS = 5
N_EXPERIMENTS = 500

distrs = [
    np.random.uniform,
    np.random.beta,
    kde.sample_and_transform,
]
params = [(0, 1), (0.5, 0.5), (lambda x: 1 / (1 + np.exp(-x)), None)]
labels = ['U[0, 1]', 'Beta(0.5, 0.5)', 'KDE']


r_u = [[[] for _ in NS] for __ in distrs]
r_n = [[[] for _ in NS] for __ in distrs]

for i in range(len(labels)):
    for n_i, n in enumerate(NS):
        print(labels[i], n)
        
        capacities = np.ones(N_COLS, dtype=int) * (n // N_COLS)
        matrix_generator = matrix_gen.MatrixGenerator(
            distrs[i], params[i], n, N_COLS
        )
        
        for _ in range(N_EXPERIMENTS):
            (_, sorted_matrix), (__, sorted_increase_matrix) = matrix_generator.get_increase_matrix_pair(
                round_=False
            )
            
            
            # L_u
            lex_u_assigner = leximin_assignment.RowSortedLeximinAssignmentHelper(
                sorted_matrix, capacities
            )
            lex_u_assignments = lex_u_assigner.solve()
            
            b_u = lex_u_assigner.get_cost_increases(
                lex_u_assignments, increase_matrix=sorted_matrix
            ).max()
            l_u = sorted_matrix[:, -1].min()
            
            r_u[i][n_i].append(b_u / l_u)
            
            # L_n
            lex_n_assigner = leximin_assignment.RowSortedLeximinAssignmentHelper(
                sorted_increase_matrix, capacities
            )
            lex_n_assignments = lex_n_assigner.solve()
            
            b_n = lex_n_assigner.get_cost_increases(
                lex_n_assignments, increase_matrix=sorted_increase_matrix
            ).max()
            # l_n = sorted_increase_matrix[:, -1].min()
            
            r_n[i][n_i].append(b_n / l_u)


fig, ax = plt.subplots(1, len(labels), figsize=(20, 5), sharey=True)

for i in range(len(labels)):
    r_percnts = np.percentile(np.array(r_u[i]), [25, 75], axis=1)
    errors = r_percnts - np.array(r_u[i]).mean(axis=1)
    errors[0] *= -1
    errors = 1 + (N_COLS - 1) * errors
    ax[i].errorbar(
        NS, 1 + (N_COLS - 1) * np.array(r_u[i]).mean(axis=1),
        yerr=errors, label=r'$L_u$',
        capsize=5, capthick=3, marker='o', markersize=10
    )
    
    r_percnts = np.percentile(np.array(r_n[i]), [25, 75], axis=1)
    errors = r_percnts - np.array(r_n[i]).mean(axis=1)
    errors[0] *= -1
    errors = 1 + (N_COLS - 1) * errors
    ax[i].errorbar(
        NS, 1 + (N_COLS - 2) * np.array(r_n[i]).mean(axis=1),
        yerr=errors, label=r'$L_n$',
        capsize=5, capthick=3, marker='^', markersize=10
    )
    
    ax[i].set_title('\n\n' + '$' + labels[i] + r'$' + '\n')

ax[-1].legend()
ax[0].set_ylabel('Price of fairness bound\n')
ax[1].set_xlabel('\nNumber of agents')
    
plt.show()


pof_u = [[[] for _ in NS] for __ in distrs]
pof_n = [[[] for _ in NS] for __ in distrs]

pof_sorted_u = [[[] for _ in NS] for __ in distrs]
pof_sorted_n = [[[] for _ in NS] for __ in distrs]

for i in range(len(labels)):
    for n_i, n in enumerate(NS):
        print(labels[i], n)
        
        capacities = np.ones(N_COLS, dtype=int) * (n // N_COLS)
        matrix_generator = matrix_gen.MatrixGenerator(
            distrs[i], params[i], n, N_COLS
        )
        
        for _ in range(N_EXPERIMENTS):
            (
                (matrix, sorted_matrix),
                (increase_matrix, sorted_increase_matrix)
            ) = matrix_generator.get_increase_matrix_pair(round_=False)
            
            
            ### Original matrix
            # L_n
            lex_n_assigner = leximin_assignment.LeximinAssignmentHelperV3(
                increase_matrix * 1000, capacities
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lex_n_assignments = lex_n_assigner.solve(verbose=False)
            if lex_n_assignments is False:
                continue
            lex_n_cost = lex_n_assigner.get_cost(
                lex_n_assignments, cost_matrix=matrix
            )
            
            # L_u
            lex_u_assigner = leximin_assignment.LeximinAssignmentHelperV3(
                matrix * 1000, capacities
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lex_u_assignments = lex_u_assigner.solve(verbose=False)
            if lex_u_assignments is False:
                continue
            lex_u_cost = lex_u_assigner.get_cost(
                lex_u_assignments, cost_matrix=matrix
            )
            
            # Efficient
            eff_assignments, eff_cost = custom_eff_assignment(matrix, capacities)
            
            pof_u[i][n_i].append(lex_u_cost / eff_cost)
            pof_n[i][n_i].append(lex_n_cost / eff_cost)
            
            
            ### Increase (normalized) matrix
            # Efficient
            eff_assignments, eff_cost = custom_eff_assignment(sorted_matrix, capacities)
            
            # L_u
            lex_u_assigner = leximin_assignment.RowSortedLeximinAssignmentHelper(
                sorted_matrix, capacities
            )
            lex_u_assignments = lex_u_assigner.solve()
            lex_u_cost = lex_u_assigner.get_cost(
                lex_u_assignments, cost_matrix=sorted_matrix
            )
            
            pof_sorted_u[i][n_i].append(lex_u_cost / eff_cost)
            
            # L_n
            lex_n_assigner = leximin_assignment.RowSortedLeximinAssignmentHelper(
                sorted_increase_matrix, capacities
            )
            lex_n_assignments = lex_n_assigner.solve()
            lex_n_cost = lex_n_assigner.get_cost(
                lex_n_assignments, cost_matrix=sorted_matrix
            )
            
            pof_sorted_n[i][n_i].append(lex_n_cost / eff_cost)


fig, ax = plt.subplots(2, len(labels), figsize=(20, 10), sharey=True)

for i in range(len(labels)):
    # PoF(L_u)
    errors = np.percentile(np.array(pof_u[i]), [25, 75], axis=1)
    ax[0][i].errorbar(
        NS, np.array(pof_u[i]).mean(axis=1),
        yerr=errors, label=r'Empirical PoF',
        capsize=5, capthick=3
    )
    
    errors = np.percentile(np.array(pof_sorted_u[i]), [25, 75], axis=1)
    ax[0][i].errorbar(
        NS, np.array(pof_sorted_u[i]).mean(axis=1),
        yerr=errors, label=r'Empirical PoF with sorted matrices',
        capsize=5, capthick=3
    )
    
    # PoF(L_n)
    errors = np.percentile(np.array(pof_n[i]), [25, 75], axis=1)
    ax[1][i].errorbar(
        NS, np.array(pof_n[i]).mean(axis=1),
        yerr=errors, label=r'Empirical PoF',
        capsize=5, capthick=3
    )
    
    errors = np.percentile(np.array(pof_sorted_n[i]), [25, 75], axis=1)
    ax[1][i].errorbar(
        NS, np.array(pof_sorted_n[i]).mean(axis=1),
        yerr=errors, label=r'Empirical PoF with sorted matrices',
        capsize=5, capthick=3
    )
    
    ax[1][i].set_xlabel(labels[i])

ax[0][0].set_ylabel(r'$L_u$')
# ax[0][0].legend()
ax[1][0].set_ylabel(r'$L_n$')
# ax[1][0].legend()
    
plt.show()


pof_lu_df


percnts


means


fig, ax = plt.subplots(1, len(labels), figsize=(20, 5), sharey=True)

for i in range(len(labels)):
    # PoF(L_u)
    grouped_pofs = pof_lu_df[~pof_lu_df['Sorted rows']][pof_lu_df['Distr'] == labels[i]].groupby('N')['PoF']
    percnts = grouped_pofs.quantile(q=[0.25, 0.75]).to_numpy().reshape(len(NS), 2).T
    means = grouped_pofs.mean().to_numpy()
    errors = percnts - means
    errors[0] *= -1
    ax[i].errorbar(
        NS, means,
        yerr=errors, label=r'$L_u$',
        capsize=5, capthick=3, marker='o', markersize=10
    )
    
    # PoF(L_n)
    grouped_pofs = pof_ln_df[~pof_ln_df['Sorted rows']][pof_ln_df['Distr'] == labels[i]].groupby('N')['PoF']
    percnts = grouped_pofs.quantile(q=[0.25, 0.75]).to_numpy().reshape(len(NS), 2).T
    means = grouped_pofs.mean().to_numpy()
    errors = percnts - means
    errors[0] *= -1
    ax[i].errorbar(
        NS, means,
        yerr=errors, label=r'$L_n$',
        capsize=5, capthick=3, marker='^', markersize=10
    )
    
    # ax[i].set_xlabel(r'$n$' + '\n\n' + '$' + labels[i] + r'$')
    # ax[i].set_xlabel('Number of agents' + '\n\n' + '$' + labels[i] + r'$')
    ax[i].set_title('\n\n' + '$' + labels[i] + r'$' + '\n')

ax[-1].legend()
ax[0].set_ylabel('Empirical price of fairness\n')
ax[1].set_xlabel('\nNumber of agents')
    
plt.show()


pof_lu_df = pd.DataFrame(columns=['PoF', 'N', 'Sorted rows'])

for i in range(len(labels)):
    for n_i, n in enumerate(NS):
        pof_lu_df = pd.concat([
            pof_lu_df,
            pd.DataFrame({
                'PoF': pof_u[i][n_i],
                'N': n,
                'Sorted rows': False
            }),
            pd.DataFrame({
                'PoF': pof_sorted_u[i][n_i],
                'N': n,
                'Sorted rows': True
            })
        ])

pof_lu_df.to_csv('pof_lu_df.csv', index=False)
pof_lu_df


pof_lu_df = pd.read_csv('pof_lu_df_new.csv')
pof_lu_df


for box in ax.artists:
    print(box.get_facecolor())


hatches = ['x', '\\\\'] * len(NS)
colors = [
    (0.19460784313725488, 0.45343137254901944, 0.632843137254902, 1),
    (0.8818627450980391, 0.5053921568627451, 0.17303921568627467, 1)
] * len(NS)

ax = sns.boxplot(x='N', y='PoF', hue='Sorted rows', data=pof_lu_df)
# for box, hatch, color in zip(ax.artists, hatches, colors):
#     box.set_hatch(hatch)
#     box.set_facecolor('w')
#     box.set_edgecolor(color)

for i, artist in enumerate(ax.artists):
    artist.set_hatch(hatches[i])
    artist.set_facecolor('w')
    artist.set_edgecolor(colors[i])
    
    for j in range(i * 6, i * 6 + 6):
        ax.lines[j].set_color(colors[i])
        ax.lines[j].set_mfc(colors[i])
        ax.lines[j].set_mec(colors[i])

my_patches = [
    # mpatches.Patch(facecolor=colors[0], hatch=hatches[0], label='False'),
    # mpatches.Patch(facecolor=colors[1], hatch=hatches[1], label='True')
    mpatches.Patch(facecolor='w', edgecolor=colors[0], hatch=hatches[0], label='False'),
    mpatches.Patch(facecolor='w', edgecolor=colors[1], hatch=hatches[1], label='True')
]

ax.legend(handles=my_patches, title='Sorted rows')
    
plt.ylabel('Price of fairness of $L_u$')
plt.show()


plt.figure(figsize=(20, 7))

colors = ['w', 'silver'] * len(NS)

ax = sns.boxplot(x='N', y='PoF', hue='Sorted rows', data=pof_lu_df)

for i, artist in enumerate(ax.artists):
    artist.set_facecolor(colors[i])

my_patches = [
    mpatches.Patch(facecolor=colors[0], edgecolor='k', label='Rows not sorted'),
    mpatches.Patch(facecolor=colors[1], edgecolor='k', label='Rows sorted')
]

ax.legend(handles=my_patches)
    
plt.ylabel(r'PoF$(L_u)$' + '\n')
plt.xlabel('\nNumber of agents')

plt.show()


pof_ln_df = pd.DataFrame(columns=['PoF', 'N', 'Sorted rows'])

for i in range(len(labels)):
    for n_i, n in enumerate(NS):
        pof_ln_df = pd.concat([
            pof_ln_df,
            pd.DataFrame({
                'PoF': pof_n[i][n_i],
                'N': n,
                'Sorted rows': False
            }),
            pd.DataFrame({
                'PoF': pof_sorted_n[i][n_i],
                'N': n,
                'Sorted rows': True
            })
        ])

pof_ln_df.to_csv('pof_ln_df.csv', index=False)
pof_ln_df


pof_ln_df = pd.read_csv('pof_ln_df_new.csv')
pof_ln_df


sns.boxplot(x='N', y='PoF', hue='Sorted rows', data=pof_ln_df)
plt.ylabel('Price of fairness of $L_n$')
plt.show()



