
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.colors import BoundaryNorm
from tqdm import tqdm

def run(solution):
    """Main simulation function"""

    time_horizon_days     = 365
    step_size             = 1
    beta                  = np.array([0.2723], dtype=float)
    gamma                 = np.array([1/9], dtype=float)

    number_of_strains       = 1
    number_of_age_groups    = 3

    number_of_regions = 2
    population_sizes = np.array([[1000000, 0, 0], [1000000, 0, 0]], dtype=np.uint64)
    initial_cases = np.array([[[1], [0], [0]], [[1], [0], [0]]], dtype=int)

    # number_of_regions = 3
    # population_sizes = np.array([[1000, 8000, 1000], [1000, 8000, 1000], [10000, 80000, 10000]], dtype=np.uint64)
    # initial_cases = np.array([[[1], [8], [1]], [[1], [8], [1]], [[10], [80], [10]]], dtype=int)

    contact_matrix = np.identity(number_of_regions, dtype=int)
    ifr = np.array([[0.0], [0.01], [0.05]], dtype=float)

    vaccination_input =\
        np.zeros((time_horizon_days, number_of_regions, number_of_age_groups), dtype=int)
    vaccination_input[0] = solution

    # Compartments (S: Susceptible and not vaccine hesitant,
    #               H: Susceptible and vaccine hesitant,
    #               I: Infected,
    #               R: Recovered,
    #               D: Dead)
    T = int(time_horizon_days * (1 / step_size))
    N = np.zeros((number_of_regions, number_of_age_groups), dtype=np.float64)
    S = np.zeros((T, number_of_regions, number_of_age_groups), dtype=np.float64)
    I = np.zeros((T, number_of_regions, number_of_age_groups, number_of_strains), dtype=np.float64)
    R = np.zeros((T, number_of_regions, number_of_age_groups), dtype=np.float64)
    D = np.zeros((T, number_of_regions, number_of_age_groups), dtype=np.float64)

    # Initial conditions
    for k in range(number_of_regions):
        for a in range(number_of_age_groups):
            N[k][a] = population_sizes[k][a]
            R[0][k][a] = 0
            D[0][k][a] = 0
            for i in range(number_of_strains):
                I[0][k][a][i] = initial_cases[k][a][i]
            S[0][k][a] = N[k][a] - R[0][k][a] -\
                         sum([I[0][k][a][i] for i in range(number_of_strains)])

    # Precalculate N_bar
    N_bar = np.zeros((number_of_regions), dtype=float)
    for j in range(number_of_regions):
        N_bar[j] = 1 / np.sum(N[j])

    total_doses_administered = np.zeros((number_of_regions, number_of_age_groups), dtype=np.uint64)
    t = 0
    done = False
    while not done:

        steps_in_a_day = int(1 / step_size)

        day = t // steps_in_a_day

        # Simulate transmission
        if t < T - 1:

            # Vaccination
            new_doses_administered = np.minimum(S[t], vaccination_input[day] / steps_in_a_day)
            S[t] = S[t] - new_doses_administered
            R[t] = R[t] + new_doses_administered
            total_doses_administered =\
                (total_doses_administered + new_doses_administered).astype(np.uint64)

            # Update
            A = np.matmul(contact_matrix, np.multiply(np.sum(I[t], axis=1), N_bar[:, None]))
            ones = np.ones((number_of_regions, number_of_age_groups, number_of_strains))

            S[t+1] = S[t] - step_size*np.multiply(np.matmul(A, beta)[:, None], S[t])
            I[t+1] = I[t] + step_size*np.multiply(np.multiply(A, beta[None, :])[:, None, :],
                                                  (S[t])[:, :, None])\
                          - step_size*np.multiply(I[t], gamma)
            R[t+1] = R[t] + step_size*np.matmul(np.multiply(I[t], ones - ifr[None, :, :]), gamma)
            D[t+1] = D[t] + step_size*np.matmul(np.multiply(I[t], ifr[None, :, :]), gamma)

        # Update step
        t += 1
        if t == T - 1:
            done = True

    total_deaths = round(np.sum(D[t]))
    total_cases = round(np.sum(R[t]) - np.sum(total_doses_administered)) + total_deaths

    return total_cases

def plot_heat_map(data):
    """Plots heat map"""

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    cmap = plt.get_cmap('PuOr') # RdYlGn
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    lower_bound = np.min(data)
    upper_bound = np.max(data)
    bounds = np.arange(lower_bound, upper_bound, math.ceil(((upper_bound - lower_bound) + 1)/256))
    idx=np.searchsorted(bounds,0)
    bounds=np.insert(bounds,idx,0)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.imshow(data,interpolation='none',norm=norm,cmap=cmap)
    cbar = plt.colorbar(ticks=np.linspace(lower_bound, upper_bound, 11), fraction=0.04525)
    cbar.ax.tick_params(labelsize=18)
    plt.show()

# TOTAL_DOSES = 7000
# dim = 100
# data = np.zeros((dim + 1, dim + 1), dtype=np.uint64)
# for i in tqdm(range(dim + 1)):
#     for j in range(dim + 1):
#         a = i/dim
#         b = j/dim
#         solution = np.array([[0, 0, TOTAL_DOSES * (1-a)],
#                              [0, 0, TOTAL_DOSES * a * (1-b)],
#                              [0, 0, TOTAL_DOSES * a * b]], dtype=np.uint64)
#         data[i][j] = run(solution)

# plot_heat_map(data)

# ind = np.unravel_index(np.argmin(data, axis=None), data.shape)

# a = ind[0]/dim
# b = ind[1]/dim

# print([0, 0, TOTAL_DOSES * (1-a)],
#       [0, 0, TOTAL_DOSES * a * (1-b)],
#       [0, 0, TOTAL_DOSES * a * b])

TOTAL_DOSES = 500000
dim = 100
data = np.zeros((dim + 1, ), dtype=np.uint64)
for i in tqdm(range(dim + 1)):
    a = i/dim
    solution = np.array([[TOTAL_DOSES * (1-a), 0, 0],
                         [TOTAL_DOSES * a, 0, 0]], dtype=np.uint64)
    data[i] = run(solution)

ind = np.unravel_index(np.argmin(data, axis=None), data.shape)

a = ind[0]/dim

print([TOTAL_DOSES * (1-a), 0, 0], [TOTAL_DOSES * a, 0, 0])

plt.figure(figsize=(6, 6))
font = {'size' : 12}
plt.rc('font', **font)

data = data / 1000

plt.ylim(np.min(data), np.max(data))
plt.xlim(0,100)

plt.ticklabel_format(style='plain')

plt.ylabel('Total Cases (thousands)')
plt.xlabel('Proportion of Country A vaccinated')
plt.xticks(ticks=[10 * a for a in range(11)],
           labels=[a / 10 for a in range(11)])

plt.plot(list(range(len(data))), data)
plt.show()
# plt.savefig("test.png")

"""With limited supply of vaccines, aim for herd immunity in the small countries. Eventually, when
the supply is large enough, it becomes optimal to vaccination only the large country (even if this
doesn't achieve herd immunity there, the saving in cases may still be more than the saving got by
vaccinating the small countries). Once the supply is large enough to get herd immunity in the large
country, then start vaccinating the small countries again."""