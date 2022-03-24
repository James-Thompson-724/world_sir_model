import matplotlib.pyplot as plt
import os, warnings, csv
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

warnings.simplefilter("ignore")


from obj_func import *



# -------------------------------------- CONFIG ----------------------------------------------------

config_world =\
       {'render': False,
        'game_input': False,
        'display_width': 1200,
        'display_height': 700,
        'font_size': 20,
        'points_per_polygon': 400,
        'cmap': "Oranges",
        'refresh_rate': 60,
        'time_horizon_days': 365,
        'euler_scheme_step_size_days': 1/24,
        'transmission_probabilities': np.array([0.00035, 0.00035]),
        'removal_rates_per_day': np.array([1 / 9, 1 / 9]),
        'infection_fatality_rate_by_age': np.array([[0.0, 0.001, 0.01], [0.0, 0.001, 0.01]]),
        'initial_cases_dict': {'CN': 1000},
        'local_travel_prob_per_day': 0.0,
        'distance_threshold': 50,
        'contacts_per_day': 778,
        'lockdown_factor': 1/10,
        'border_closure_factor': 1/10,
        'max_norm_prevalance_to_plot': 1.0,
        'shp_path': "data/CNTR_RG_60M_2020_4326.shp",
        'pop_path': "data/country_data.csv",
        'airport_path': "data/Airports_2010.csv",
        'air_travel_path': "data/Prediction_Monthly.csv"}

# -------------------------------------- BUILD SIMULATOR -------------------------------------------

sim = sim_factory(config_world)

# -------------------------------------- INPUT -----------------------------------------------------

time_horizon_days = sim.config['time_horizon_days']
number_of_regions = sim.number_of_regions

# Vaccination input refers to the number of doses allocated to each region each day. The total
# number of doses allocated to a region over the simulation need not be more than the number of
# people in that region who are willing to be vaccinated. For a region with id i, this number is
# given by the difference
#
#                   max_doses := sim.population_sizes[i] - sim.vaccine_hesitant[i]
#
# One could, for example, set
#
#             vaccination_input[t][i] = int(max_doses / time_horizon_days)
#
# to have all willing inviduals vaccinated at a uniform rate over the course of the simulation.

# vaccination_input = np.full((time_horizon_days, number_of_regions), 0, dtype=int)


# Lockdown and border closure inputs should take the value 1 or -1. The value 1 indicates that the
# lockdown or border closure status does not change at that time step. The value -1 indicates that
# the status changes at that time step. For example, in the case of only one region, the lockdown
# input
#
#                        lockdown_input = [[1,1,1,1,-1,1,1,1,1,-1,1,1,1]]
#
# indicates that a lockdown starts at day = 4 and ends at day = 9, and similarly for the border
# closure input.

# lockdown_input       = np.full((time_horizon_days, number_of_regions), 1, dtype=int)
# border_closure_input = np.full((time_horizon_days, number_of_regions), 1, dtype=int)

# print("Total cost:", total_cost)

# -------------------------------------- OPTIMIZATION --------------------------------------------------


max_doses = sim.population_sizes - sim.vaccine_hesitant


# Run one model evaluation with a given solution
def evaluate(solution):
    max_vaccines = solution[0:number_of_regions] * max_doses
    M = solution[number_of_regions::].reshape((time_horizon_days*3, number_of_regions))
    vaccination_sol = M[0:time_horizon_days]
    vaccination_sol = (vaccination_sol / np.tile(vaccination_sol.sum(axis=0), (time_horizon_days,1)))
    vaccination_sol = np.floor(vaccination_sol*np.tile(max_vaccines, (time_horizon_days,1)))
    lockdown_sol = M[time_horizon_days:time_horizon_days*2]
    border_closure_sol = M[time_horizon_days*2:time_horizon_days*3]
    total_deaths = run(config_world, sim, lockdown_sol, border_closure_sol, vaccination_sol)
    total_cost = cost(total_deaths, lockdown_sol, border_closure_sol, vaccination_sol)
    return total_cost


# switches to decide what to optimize
optimize_max_vaccines = True
optimize_vaccine_timing = True
optimize_lockdown = True
optimize_border_closure = True


# optimization parameters
popsize = 48 # number og solutions to be evaluated each generation
njobs = 48 # number of evalations to be performed in parallel
mut=0.8 # matation rate
crossp=0.7 # crossover rate
n_gens = 25 # number of generations to run


# initialize a popualation of solutions with random solutions
if optimize_max_vaccines:
    init_sol_max_vaccines = np.random.rand(popsize, number_of_regions)
else:
    # change this to define an alternative fixed solution
    init_sol_max_vaccines = np.tile(max_doses, (popsize,1))

if optimize_vaccine_timing:
    init_sol_vaccines_dist = np.random.rand(popsize, time_horizon_days*number_of_regions)
else:
    # change this to define an alternative fixed solution
    init_sol_vaccines_dist = np.zeros((popsize, time_horizon_days*number_of_regions))
    
if optimize_lockdown:
    init_sol_lockdown = np.random.choice([-1,1], size=(popsize,time_horizon_days*number_of_regions))
else:
    # change this to define an alternative fixed solution
    init_sol_lockdown = np.ones((popsize, time_horizon_days*number_of_regions))

if optimize_lockdown:
    init_sol_border_closure = np.random.choice([-1,1], size=(popsize,time_horizon_days*number_of_regions))
else:
    # change this to define an alternative fixed solution
    init_sol_border_closure = np.ones((popsize, time_horizon_days*number_of_regions))

population = np.concatenate([init_sol_max_vaccines, init_sol_vaccines_dist, init_sol_lockdown, init_sol_border_closure], axis=1)

# initial conditions
best_fitness = 10e10 # best fitness so far
best_sol = None # best solution so far

# only relevant for differential evolution
bounds = np.array([[0,1] for i in range(number_of_regions)] + [[0, 1] for i in range(time_horizon_days*number_of_regions)] + [[-1,1] for i in range(time_horizon_days*number_of_regions*2)])
min_b, max_b = np.asarray(bounds).T
diff = np.fabs(min_b - max_b)

# containers tro store best solutions
history_solutions = []
history_fitness = []

print('Finding via DE...')
for step in range(n_gens):
    
    # evaluate the entire population of solutions and select the survivors
    print(step, best_fitness)
    fitness = Parallel(n_jobs=njobs, verbose=0)(delayed(evaluate)(solution) for solution in population)
    best_idx = np.argmin(fitness)
    
    if fitness[best_idx] < best_fitness:
        best_sol = population[best_idx]
        best_fitness = fitness[best_idx]
        history_solutions.append(best_sol)
        history_fitness.append(best_fitness)
    
    sorter = np.argsort(fitness)
    survivors = population[sorter][0:int(len(fitness)/2)]
    new_pop = survivors.copy()
    
    # create the new population
    # we need to partition this in each solution matrix as we need to use
    # differential evolution for continous numbers and genetic algorithms for discrete ones
    for j, solution in enumerate(survivors):
        
        
        # vaccination levels across regions
        if optimize_max_vaccines:
            idxs = [idx for idx in range(len(survivors)) if idx != j]
            init = 0
            fin = number_of_regions
            a, b, c = population[np.random.choice(idxs, 3, replace = False), init:fin]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            length_s = fin-init
            cross_points = np.random.rand(length_s) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, length_s)] = True
            trial = np.where(cross_points, mutant, population[j,init:fin])
            new_max_vaccinations = min_b[init:fin] + trial * diff[init:fin]
        else:
            new_max_vaccinations = population[0, init:fin]
        
        
        # vaccination inter-temporal distribution (timing)
        if optimize_vaccine_timing:
            idxs = [idx for idx in range(len(survivors)) if idx != j]
            init = number_of_regions
            fin = number_of_regions+time_horizon_days*number_of_regions
            a, b, c = population[np.random.choice(idxs, 3, replace = False), init:fin]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            length_s = fin-init
            cross_points = np.random.rand(length_s) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, length_s)] = True
            trial = np.where(cross_points, mutant, population[j,init:fin])
            new_vaccination_dist = min_b[init:fin] + trial * diff[init:fin]
        else:
            new_vaccination_dist = population[0, init:fin]
        
        
        # lockdowns
        if optimize_lockdown:
            init = number_of_regions+time_horizon_days*number_of_regions
            fin = number_of_regions+time_horizon_days*number_of_regions + time_horizon_days*number_of_regions
            length_s = fin-init
            p1, p2 = survivors[np.random.choice(range(len(survivors)), 2, replace=False), init:fin]
            c1 = p1.copy()
            if np.random.rand() < crossp:
            		pt = np.random.randint(1, len(p1)-2)
            		c1[pt:] = p2[pt:]
            c1[np.random.rand(length_s) < mut] *= -1
            new_lockdown = c1
        else:
            new_lockdown = population[0, init:fin]
            
        
        # border closures
        if optimize_lockdown:
            init = number_of_regions+time_horizon_days*number_of_regions + time_horizon_days*number_of_regions
            fin = number_of_regions+time_horizon_days*number_of_regions + time_horizon_days*number_of_regions * 2
            length_s = fin-init
            p1, p2 = survivors[np.random.choice(range(len(survivors)), 2, replace=False), init:fin]
            c1 = p1.copy()
            if np.random.rand() < crossp:
            		pt = np.random.randint(1, len(p1)-2)
            		c1[pt:] = p2[pt:]
            c1[np.random.rand(length_s) < mut] *= -1
            new_border_closure = c1
        else:
            new_border_closure = population[0, init:fin]
        
        
        # stitch back all the solutions
        new_solution = np.concatenate([new_max_vaccinations, new_vaccination_dist, new_lockdown, new_border_closure])
        new_pop = np.concatenate([new_pop, [new_solution]])
        
    population = new_pop.copy()
    


















































































