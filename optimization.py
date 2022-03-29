
from traceback import print_tb
import uuid
import os, warnings
import numpy as np
from joblib import Parallel, delayed

warnings.simplefilter("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from obj_func import *

class Optimizer():

    def __init__(self, sim, cost_function, config):

        self.population = None
        self.best_sol   = None
        self.sim        = sim
        self.cost       = cost_function
        self.config     = config

        # Switches to decide what to optimize
        self.optimize_max_vaccines   = config['optimize_max_vaccines']
        self.optimize_vaccine_timing = config['optimize_vaccine_timing']
        self.optimize_lockdown       = config['optimize_lockdown']
        self.optimize_border_closure = config['optimize_border_closure']

        # Optimization parameters
        self.popsize = config['popsize'] # Number of solutions to be evaluated each generation
        self.njobs   = config['njobs']   # Number of evalations to be performed in parallel
        self.mut     = config['mut']     # Mutation rate
        self.crossp  = config['crossp']  # Crossover rate
        self.n_gens  = config['n_gens']  # Number of generations to run

        # Simulation parameters
        self.time_horizon_days = sim.config['time_horizon_days']
        self.number_of_regions = sim.number_of_regions
        self.population_sizes  = sim.population_sizes
        self.max_doses         = sim.population_sizes - sim.vaccine_hesitant

        # Directory for saving and loading solutions
        self.populations_directory      = config['populations_directory']
        self.best_solutions_directory = config['best_solutions_directory']

        # Random seed
        self.random_seed = config['random_seed']
        np.random.seed(self.random_seed)

    def run_baseline(self):
        """Runs baseline scenario"""

        time_horizon_days = self.time_horizon_days
        number_of_regions = self.number_of_regions

        lockdown_input       = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
        border_closure_input = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
        vaccination_input    = np.full((time_horizon_days, number_of_regions), 0, dtype=int)

        total_deaths = run(config_sim, sim, lockdown_input,
                           border_closure_input, vaccination_input)

        print("Total baseline deaths:" + str(total_deaths))

    def initialize_population(self):
        """Generates intial population of solutions"""

        time_horizon_days = self.time_horizon_days
        number_of_regions = self.number_of_regions
        popsize           = self.popsize

        total_population_size = np.sum(self.population_sizes)
        total_doses_available = min(self.cost.total_available_doses, total_population_size)

        if self.optimize_max_vaccines:
            init_sol_max_vaccines = 0.8 * (total_doses_available / total_population_size) *\
                                    np.random.rand(popsize, number_of_regions)
        else:
            # If not optimizing this, share the doses equally by population size:
            init_sol_max_vaccines = (total_doses_available / total_population_size) *\
                                    np.full((popsize, number_of_regions), 1)

        if self.optimize_vaccine_timing:
            init_sol_vaccines_dist =\
                np.random.rand(popsize, time_horizon_days * number_of_regions)
        else:
            # If not optimizing this, share the doses equally across time:
            init_sol_vaccines_dist =\
                np.full((popsize, time_horizon_days * number_of_regions),
                        1 / time_horizon_days)

        if self.optimize_lockdown:
            init_sol_lockdown =\
                np.random.choice([-1,1], size=(popsize, time_horizon_days * number_of_regions))
        else:
            # If not optimizing this, no regions lockdown:
            init_sol_lockdown =\
                np.full((popsize, time_horizon_days * number_of_regions), -1)

        if self.optimize_border_closure:
            init_sol_border_closure =\
                np.random.choice([-1,1], size=(popsize, time_horizon_days * number_of_regions))
        else:
            # If not optimizing this, no regions do border closures:
            init_sol_border_closure =\
                np.full((popsize, time_horizon_days * number_of_regions), -1)

        initial_population =\
            np.concatenate([init_sol_max_vaccines, init_sol_vaccines_dist,
                            init_sol_lockdown, init_sol_border_closure], axis=1)

        self.population = initial_population.copy()

    def evaluate(self, solution):
        """Run one model evaluation with a given solution"""

        number_of_regions = self.number_of_regions
        time_horizon_days = self.time_horizon_days

        max_vaccines = solution[0:number_of_regions] * self.max_doses

        M = solution[number_of_regions::].reshape((time_horizon_days * 3, number_of_regions))

        vaccination_sol = M[0:time_horizon_days]
        vaccination_sol =\
                    (vaccination_sol / np.tile(vaccination_sol.sum(axis=0), (time_horizon_days, 1)))
        vaccination_sol = np.floor(vaccination_sol * np.tile(max_vaccines, (time_horizon_days, 1)))

        lockdown_sol = M[time_horizon_days:time_horizon_days * 2]
        border_closure_sol = M[time_horizon_days * 2:time_horizon_days * 3]

        total_deaths = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)
        total_cost = self.cost.cost(total_deaths, lockdown_sol, border_closure_sol, vaccination_sol)

        return total_cost

    def optimize(self):
        """Performs genetic algorithm"""

        time_horizon_days = self.time_horizon_days
        number_of_regions = self.number_of_regions

        # Containers to store best solutions
        history_fitness = []

        # Only relevant for differential evolution
        bounds = np.array([[0, 1] for _ in range(number_of_regions)] +\
                          [[0, 1] for _ in range(time_horizon_days * number_of_regions)] +\
                          [[-1, 1] for _ in range(time_horizon_days * number_of_regions * 2)])
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)

        # Initial conditions
        best_fitness = 10e10 # Best fitness so far
        best_sol = None      # Best solution so far

        population = self.population

        assert population is not None

        # Must have at least five solutions to perform genetic algorithm
        assert self.popsize >= 5

        print('Optimizing via DE...')

        for step in range(self.n_gens):

            print("Generation:", step, "Fitness:", best_fitness)

            # Evaluate the entire population of solutions and select the survivors
            fitness = Parallel(n_jobs = self.njobs, verbose=0)(delayed(self.evaluate)(solution)
                                                               for solution in population)
            best_idx = np.argmin(fitness)

            if fitness[best_idx] < best_fitness:
                best_sol = population[best_idx]
                best_fitness = fitness[best_idx]
                history_fitness.append(best_fitness)

            sorter = np.argsort(fitness)
            survivors = population[sorter][0:int(len(fitness)/2)]
            new_pop = survivors.copy()

            # Create the new population. We need to partition this in each solution matrix as we
            # need to use differential evolution for continous numbers and genetic algorithms for
            # discrete ones.
            for j, _ in enumerate(survivors):

                # Vaccination levels across regions
                init = 0
                fin = number_of_regions
                if self.optimize_max_vaccines:
                    idxs = [idx for idx in range(len(survivors)) if idx != j]
                    a, b, c = population[np.random.choice(idxs, 3, replace = False), init:fin]
                    mutant = np.clip(a + self.mut * (b - c), 0, 1)
                    length_s = fin - init
                    cross_points = np.random.rand(length_s) < self.crossp
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, length_s)] = True
                    trial = np.where(cross_points, mutant, population[j, init:fin])
                    new_max_vaccinations = min_b[init:fin] + trial * diff[init:fin]
                else:
                    new_max_vaccinations = population[0, init:fin]

                # Vaccination inter-temporal distribution (timing)
                init = number_of_regions
                fin = number_of_regions + time_horizon_days * number_of_regions
                if self.optimize_vaccine_timing:
                    idxs = [idx for idx in range(len(survivors)) if idx != j]
                    a, b, c = population[np.random.choice(idxs, 3, replace = False), init:fin]
                    mutant = np.clip(a + self.mut * (b - c), 0, 1)
                    length_s = fin - init
                    cross_points = np.random.rand(length_s) < self.crossp
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, length_s)] = True
                    trial = np.where(cross_points, mutant, population[j, init:fin])
                    new_vaccination_dist = min_b[init:fin] + trial * diff[init:fin]
                else:
                    new_vaccination_dist = population[0, init:fin]

                # Lockdowns
                init = number_of_regions + time_horizon_days * number_of_regions
                fin = number_of_regions + time_horizon_days * number_of_regions +\
                                          time_horizon_days * number_of_regions
                if self.optimize_lockdown:
                    length_s = fin - init
                    p1, p2 = survivors[np.random.choice(range(len(survivors)),
                                                        2, replace=False), init:fin]
                    c1 = p1.copy()
                    if np.random.rand() < self.crossp:
                        pt = np.random.randint(1, len(p1) - 2)
                        c1[pt:] = p2[pt:]
                    c1[np.random.rand(length_s) < self.mut] *= -1
                    new_lockdown = c1
                else:
                    new_lockdown = population[0, init:fin]

                # Border closures
                init = number_of_regions + time_horizon_days * number_of_regions +\
                                           time_horizon_days * number_of_regions
                fin = number_of_regions + time_horizon_days * number_of_regions +\
                                          time_horizon_days * number_of_regions * 2
                if self.optimize_lockdown:
                    length_s = fin - init
                    p1, p2 = survivors[np.random.choice(range(len(survivors)),
                                                        2, replace=False), init:fin]
                    c1 = p1.copy()
                    if np.random.rand() < self.crossp:
                        pt = np.random.randint(1, len(p1) - 2)
                        c1[pt:] = p2[pt:]
                    c1[np.random.rand(length_s) < self.mut] *= -1
                    new_border_closure = c1
                else:
                    new_border_closure = population[0, init:fin]

                # Stitch back all the solutions
                new_solution = np.concatenate([new_max_vaccinations, new_vaccination_dist,
                                               new_lockdown, new_border_closure])
                new_pop = np.concatenate([new_pop, [new_solution]])

            population = new_pop.copy()

        self.population = population

        self.best_sol = best_sol

    def save_population(self):
        """Saves population to disk"""

        print("Saving population")

        np.save(self.populations_directory + str(uuid.uuid4().hex), self.population)

    def save_best_solution(self):
        """Saves best solution to disk"""

        print("Saving best solution")

        if self.best_sol is not None:
            np.save(self.best_solutions_directory + str(uuid.uuid4().hex), self.best_sol)

    def load_population(self):
        """Loads population from disk"""

        populations_directory = self.populations_directory

        disk_population = []

        for filename in os.listdir(populations_directory):
            f = os.path.join(populations_directory, filename)
            if os.path.isfile(f):
                sol = np.load(f)
                disk_population.append(sol)

        if len(disk_population) > 0:
            print("Loading a saved population")
            self.population = disk_population[0]

    def load_best_solution(self):
        """Loads solutions from disk and inserts them into the population"""

        best_solutions_directory = self.best_solutions_directory

        disk_population = []

        for filename in os.listdir(best_solutions_directory):
            f = os.path.join(best_solutions_directory, filename)
            if os.path.isfile(f):
                sol = np.load(f)
                disk_population.append(sol)

        if len(disk_population) > 0:
            print("Loading a best solution")
            self.best_sol = disk_population[0]

        if self.population is not None:
            num_to_insert = min(len(self.population), len(disk_population))
            print("Inserting " + str(num_to_insert) + " solution(s) from disk")
            for i in range(num_to_insert):
                self.population[i] = disk_population[i]

    def empty_populations_directory(self):
        """Deletes populations from disk"""

        print("Emptying populations directory")

        for filename in os.listdir(self.populations_directory):
            f = os.path.join(self.populations_directory, filename)
            if os.path.isfile(f):
                os.remove(f)

    def empty_best_solutions_directory(self):
        """Deletes best solutions from disk"""

        print("Emptying best solutions directory")

        for filename in os.listdir(self.best_solutions_directory):
            f = os.path.join(self.best_solutions_directory, filename)
            if os.path.isfile(f):
                os.remove(f)

    def play_sol(self, solution):
        """Run best solution with render mode activated"""

        if solution is not None:
            self.sim.config['render'] = True
            self.evaluate(solution)
            self.sim.config['render'] = False

# -------------------------------------- COST FUNCTION ---------------------------------------------

class CostFunction():

    def __init__(self, config):

        self.total_available_doses              = config['total_available_doses']
        self.max_days_allowed_in_lockdown       = config['max_days_allowed_in_lockdown']
        self.max_days_allowed_in_border_closure = config['max_days_allowed_in_border_closure']

    def cost(self, total_deaths, lockdown_input, border_closure_input, vaccination_input):
        """Calculates total cost of a simulation"""

        cost = total_deaths

        total_doses = np.sum(vaccination_input)
        if total_doses > self.total_available_doses:
            cost += 10e10

        days_in_lockdown = ((1 + lockdown_input) / 2).sum(axis=0)
        if np.max(days_in_lockdown) > self.max_days_allowed_in_lockdown:
            cost += 10e10

        days_in_border_closure = ((1 + border_closure_input) / 2).sum(axis=0)
        if np.max(days_in_border_closure) > self.max_days_allowed_in_border_closure:
            cost += 10e10

        return cost

# -------------------------------------- CONFIG SIM ------------------------------------------------

config_sim =\
       {'render': False,
        'save_screeshot': True,
        'screenshot_filename': 'screenshots/screenshot_3_2.jpg',
        'display_width': 1200,
        'display_height': 700,
        'font_size': 20,
        'points_per_polygon': 400,
        'infection_cmap': "Oranges",
        'vaccination_cmap': "Blues",
        'international_travel_enabled': True,
        'refresh_rate': 60,
        'time_horizon_days': 365,
        'euler_scheme_step_size_days': 1/1,
        'transmission_probabilities': np.array([0.00035]),
        'removal_rates_per_day': np.array([1 / 9]),
        'infection_fatality_rate_by_age': np.array([[0.0, 0.001, 0.01]]),
        'initial_cases_dict': {'CN': [10000]},
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

sim = sim_factory(config_sim)

# -------------------------------------- CONFIG COST FUNCTION --------------------------------------

config_cost =\
       {'total_available_doses': 1000000000,
        'max_days_allowed_in_lockdown': 100,
        'max_days_allowed_in_border_closure': 100}

# -------------------------------------- BUILD COST FUNCTION ---------------------------------------

cost_function = CostFunction(config_cost)

# -------------------------------------- CONFIG OPTIMIZER ------------------------------------------

config_optimizer =\
       {'optimize_max_vaccines': True,
        'optimize_vaccine_timing': False,
        'optimize_lockdown': False,
        'optimize_border_closure': False,
        'popsize': 24,
        'njobs': 12,
        'n_gens': 1000,
        'mut': 0.8,
        'crossp': 0.7,
        'random_seed': 2,
        'populations_directory': './populations/',
        'best_solutions_directory': './best_solutions/'}

# -------------------------------------- BUILD OPTIMIZER -------------------------------------------

optimizer = Optimizer(sim, cost_function, config_optimizer)

# -------------------------------------- OPTIMIZE --------------------------------------------------

# optimizer.run_baseline()

optimizer.initialize_population()

# optimizer.load_best_solution()

# optimizer.load_population()

optimizer.optimize()

# optimizer.empty_populations_directory()

# optimizer.empty_best_solutions_directory()

# optimizer.save_population()

# optimizer.save_best_solution()

# optimizer.load_best_solution()

optimizer.play_sol(optimizer.best_sol)

# -------------------------------------- COMMENTS --------------------------------------------------

# - Implement time varying vaccine dose limits in the cost function etc...
