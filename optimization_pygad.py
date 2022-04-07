import pygad
import numpy
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from multiprocessing import Pool

from obj_func import *

# -------------------------------------- CONFIG SIM ------------------------------------------------

config_sim =\
{
    'render': False,
    'save_screeshot': True,
    'screenshot_filename': 'screenshots/screenshot.jpg',
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
    'pop_path': "data/country_data_vac_uniform.csv",
    'airport_path': "data/Airports_2010.csv",
    'air_travel_path': "data/Prediction_Monthly.csv"
}

# -------------------------------------- SIM -------------------------------------------------------

sim = sim_factory(config_sim)

def run_baseline(sim):
    """Runs baseline scenario"""

    time_horizon_days = sim.config['time_horizon_days']
    number_of_regions = sim.number_of_regions

    lockdown_input       = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_input = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    vaccination_input    = np.full((time_horizon_days, number_of_regions), 0, dtype=int)

    total_deaths = run(config_sim, sim, lockdown_input, border_closure_input, vaccination_input)

    return total_deaths

baseline_cost = run_baseline(sim)

# -------------------------------------- FITNESS FUNCTIONS -----------------------------------------

time_horizon_days = sim.config['time_horizon_days']
number_of_regions = sim.number_of_regions
population_sizes  = sim.population_sizes

def fitness_func_full(solution, solution_idx):
    """Lockdown and border closure for certain periods of time. This function requires
    num_genes=number_of_regions * 3."""

    TOTAL_DOSES = 1000000000
    VACCINATION_RATE = 0.005

    vaccination_rate = np.full((number_of_regions), VACCINATION_RATE, dtype=float)

    sol = solution[0:number_of_regions]
    share = np.absolute(sol)
    share = ((share / np.sum(share)) * TOTAL_DOSES).astype(np.uint64)

    num_can_vaccinate_each_day = (np.multiply(vaccination_rate, population_sizes)).astype(int)

    days_required = np.floor_divide(share, num_can_vaccinate_each_day).astype(int)
    num_vaccinated_last_day = np.mod(share, num_can_vaccinate_each_day).astype(int)

    vaccination_sol = np.full((time_horizon_days, number_of_regions), 0, dtype=int)
    for r in range(number_of_regions):
        for t in range(time_horizon_days):
            if t < days_required[r]:
                vaccination_sol[t][r] = num_can_vaccinate_each_day[r]
            if t == days_required[r]:
                vaccination_sol[t][r] = num_vaccinated_last_day[r]

    LEN_NPI_PERIOD = 28

    sol = solution[number_of_regions:]
    solution_mod = np.mod(sol.astype(int), time_horizon_days)
    npi_sol = solution_mod.reshape((number_of_regions, 2))

    lockdown_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    for r in range(number_of_regions):
        start_day_lockdown = npi_sol[r][0]
        end_day_lockdown = min(start_day_lockdown + LEN_NPI_PERIOD, time_horizon_days)
        for day in range(start_day_lockdown, end_day_lockdown):
            lockdown_sol[day][r] = 1
        start_day_border_closure = npi_sol[r][1]
        end_day_border_closure = min(start_day_border_closure + LEN_NPI_PERIOD, time_horizon_days)
        for day in range(start_day_border_closure, end_day_border_closure):
            border_closure_sol[day][r] = 1

    total_deaths = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_deaths

    return fitness

def fitness_func_npis(solution, solution_idx):
    """Lockdown and border closure only once, for certain periods of time. This function requires
    num_genes=number_of_regions * 2."""

    LEN_NPI_PERIOD = 28

    solution = np.mod(solution.astype(int), time_horizon_days)

    npi_sol = solution.reshape((number_of_regions, 2))

    lockdown_sol       = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    for r in range(number_of_regions):
        start_day_lockdown = npi_sol[r][0]
        end_day_lockdown = min(start_day_lockdown + LEN_NPI_PERIOD, time_horizon_days)
        for day in range(start_day_lockdown, end_day_lockdown):
            lockdown_sol[day][r] = 1
        start_day_border_closure = npi_sol[r][1]
        end_day_border_closure = min(start_day_border_closure + LEN_NPI_PERIOD, time_horizon_days)
        for day in range(start_day_border_closure, end_day_border_closure):
            border_closure_sol[day][r] = 1

    vaccination_sol = np.zeros((time_horizon_days, number_of_regions), dtype=int)

    total_deaths = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_deaths

    return fitness

def fitness_func_lockdown(solution, solution_idx):
    """Lockdown for certain periods of time. This function requires
    num_genes=number_of_regions * max_npi_periods."""

    LEN_NPI_PERIOD = 90
    MAX_NPI_PERIODS = 3

    num_npi_periods = (time_horizon_days // LEN_NPI_PERIOD) + 1

    solution = np.mod(solution.astype(int), num_npi_periods)

    lock_sol = solution.reshape((number_of_regions, MAX_NPI_PERIODS))

    # lock_sol = np.sign(lock_sol) + (lock_sol == 0)

    lockdown_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    for r in range(number_of_regions):
        for index in range(MAX_NPI_PERIODS):
            start_day = lock_sol[r][index] * LEN_NPI_PERIOD
            end_day = min(start_day + LEN_NPI_PERIOD, time_horizon_days)
            for day in range(start_day, end_day):
                lockdown_sol[day][r] = 1

    vaccination_sol = np.zeros((time_horizon_days, number_of_regions), dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)

    total_deaths = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_deaths

    return fitness

def fitness_func_variable_supply(solution, solution_idx):
    """Supply of vaccines is variable, where in a given supply period, available doses are shared
    out and administered no faster than a certain fixed rate. This function requires
    num_genes=number_of_regions * num_supply_periods."""

    TOTAL_DOSES = 1000000000
    VACCINATION_RATE = 0.005
    LEN_SUPPLY_PERIOD = 30

    vaccination_rate = np.full((number_of_regions), VACCINATION_RATE, dtype=float)

    num_supply_periods = (time_horizon_days // LEN_SUPPLY_PERIOD) + 1
    supply = np.full((num_supply_periods), int(TOTAL_DOSES / num_supply_periods), dtype=int)

    vac_sol = np.absolute(solution.reshape((num_supply_periods, number_of_regions)))
    vac_sol = ((vac_sol.T / np.sum(vac_sol, axis=1)) * supply).T

    num_can_vaccinate_each_day = (np.multiply(vaccination_rate, population_sizes)).astype(int)

    vaccination_sol = np.zeros((time_horizon_days, number_of_regions), dtype=int)
    for day in range(time_horizon_days):
        vaccination_sol[day] =\
            min((vac_sol[day // LEN_SUPPLY_PERIOD] / LEN_SUPPLY_PERIOD).astype(int),
                num_can_vaccinate_each_day)

    lockdown_sol       = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)

    total_deaths = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_deaths

    return fitness

def fitness_func_constant_rate(solution, solution_idx):
    """Shares out all doses on day 0 and administers the doses in each country at a constant rate
    until the doses are used. This function requires num_genes=number_of_regions."""

    TOTAL_DOSES = 1000000000
    VACCINATION_RATE = 0.005

    vaccination_rate = np.full((number_of_regions), VACCINATION_RATE, dtype=float)

    share = np.absolute(solution)
    share = ((share / np.sum(share)) * TOTAL_DOSES).astype(np.uint64)

    num_can_vaccinate_each_day = (np.multiply(vaccination_rate, population_sizes)).astype(int)

    days_required = np.floor_divide(share, num_can_vaccinate_each_day).astype(int)
    num_vaccinated_last_day = np.mod(share, num_can_vaccinate_each_day).astype(int)

    vaccination_sol = np.full((time_horizon_days, number_of_regions), 0, dtype=int)
    for r in range(number_of_regions):
        for t in range(time_horizon_days):
            if t < days_required[r]:
                vaccination_sol[t][r] = num_can_vaccinate_each_day[r]
            if t == days_required[r]:
                vaccination_sol[t][r] = num_vaccinated_last_day[r]

    lockdown_sol       = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)

    total_deaths = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_deaths

    return fitness

def fitness_func_day_zero(solution, solution_idx):
    """Shares out all doses on day 0 and immediately administers the doses. This function requires
    num_genes=number_of_regions."""

    TOTAL_DOSES = 1000000000

    share = np.absolute(solution)
    share = ((share / np.sum(share)) * TOTAL_DOSES).astype(np.uint64)

    vaccination_sol = np.full((time_horizon_days, number_of_regions), 0, dtype=int)
    vaccination_sol[0] = share

    lockdown_sol       = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)

    total_deaths = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_deaths

    return fitness

# -------------------------------------- CONFIG OPTIMIZER ------------------------------------------

config_optimizer =\
{
    'num_generations': 1000,
    'num_parents_mating': 12,
    'sol_per_pop': 24,
    'popsize': 24,
    'fitness_func': fitness_func_full,
    'num_genes': number_of_regions * 3
}

# -------------------------------------- OPTIMIZER -------------------------------------------------

fitness_func = config_optimizer['fitness_func']

def fitness_wrapper(solution):
    return fitness_func(solution, 0)

class PooledGA(pygad.GA):
    def cal_pop_fitness(self):
        global pool
        pop_fitness = pool.map(fitness_wrapper, self.population)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness

def on_generation(ga_instance):
    """Prints generation number and cost of best solution"""
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Cost       = {cost}".format(cost=\
          ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))

ga_instance = PooledGA(num_generations=config_optimizer['num_generations'],
                       num_parents_mating=config_optimizer['num_parents_mating'],
                       sol_per_pop=config_optimizer['sol_per_pop'],
                       num_genes=config_optimizer['num_genes'],
                       fitness_func=fitness_func,
                       on_generation=on_generation,
                       gene_type=float)

if __name__ == "__main__":

    # Load the saved GA instance
    # filename = 'genetic'
    # ga_instance = pygad.load(filename=filename)

    # Run the genetic algorithm
    with Pool(processes=12) as pool:
        ga_instance.run()

    # Plot the fitness curve
    ga_instance.plot_fitness()

    # Return the details of the best solution
    solution, solution_fitness, solution_idx =\
        ga_instance.best_solution(ga_instance.last_generation_fitness)

    # Save the GA instance
    filename = 'genetic'
    ga_instance.save(filename=filename)

    # Play back the best solution and save a screenshot of the final distribution
    sim.config['render'] = True
    fitness_func(solution, 0)
    sim.config['render'] = False

# -------------------------------------- COMMENTS --------------------------------------------------

# - Make each supply period (e.g. day) a gene (as a numpy array)?
# - Vaccine hesitancy data
# - Vaccine supply data
# - Vaccination rate data
