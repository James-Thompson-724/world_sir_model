
import pygad
import numpy as np
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from multiprocessing import Pool

from obj_func_age import *

# -------------------------------------- CONFIG SIM ------------------------------------------------

config_sim =\
{
    'render': False,
    'save_data': False,
    'save_screeshot': True,
    'screenshot_filename': 'screenshots/screenshot.jpg',
    'display_width': 1200,
    'display_height': 700,
    'font_size': 20,
    'points_per_polygon': 400,
    'infection_cmap': "Oranges",
    'vaccination_cmap': "Blues",
    'vaccination_0_cmap': "Greens",
    'vaccination_1_cmap': "Reds",
    'vaccination_2_cmap': "Purples",
    'international_travel_enabled': True,
    'refresh_rate': 60,
    'time_horizon_days': 365,
    'euler_scheme_step_size_days': 1/1,
    'transmission_probabilities': np.array([0.00035]),
    'removal_rates_per_day': np.array([1 / 9]),
    'infection_fatality_rate_by_age':
      np.array([[0.000070, 0.000054, 0.000040, 0.000032, 0.000027,
                 0.000024, 0.000023, 0.000023, 0.000023, 0.000025,
                 0.000028, 0.000031, 0.000036, 0.000042, 0.000050,
                 0.000060, 0.000071, 0.000085, 0.000100, 0.000118,
                 0.000138, 0.000162, 0.000188, 0.000219, 0.000254,
                 0.000293, 0.000337, 0.000386, 0.000442, 0.000504,
                 0.000573, 0.000650, 0.000735, 0.000829, 0.000932,
                 0.001046, 0.001171, 0.001307, 0.001455, 0.001616,
                 0.001789, 0.001976, 0.002177, 0.002391, 0.002620,
                 0.002863, 0.003119, 0.003389, 0.003672, 0.003968,
                 0.004278, 0.004606, 0.004958, 0.005342, 0.005766,
                 0.006242, 0.006785, 0.007413, 0.008149, 0.009022,
                 0.010035, 0.011162, 0.012413, 0.013803, 0.015346,
                 0.017058, 0.018957, 0.021064, 0.023399, 0.025986,
                 0.028851, 0.032022, 0.035527, 0.039402, 0.043679,
                 0.048397, 0.053597, 0.059320, 0.065612, 0.072520,
                 0.080093, 0.088381, 0.097437, 0.107311, 0.118054,
                 0.129717, 0.142346, 0.155984, 0.170669, 0.186431,
                 0.203292, 0.221263, 0.240344, 0.260519, 0.281760,
                 0.304021, 0.327239, 0.351335, 0.376213, 0.401762,
                 0.427856]]),
    'initial_cases_dict': {'CN': [10000]},
    'local_travel_prob_per_day': 0.00005,
    'distance_threshold': 50,
    'contacts_per_day': 778,
    'lockdown_factor': 1/10,
    'border_closure_factor': 1/10,
    'max_norm_prevalance_to_plot': 1.0,
    'shp_path': "data/data_shapefiles/CNTR_RG_60M_2020_4326.shp",
    'pop_path': "data/compiled_data_no_hesitancy.csv",
    'airport_path': "data/data_air_travel/Airports_2010.csv",
    'air_travel_path': "data/data_air_travel/Prediction_Monthly.csv"
}

# -------------------------------------- SIM -------------------------------------------------------

sim = sim_factory(config_sim)

def run_baseline(sim):
    """Runs baseline scenario"""

    time_horizon_days    = sim.config['time_horizon_days']
    number_of_regions    = sim.number_of_regions
    number_of_age_groups = sim.number_of_age_groups

    lockdown_input =\
        np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_input =\
        np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    vaccination_input =\
        np.full((time_horizon_days, number_of_regions, number_of_age_groups), 0, dtype=int)

    total_cost = run(config_sim, sim, lockdown_input, border_closure_input, vaccination_input)

    return total_cost

baseline_cost = run_baseline(sim)

# -------------------------------------- FITNESS FUNCTIONS -----------------------------------------

time_horizon_days    = sim.config['time_horizon_days']
number_of_regions    = sim.number_of_regions
population_sizes     = np.sum(sim.population_sizes_by_age_group, axis=1)
number_of_age_groups = sim.number_of_age_groups

def fitness_func_full(solution, solution_idx):
    """Lockdown and border closure for certain periods of time and vaccinate at a constant rate.
    This function requires:

    num_genes = (number_of_regions * 2) +
                number_of_regions + (number_of_regions * number_of_age_groups)."""

    TOTAL_DOSES = 1000000000 # The total number of doses available
    VACCINATION_RATE = 0.006 # The maximum proportion of population a country can vaccinate each day

    sol = solution[0:number_of_regions + (number_of_regions*number_of_age_groups)]

    vaccination_rate = np.full((number_of_regions), VACCINATION_RATE, dtype=float)

    solution_part_1 = sol[0:number_of_regions]
    solution_part_2 = sol[number_of_regions:]

    share = np.absolute(solution_part_1)
    share = ((share / np.sum(share)) * TOTAL_DOSES).astype(np.uint64)

    dist = np.absolute(solution_part_2)
    dist = np.reshape(dist, (number_of_regions, number_of_age_groups))
    dist_sum = dist.sum(axis=1)
    dist_sum[dist_sum == 0] = 1
    dist = np.transpose(np.transpose(dist) / dist_sum)

    num_can_vaccinate_each_day = (np.multiply(vaccination_rate, population_sizes)).astype(int)
    days_required = np.floor_divide(share, num_can_vaccinate_each_day).astype(int)
    num_vaccinated_last_day = np.mod(share, num_can_vaccinate_each_day).astype(int)

    vaccination_sol =\
        np.zeros((number_of_regions, time_horizon_days, number_of_age_groups), dtype=int)

    for r in range(number_of_regions):

        phase_1 = (num_can_vaccinate_each_day[r] * dist[r]).astype(int)
        phase_2 = (num_vaccinated_last_day[r] * dist[r]).astype(int)
        phase_3 = (0 * dist[r]).astype(int)

        phase_1_duration = min(time_horizon_days, days_required[r])
        phase_2_duration = min(time_horizon_days - phase_1_duration, 1)
        phase_3_duration = time_horizon_days - phase_1_duration - phase_2_duration

        part_1 = np.tile(phase_1, (phase_1_duration, 1))
        part_2 = np.tile(phase_2, (phase_2_duration, 1))
        part_3 = np.tile(phase_3, (phase_3_duration, 1))

        vaccination_sol[r] = np.concatenate((part_1, part_2, part_3), axis=0)

    vaccination_sol = np.swapaxes(vaccination_sol, 0, 1)

    LEN_NPI_PERIOD = 28

    sol = solution[number_of_regions + (number_of_regions*number_of_age_groups):]
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

    total_cost = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_cost

    return fitness

def fitness_func_npis(solution, solution_idx):
    """Lockdown and border closure only once, for a certain length of time. This function requires:

    num_genes = number_of_regions * 2."""

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

    vaccination_sol =\
        np.zeros((time_horizon_days, number_of_regions, number_of_age_groups), dtype=int)

    total_cost = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_cost

    return fitness

def fitness_func_lockdown(solution, solution_idx):
    """Multiple lockdowns for certain lengths of time. This function requires:

    num_genes = number_of_regions * max_npi_periods."""

    LEN_NPI_PERIOD = 28
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

    vaccination_sol =\
        np.zeros((time_horizon_days, number_of_regions, number_of_age_groups), dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)

    total_cost = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_cost

    return fitness

def fitness_func_variable_supply(solution, solution_idx):
    """Supply of vaccines is variable, where in a given supply period, available doses are shared
    out and administered no faster than a certain fixed rate. This function requires:

    num_genes = (number_of_regions * num_supply_periods) +
                (number_of_regions * number_of_age_groups * num_supply_periods)."""

    TOTAL_DOSES = 1000000000 # The total number of doses available
    VACCINATION_RATE = 0.006 # The maximum proportion of population a country can vaccinate each day
    LEN_SUPPLY_PERIOD = 30   # The time period is divided into supply periods of this length in days

    vaccination_rate = np.full((number_of_regions), VACCINATION_RATE, dtype=float)

    num_supply_periods = (time_horizon_days // LEN_SUPPLY_PERIOD) + 1

    # Supply the doses equally over time
    supply = np.full((num_supply_periods), int(TOTAL_DOSES / num_supply_periods), dtype=int)

    # Supply the doses according to a time-dependent distribution of appropriate length
    # supply = np.array([0.0005, 0.0004, 0.0007, 0.0012, 0.0023, 0.0030,
    #                    0.0040, 0.0040, 0.0040, 0.0040, 0.0040, 0.0040, 0.0040])
    # supply = ((supply / np.sum(supply)) * TOTAL_DOSES).astype(int)

    solution_part_1 = solution[0:number_of_regions * num_supply_periods]
    solution_part_2 = solution[number_of_regions * num_supply_periods:]

    vac_sol = np.absolute(solution_part_1.reshape((num_supply_periods, number_of_regions)))
    vac_sol = ((vac_sol.T / np.sum(vac_sol, axis=1)) * supply).T

    dist = np.absolute(solution_part_2)
    dist = np.reshape(dist, (num_supply_periods, number_of_regions, number_of_age_groups))
    for l in range(num_supply_periods):
        dist_sum = dist[l].sum(axis=1)
        dist_sum[dist_sum == 0] = 1
        dist[l] = np.transpose(np.transpose(dist[l]) / dist_sum)

    num_can_vaccinate_each_day = (np.multiply(vaccination_rate, population_sizes)).astype(int)

    vaccination_sol =\
        np.zeros((time_horizon_days, number_of_regions, number_of_age_groups), dtype=int)
    for day in range(time_horizon_days):
        supply_period = day // LEN_SUPPLY_PERIOD
        share = np.minimum((vac_sol[supply_period] / LEN_SUPPLY_PERIOD).astype(int),
                            num_can_vaccinate_each_day)
        vaccination_sol[day] = (np.multiply(share[:, None], dist[supply_period])).astype(int)

    lockdown_sol       = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)

    total_cost = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_cost

    return fitness

def fitness_func_constant_rate(solution, solution_idx):
    """Shares out all doses on day 0 and administers the doses in each country at a constant rate
    until the doses are used. This function requires:

    num_genes = number_of_regions + (number_of_regions * number_of_age_groups)."""

    TOTAL_DOSES = 1000000000 # The total number of doses available
    VACCINATION_RATE = 0.006 # The maximum proportion of population a country can vaccinate each day

    vaccination_rate = np.full((number_of_regions), VACCINATION_RATE, dtype=float)

    solution_part_1 = solution[0:number_of_regions]
    solution_part_2 = solution[number_of_regions:]

    share = np.absolute(solution_part_1)
    share = ((share / np.sum(share)) * TOTAL_DOSES).astype(np.uint64)

    dist = np.absolute(solution_part_2)
    dist = np.reshape(dist, (number_of_regions, number_of_age_groups))
    dist_sum = dist.sum(axis=1)
    dist_sum[dist_sum == 0] = 1
    dist = np.transpose(np.transpose(dist) / dist_sum)

    num_can_vaccinate_each_day = (np.multiply(vaccination_rate, population_sizes)).astype(int)
    days_required = np.floor_divide(share, num_can_vaccinate_each_day).astype(int)
    num_vaccinated_last_day = np.mod(share, num_can_vaccinate_each_day).astype(int)

    vaccination_sol =\
        np.zeros((number_of_regions, time_horizon_days, number_of_age_groups), dtype=int)

    for r in range(number_of_regions):

        phase_1 = (num_can_vaccinate_each_day[r] * dist[r]).astype(int)
        phase_2 = (num_vaccinated_last_day[r] * dist[r]).astype(int)
        phase_3 = (0 * dist[r]).astype(int)

        phase_1_duration = min(time_horizon_days, days_required[r])
        phase_2_duration = min(time_horizon_days - phase_1_duration, 1)
        phase_3_duration = time_horizon_days - phase_1_duration - phase_2_duration

        part_1 = np.tile(phase_1, (phase_1_duration, 1))
        part_2 = np.tile(phase_2, (phase_2_duration, 1))
        part_3 = np.tile(phase_3, (phase_3_duration, 1))

        vaccination_sol[r] = np.concatenate((part_1, part_2, part_3), axis=0)

    vaccination_sol = np.swapaxes(vaccination_sol, 0, 1)

    lockdown_sol       = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)

    total_cost = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_cost

    return fitness

def fitness_func_day_zero(solution, solution_idx):
    """Shares out all doses on day 0 and immediately administers the doses. This function requires:

    num_genes = number_of_regions * number_of_age_groups."""

    TOTAL_DOSES = 1000000000 # The total number of doses available

    share = np.absolute(solution)
    share = ((share / np.sum(share)) * TOTAL_DOSES).astype(np.uint64)
    share = np.reshape(share, (number_of_regions, number_of_age_groups))

    vaccination_sol =\
        np.full((time_horizon_days, number_of_regions, number_of_age_groups), 0, dtype=int)
    vaccination_sol[0] = share

    lockdown_sol       = np.full((time_horizon_days, number_of_regions), -1, dtype=int)
    border_closure_sol = np.full((time_horizon_days, number_of_regions), -1, dtype=int)

    total_cost = run(config_sim, sim, lockdown_sol, border_closure_sol, vaccination_sol)

    fitness = baseline_cost - total_cost

    return fitness

# -------------------------------------- CONFIG OPTIMIZER ------------------------------------------

config_optimizer =\
{
    'num_generations': 10000,
    'num_parents_mating': 7,
    'sol_per_pop': 21,
    'fitness_func': fitness_func_day_zero,
    'num_genes': (number_of_regions * number_of_age_groups)
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
    print("Fitness    = {cost}".format(cost=\
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
    with Pool(processes=7) as pool:
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
    sim.config['save_data'] = True
    fitness_func(solution, 0)
    sim.config['render'] = False
    sim.config['save_data'] = False

# -------------------------------------- COMMENTS --------------------------------------------------

# - Make each supply period (e.g. day) a gene (as a numpy array)?
