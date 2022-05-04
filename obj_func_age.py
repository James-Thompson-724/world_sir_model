
import os
import pygame
import copy
import csv
import random
import numpy as np
import shapefile as shp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from collections import defaultdict

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

class Region:
    def __init__(self, iso):
        self.id = None
        self.iso = iso
        self.multipolygon = None
        self.list_of_points = None
        self.colour = None
        self.population_size = None
        self.age_distribution = None
        self.vaccine_hesitancy = None

class Simulator:
    def __init__(self, config):
        self.config = config
        self.regions = None
        self.number_of_regions = None
        self.number_of_age_groups = None
        self.number_of_strains = None
        self.contact_matrix = None
        self.population_sizes_by_age_group = None
        self.initial_cases = None
        self.ifr = None
        self.scalar_cmap = None
        self.vaccine_hesitant = None

def initialize_regions(regions_data_path):
    """Initialize regions"""

    # Initialize regions using population data
    regions = []
    with open(regions_data_path, newline='') as csvfile:
        next(csvfile)
        region_data = csv.reader(csvfile, delimiter=',')
        for row in region_data:
            iso = str(row[1])
            if iso == 'GB':
                iso = 'UK'
            new_region = Region(iso)
            regions.append(new_region)

    # Determine an integer identifier for each region
    for region in regions:
        region.id = regions.index(region)

    return regions

def add_population_data(regions, regions_data_path):
    """Add population data to regions"""

    # Add population data to regions
    population_data_dict = defaultdict(dict)
    with open(regions_data_path, newline='') as csvfile:
        next(csvfile)
        region_data = csv.reader(csvfile, delimiter=',')
        for row in region_data:
            iso = str(row[1])
            if iso == 'GB':
                iso = 'UK'
            population_size = int(row[2])
            vaccine_hesitancy = float(row[3])
            age_distribution = [float(row[4 + r]) for r in range(101)]
            population_data_dict[iso]['population_size'] = population_size
            population_data_dict[iso]['age_distribution'] = age_distribution
            population_data_dict[iso]['vaccine_hesitancy'] = vaccine_hesitancy
    for region in regions:
        iso = region.iso
        region.population_size = population_data_dict[iso]['population_size']
        region.age_distribution = population_data_dict[iso]['age_distribution']
        region.vaccine_hesitancy = population_data_dict[iso]['vaccine_hesitancy']

def add_shape_data(regions, regions_shape_path, points_per_polygon,
                   display_width, display_height):
    """Add polgonal shapes to regions for rendering"""

    sf = shp.Reader(regions_shape_path)
    shape_recs = sf.shapeRecords()

    # Get minimum and maxiumum coordinates
    min_x = min([pair[0] for shape in shape_recs for pair in shape.shape.points[:]])
    min_y = min([pair[1] for shape in shape_recs for pair in shape.shape.points[:]])
    max_x = max([pair[0] for shape in shape_recs for pair in shape.shape.points[:]])
    max_y = max([pair[1] for shape in shape_recs for pair in shape.shape.points[:]])

    # Determine for which regions coordinate data can be found
    regions_to_coordinates = {}
    for region in regions:
        regions_to_coordinates[region] = None
        for id in range(len(shape_recs)):
            iso = shape_recs[id].record[0]
            if region.iso == iso:
                # Extract coordinates according to shape type
                geoj = sf.shape(id).__geo_interface__
                if geoj["type"] == "Polygon":
                    coordinates = [geoj["coordinates"]]
                elif geoj["type"] == "MultiPolygon":
                    coordinates = geoj["coordinates"]
                else:
                    coordinates = None
                regions_to_coordinates[region] = coordinates

    # Assign polygonal shape to regions for rendering using coordinate data
    for region in regions:
        if regions_to_coordinates[region] is not None:
            region.colour = "white"
            coordinates = regions_to_coordinates[region]
            # Transform coordinates and create new polygon
            list_of_points = []
            for part in coordinates:
                points = part[0]
                # Transform coordinates to display size
                x = [int(display_width * (i[0] - min_x) / (max_x - min_x)) for i in points[:]]
                y = [int(display_height * (1 - ((i[1] - min_y) / (max_y - min_y))))
                     for i in points[:]]
                # Put coordinates into a list of tuples
                new_points = list(zip(x,y))
                # Subsample
                new_points = [new_points[i] for i in sorted(random.sample(range(len(new_points)),
                              min(len(new_points), points_per_polygon)))]
                # Add to list of points
                list_of_points.append(new_points)
            # Update region with shape data
            region.multipolygon = MultiPolygon([Polygon(points) for points in list_of_points])
            region.list_of_points = list_of_points

def verify_matrix(matrix):
    """Checks that a given numpy matrix is a valid probability matrix"""

    num_rows = np.shape(matrix)[0]
    num_columns = np.shape(matrix)[1]

    for i in range(num_rows):
        try:
            assert sum(matrix[i]) == 1.0
        except AssertionError:
            print('Invalid matrix row sum: ', sum(matrix[i]))
        for j in range(num_columns):
            try:
                assert matrix[i][j] <= 1.0 and matrix[i][j] >= 0.0
            except AssertionError:
                print('Invalid matrix entry: ', matrix[i][j])

def get_mixing_prob_matrix(regions, airport_path, air_travel_path,
                           local_travel_prob_per_day, distance_threshold):
    """Constructs stochastic matrix of mixing between regions. The matrix travel_prob_matrix will
    be returned by this function. It will be constructed via a combination of air_travel_prob_matrix
    and local_travel_prob_matrix, defined below. Since the air travel data used for these
    simulations also records the month of travel, we additionally calculate
    air_travel_prob_matrix_by_month, for later use."""

    num_of_regions = len(regions)
    months_in_year = 12

    # This will be the matrix returned by the function
    travel_prob_matrix = np.zeros((num_of_regions, num_of_regions), dtype=float)

    # It will be constructed by obtaining air travel with local travel, the latter meaning travel
    # to neighbouring regions
    air_travel_prob_matrix = np.zeros((num_of_regions, num_of_regions), dtype=float)
    air_travel_prob_matrix_by_month =\
        np.zeros((months_in_year, num_of_regions, num_of_regions), dtype=float)
    local_travel_prob_matrix = np.zeros((num_of_regions, num_of_regions), dtype=float)

    # Map airports to region isos
    airports_to_region_iso = {}
    with open(airport_path, newline='') as csvfile:
        airport_data = csv.reader(csvfile, delimiter=',')
        next(airport_data, None)
        for row in airport_data:
            airport = str(row[0])
            region_iso = str(row[3])
            airports_to_region_iso[airport] = region_iso

    # Get air travel matrix, recording the number of travellers between regions per month or year
    region_isos = [region.iso for region in regions]
    region_isos_to_ids = {region.iso: region.id for region in regions}
    air_travel_matrix_by_month =\
        np.zeros((months_in_year, num_of_regions, num_of_regions), dtype=int)
    air_travel_matrix = np.zeros((num_of_regions, num_of_regions), dtype=int)
    with open(air_travel_path, newline='') as csvfile:
        travel_data = csv.reader(csvfile, delimiter=',')
        next(travel_data, None)
        for row in travel_data:
            origin_airport = str(row[0])
            destination_airport = str(row[1])
            origin_region_iso = airports_to_region_iso[origin_airport]
            destination_region_iso = airports_to_region_iso[destination_airport]
            if (origin_region_iso in region_isos) and (destination_region_iso in region_isos):
                origin_region_id = region_isos_to_ids[origin_region_iso]
                destination_region_id = region_isos_to_ids[destination_region_iso]
                # Discard internal flights
                if origin_region_id != destination_region_id:
                    month = int(row[2]) - 1
                    prediction = int(row[3])
                    air_travel_matrix_by_month[month][origin_region_id][destination_region_id] +=\
                        prediction
                    air_travel_matrix[origin_region_id][destination_region_id] += prediction

    # Calculate air travel probabilities, rescaled from months or years to days
    days_in_year_2010 = 365
    days_in_month_2010 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ids_to_population_sizes = {region.id: region.population_size for region in regions}
    for month in range(months_in_year):
        for i in range(num_of_regions):
            for j in range(num_of_regions):
                air_travel_per_day =\
                    air_travel_matrix_by_month[month][i][j] / days_in_month_2010[month]
                air_travel_prob_matrix_by_month[month][i][j] =\
                    air_travel_per_day / ids_to_population_sizes[i]
    for i in range(num_of_regions):
        for j in range(num_of_regions):
            air_travel_per_day = air_travel_matrix[i][j] / days_in_year_2010
            air_travel_prob_matrix[i][j] = air_travel_per_day / ids_to_population_sizes[i]

    # Get adjacency matrix, recording which regions border which others
    adjacency_matrix = np.zeros((num_of_regions, num_of_regions), dtype=int)
    for region in regions:
        for other_region in regions:
            if (region.multipolygon is not None) and (other_region.multipolygon is not None):
                if region.multipolygon.distance(other_region.multipolygon) < distance_threshold:
                    adjacency_matrix[region.id][other_region.id] = 1
    np.fill_diagonal(adjacency_matrix, 0)

    # Calculate local travel probabilities
    share_matrix = np.zeros((num_of_regions, num_of_regions), dtype=float)
    for region in regions:
        for other_region in regions:
            if adjacency_matrix[region.id][other_region.id] == 1:
                share_matrix[region.id][other_region.id] = other_region.population_size
    for i in range(num_of_regions):
        row_sum = sum(share_matrix[i])
        for j in range(num_of_regions):
            if share_matrix[i][j] > 0:
                share_matrix[i][j] = share_matrix[i][j] / row_sum
    for i in range(num_of_regions):
        for j in range(num_of_regions):
            local_travel_prob_matrix[i][j] = share_matrix[i][j] * local_travel_prob_per_day

    # Combine air travel and local travel probability matrices to get final travel probabilities
    for i in range(num_of_regions):
        for j in range(num_of_regions):
            travel_prob_matrix[i][j] = 1 - ((1 - local_travel_prob_matrix[i][j]) *\
                                            (1 - air_travel_prob_matrix[i][j]))
    for i in range(num_of_regions):
        row_sum = sum(travel_prob_matrix[i])
        travel_prob_matrix[i][i] = 1 - row_sum

    # Check that the resulting matrix is a valid probability matrix
    verify_matrix(travel_prob_matrix)

    return travel_prob_matrix

def get_contact_matrix(regions, mixing_prob_matrix, contacts_per_day):
    """The number of contacts on average per day between regions"""

    baseline_contact_matrix = np.zeros((len(regions), len(regions)), dtype=float)
    contact_vector = np.full((len(regions)), contacts_per_day, dtype=int)
    for i in range(len(regions)):
        for j in range(len(regions)):
            baseline_contact_matrix[i][j] = mixing_prob_matrix[i][j] * contact_vector[j]

    return baseline_contact_matrix

def get_scalar_cmap(cmap, min_val, max_val):
    """Constucts scalar colour map to represent disease prevalence when rendering"""

    cm = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=min_val, vmax=max_val)

    return cmx.ScalarMappable(norm=cNorm, cmap=cm)

def set_initial_cases(regions, number_of_regions, number_of_age_groups, number_of_strains,
                      initial_cases_dict, population_sizes_by_age_group, age_dist):
    """Sets initial case numbers for each region"""

    initial_cases =\
        np.zeros((number_of_regions, number_of_age_groups, number_of_strains), dtype=float)
    if initial_cases_dict is None:
        for id in len(regions):
            for a in range(number_of_age_groups):
                for s in range(number_of_strains):
                    initial_cases[id][a][s] = 1
    elif isinstance(initial_cases_dict, list):
        for id in range(len(regions)):
            share_of_pop = np.sum(population_sizes_by_age_group[id])\
                           / np.sum(population_sizes_by_age_group)
            for a in range(number_of_age_groups):
                share_by_age = age_dist[id][a]
                for s in range(number_of_strains):
                    num = initial_cases_dict[s] * share_of_pop * share_by_age
                    initial_cases[id][a][s] = num
    else:
        regions_with_initial_cases = list(initial_cases_dict.keys())
        for region in regions:
            iso = region.iso
            if iso in regions_with_initial_cases:
                for a in range(number_of_age_groups):
                    share_by_age = age_dist[region.id][a]
                    for s in range(number_of_strains):
                        initial_cases[region.id][a][s] = initial_cases_dict[iso][s] * share_by_age

    return initial_cases

def set_ifr(regions, number_of_age_groups, number_of_strains, ifr_full):
    """Sets infection fatality ratios for the relavent age groups"""

    ifr = np.zeros((number_of_age_groups, number_of_strains), dtype=float)
    ifr_full = np.array(ifr_full, dtype=float)
    world_pop_by_age = np.zeros(101, dtype=float)
    for a in range(101):
        for region in regions:
            world_pop_by_age[a] += region.population_size * region.age_distribution[a]
    world_age_distribution = world_pop_by_age / np.sum(world_pop_by_age)
    world_age_distribution_0 =\
        world_age_distribution[0:18] / np.sum(world_age_distribution[0:18])
    world_age_distribution_1 =\
        world_age_distribution[18:65] / np.sum(world_age_distribution[18:65])
    world_age_distribution_2 =\
        world_age_distribution[65:] / np.sum(world_age_distribution[65:])
    for s in range(number_of_strains):
        ifr[0][s] = np.dot(ifr_full[s][0:18], world_age_distribution_0)
        ifr[1][s] = np.dot(ifr_full[s][18:65], world_age_distribution_1)
        ifr[2][s] = np.dot(ifr_full[s][65:], world_age_distribution_2)

    return ifr

def build_age_dist(regions, number_of_regions, number_of_age_groups):
    """Compresses full age distributions according to age groups"""

    age_dist = np.zeros((number_of_regions, number_of_age_groups), dtype=float)
    for region in regions:
        for a in range(101):
            if a < 18:
                age_dist[region.id][0] += region.age_distribution[a]
            if a >= 18 and a < 65:
                age_dist[region.id][1] += region.age_distribution[a]
            if a >= 65:
                age_dist[region.id][2] += region.age_distribution[a]
    age_dist = np.transpose(np.transpose(age_dist) / age_dist.sum(axis=1))

    return age_dist

def get_age_group_sizes(regions, number_of_regions, number_of_age_groups, age_dist):
    """Calculates total population in each age group for each region"""

    population_sizes_by_age_group =\
        np.zeros((number_of_regions, number_of_age_groups), dtype=np.uint64)
    for region in regions:
        for a in range(number_of_age_groups):
            population_sizes_by_age_group[region.id][a] =\
                max(int(region.population_size * age_dist[region.id][a]), 1)

    return population_sizes_by_age_group

def get_vaccine_hesitant(regions, number_of_regions, number_of_age_groups, age_dist):
    """Calculates total hesitant population in each age group for each region"""

    vaccine_hesitant =\
        np.zeros((number_of_regions, number_of_age_groups), dtype=np.uint64)
    for region in regions:
        for a in range(number_of_age_groups):
            vaccine_hesitant[region.id][a] =\
                int(region.vaccine_hesitancy * region.population_size * age_dist[region.id][a])

    return vaccine_hesitant

def draw_regions(lockdown_status, border_closure_status, prevalence, day, infected, cases, label,
                 deaths, total_doses_administered, screen, font_size, scalar_cmap, regions):
    """Draws regions as polygons, colouring the interior according to prevalence and the border
    according to lockdown and border closure status"""

    # Refresh screen
    screen.fill("white")
    # Colour regions by prevalence
    for region in regions:
        colour = scalar_cmap.to_rgba(prevalence[region], bytes = True)
        if region.list_of_points is not None:
            for points in region.list_of_points:
                pygame.draw.polygon(screen, colour, points, width=0)
    # Colour borders according to border closure status
    for region in regions:
        id = region.id
        if border_closure_status[id] == 1:
            if region.list_of_points is not None:
                for points in region.list_of_points:
                    pygame.draw.polygon(screen, "gold", points, width=4)
    # Colour borders according to lockdown status
    for region in regions:
        id = region.id
        if lockdown_status[id] == 1:
            if region.list_of_points is not None:
                for points in region.list_of_points:
                    pygame.draw.polygon(screen, "black", points, width=2)
    # Print day, cases, deaths and vaccinations
    BLACK = (0, 0, 0)
    FONT = pygame.font.Font("freesansbold.ttf", font_size)

    surface = pygame.display.get_surface()

    text_surf = FONT.render(label, True, BLACK)
    text_rect = text_surf.get_rect(topleft=(0, surface.get_height() - (6 * font_size)))
    screen.blit(text_surf, text_rect)

    text_surf = FONT.render("Day: " + str(day), True, BLACK)
    text_rect = text_surf.get_rect(topleft=(0, surface.get_height() - (5 * font_size)))
    screen.blit(text_surf, text_rect)

    text_surf = FONT.render("Infected: " + str(infected), True, BLACK)
    text_rect = text_surf.get_rect(topleft=(0, surface.get_height() - (4 * font_size)))
    screen.blit(text_surf, text_rect)

    text_surf = FONT.render("Cases: " + str(cases), True, BLACK)
    text_rect = text_surf.get_rect(topleft=(0, surface.get_height() - (3 * font_size)))
    screen.blit(text_surf, text_rect)

    text_surf = FONT.render("Deaths: " + str(deaths), True, BLACK)
    text_rect = text_surf.get_rect(topleft=(0, surface.get_height() - (2 * font_size)))
    screen.blit(text_surf, text_rect)

    text_surf = FONT.render("Vaccinations: " + str(total_doses_administered), True, BLACK)
    text_rect = text_surf.get_rect(topleft=(0, surface.get_height() - font_size))
    screen.blit(text_surf, text_rect)

    # Update display
    pygame.display.update()

def sim_factory(config):
    """Constructs the simulator object"""

    display_width             = config['display_width']
    display_height            = config['display_height']
    points_per_polygon        = config['points_per_polygon']
    infection_cmap            = config['infection_cmap']
    vaccination_cmap          = config['vaccination_cmap']
    vaccination_0_cmap        = config['vaccination_0_cmap']
    vaccination_1_cmap        = config['vaccination_1_cmap']
    vaccination_2_cmap        = config['vaccination_2_cmap']
    travel_enabled            = config['international_travel_enabled']
    distance_threshold        = config['distance_threshold']
    local_travel_prob_per_day = config['local_travel_prob_per_day']
    contacts_per_day          = config['contacts_per_day']
    shp_path                  = config['shp_path']
    pop_path                  = config['pop_path']
    airport_path              = config['airport_path']
    air_travel_path           = config['air_travel_path']
    initial_cases_dict        = config['initial_cases_dict']
    beta                      = config['transmission_probabilities']
    ifr                       = config['infection_fatality_rate_by_age']

    sim = Simulator(config)

    # Initialize regions
    sim.regions = initialize_regions(pop_path)

    # Add population data to regions
    add_population_data(sim.regions, pop_path)

    # Add shape data to regions
    add_shape_data(sim.regions, shp_path, points_per_polygon, display_width, display_height)

    # Build stochastic matrix of mixing between regions
    if travel_enabled:
        mixing_prob_matrix = get_mixing_prob_matrix(sim.regions, airport_path, air_travel_path,
                                                    local_travel_prob_per_day, distance_threshold)
    else:
        mixing_prob_matrix = np.identity(len(sim.regions), dtype=float)

    # Build contact matrix
    sim.contact_matrix = get_contact_matrix(sim.regions, mixing_prob_matrix, contacts_per_day)

    # Build colour map for rendering
    sim.infection_cmap = get_scalar_cmap(infection_cmap, 0, 1)
    sim.vaccination_cmap = get_scalar_cmap(vaccination_cmap, 0, 1)
    sim.vaccination_0_cmap = get_scalar_cmap(vaccination_0_cmap, 0, 1)
    sim.vaccination_1_cmap = get_scalar_cmap(vaccination_1_cmap, 0, 1)
    sim.vaccination_2_cmap = get_scalar_cmap(vaccination_2_cmap, 0, 1)

    # Numbers of regions and strains
    sim.number_of_regions = len(sim.regions)
    sim.number_of_strains = len(beta)

    # Build age structure
    sim.number_of_age_groups = 3
    age_dist =\
        build_age_dist(sim.regions, sim.number_of_regions, sim.number_of_age_groups)
    sim.population_sizes_by_age_group =\
        get_age_group_sizes(sim.regions, sim.number_of_regions, sim.number_of_age_groups, age_dist)
    sim.vaccine_hesitant =\
        get_vaccine_hesitant(sim.regions, sim.number_of_regions, sim.number_of_age_groups, age_dist)

    # Set initial cases
    sim.initial_cases =\
        set_initial_cases(sim.regions, sim.number_of_regions, sim.number_of_age_groups,
                          sim.number_of_strains, initial_cases_dict,
                          sim.population_sizes_by_age_group, age_dist)

    # Set infection fatality ratios
    sim.ifr = set_ifr(sim.regions, sim.number_of_age_groups, sim.number_of_strains, ifr)

    return sim

def run(config, sim, lockdown_input, border_closure_input, vaccination_input):
    """Main simulation function"""

    render                = config['render']
    save_data             = config['save_data']
    save_screeshot        = config['save_screeshot']
    screenshot_filename   = config['screenshot_filename']
    display_width         = config['display_width']
    display_height        = config['display_height']
    font_size             = config['font_size']
    max_norm_prev         = config['max_norm_prevalance_to_plot']
    refresh_rate          = config['refresh_rate']
    time_horizon_days     = config['time_horizon_days']
    step_size             = config['euler_scheme_step_size_days']
    beta                  = config['transmission_probabilities']
    gamma                 = config['removal_rates_per_day']
    lockdown_factor       = config['lockdown_factor']
    border_closure_factor = config['border_closure_factor']

    regions                 = sim.regions
    baseline_contact_matrix = sim.contact_matrix
    infection_cmap          = sim.infection_cmap
    vaccination_cmap        = sim.vaccination_cmap
    vaccination_0_cmap      = sim.vaccination_0_cmap
    vaccination_1_cmap      = sim.vaccination_1_cmap
    vaccination_2_cmap      = sim.vaccination_2_cmap
    number_of_regions       = sim.number_of_regions
    number_of_strains       = sim.number_of_strains
    population_sizes        = sim.population_sizes_by_age_group
    vaccine_hesitant        = sim.vaccine_hesitant
    initial_cases           = sim.initial_cases
    number_of_age_groups    = sim.number_of_age_groups
    ifr                     = sim.ifr

    # Compartments (S: Susceptible and not vaccine hesitant,
    #               H: Susceptible and vaccine hesitant,
    #               I: Infected,
    #               R: Recovered,
    #               D: Dead)
    T = int(time_horizon_days * (1 / step_size))
    N = np.zeros((number_of_regions, number_of_age_groups), dtype=np.float64)
    S = np.zeros((T, number_of_regions, number_of_age_groups), dtype=np.float64)
    H = np.zeros((T, number_of_regions, number_of_age_groups), dtype=np.float64)
    I = np.zeros((T, number_of_regions, number_of_age_groups, number_of_strains), dtype=np.float64)
    R = np.zeros((T, number_of_regions, number_of_age_groups), dtype=np.float64)
    D = np.zeros((T, number_of_regions, number_of_age_groups), dtype=np.float64)

    # Initial conditions
    for k in range(number_of_regions):
        for a in range(number_of_age_groups):
            N[k][a] = population_sizes[k][a]
            H[0][k][a] = vaccine_hesitant[k][a]
            R[0][k][a] = 0
            D[0][k][a] = 0
            for i in range(number_of_strains):
                I[0][k][a][i] = initial_cases[k][a][i]
            S[0][k][a] = N[k][a] - H[0][k][a] - R[0][k][a] -\
                         sum([I[0][k][a][i] for i in range(number_of_strains)])

    # Initial interventions
    lockdown_status       = np.full((number_of_regions), -1, dtype=int)
    border_closure_status = np.full((number_of_regions), -1, dtype=int)

    if render:
        pygame.init()
        screen = pygame.display.set_mode((display_width, display_height + font_size))
        clock  = pygame.time.Clock()

    # Precalculate N_bar
    N_bar = np.zeros((number_of_regions), dtype=float)
    for j in range(number_of_regions):
        N_bar[j] = 1 / np.sum(N[j])

    if save_data:
        doses_administered = np.zeros((T, number_of_regions, number_of_age_groups), dtype=int)
    total_doses_administered = np.zeros((number_of_regions, number_of_age_groups), dtype=np.uint64)

    t = 0
    display_data = 0
    color_map = infection_cmap
    done = False
    while not done:

        steps_in_a_day = int(1 / step_size)

        day = t // steps_in_a_day
        # Render
        if render:
            # Calculate deaths
            infected  = round(np.sum(I[t]))
            deaths = round(np.sum(D[t]))
            cases = round(np.sum(D[t])) + deaths
            # Mouse input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        display_data = (display_data + 1) % 5
                    if event.button == 3:
                        display_data = (display_data - 1) % 5
            # Calculate prevalence
            prevalence = {}
            max_vac_cov = 1.0
            for region in regions:
                if display_data == 0:
                    color_map = infection_cmap
                    label = 'Prevalence (all ages)'
                    normalized_prevalence = np.sum(I[t][region.id]) / np.sum(N[region.id])
                    prevalence[region] = min(normalized_prevalence, max_norm_prev) / max_norm_prev
                if display_data == 1:
                    color_map = vaccination_cmap
                    label = 'Vaccine coverage (all ages)'
                    coverage = np.sum(total_doses_administered[region.id]) / np.sum(N[region.id])
                    prevalence[region] = min(coverage, max_vac_cov) / max_vac_cov
                if display_data == 2:
                    color_map = vaccination_0_cmap
                    label = 'Vaccine coverage (age 0-17)'
                    coverage = total_doses_administered[region.id][0] / N[region.id][0]
                    prevalence[region] = min(coverage, max_vac_cov) / max_vac_cov
                if display_data == 3:
                    color_map = vaccination_1_cmap
                    label = 'Vaccine coverage (age 18-64)'
                    coverage = total_doses_administered[region.id][1] / N[region.id][1]
                    prevalence[region] = min(coverage, max_vac_cov) / max_vac_cov
                if display_data == 4:
                    color_map = vaccination_2_cmap
                    label = 'Vaccine coverage (age 65+)'
                    coverage = total_doses_administered[region.id][2] / N[region.id][2]
                    prevalence[region] = min(coverage, max_vac_cov) / max_vac_cov
            # Draw regions
            draw_regions(lockdown_status, border_closure_status, prevalence, day, infected, cases,
                         label, deaths, np.sum(total_doses_administered), screen, font_size,
                         color_map, regions)

        # Update non-pharmacheutical interventions at the end of each day
        if t % steps_in_a_day == 0:
            lockdown_status = lockdown_input[day]
            border_closure_status = border_closure_input[day]

        # Update contact matrix according to non-pharmacheutical interventions
        contact_matrix = copy.deepcopy(baseline_contact_matrix)
        vec = (((1 + border_closure_status) / 2) * border_closure_factor) +\
              (1 - ((1 + border_closure_status) / 2))
        contact_matrix = np.transpose(np.multiply(vec, np.transpose(contact_matrix)))
        baseline_contact_matrix_diag = np.diagonal(baseline_contact_matrix)
        new_diag = np.multiply(baseline_contact_matrix_diag * lockdown_factor,
                               (1 + lockdown_status) / 2) +\
                   np.multiply(baseline_contact_matrix_diag, 1 - ((1 + lockdown_status) / 2))
        np.fill_diagonal(contact_matrix, new_diag)

        # Simulate transmission
        if t < T - 1:

            # Vaccination
            new_doses_administered = np.minimum(S[t], vaccination_input[day] / steps_in_a_day)
            if save_data:
                doses_administered[t] = new_doses_administered
            S[t] = S[t] - new_doses_administered
            R[t] = R[t] + new_doses_administered
            total_doses_administered =\
                (total_doses_administered + new_doses_administered).astype(np.uint64)

            # Update
            A = np.matmul(contact_matrix, np.multiply(np.sum(I[t], axis=1), N_bar[:, None]))
            ones = np.ones((number_of_regions, number_of_age_groups, number_of_strains))

            S[t+1] = S[t] - step_size*np.multiply(np.matmul(A, beta)[:, None], S[t])
            H[t+1] = H[t] - step_size*np.multiply(np.matmul(A, beta)[:, None], H[t])
            I[t+1] = I[t] + step_size*np.multiply(np.multiply(A, beta[None, :])[:, None, :],
                                                  (S[t] + H[t])[:, :, None])\
                          - step_size*np.multiply(I[t], gamma)
            R[t+1] = R[t] + step_size*np.matmul(np.multiply(I[t], ones - ifr[None, :, :]), gamma)
            D[t+1] = D[t] + step_size*np.matmul(np.multiply(I[t], ifr[None, :, :]), gamma)

        # Update step
        t += 1
        if t == T - 1:
            done = True

        if render:
            clock.tick(refresh_rate)

    if render:
        if save_screeshot:
            pygame.image.save(screen, screenshot_filename)
        pygame.quit()

    if save_data:
        np.save("doses_administered.npy", doses_administered)
        np.save("population_sizes_by_age.npy", N)

    total_deaths = round(np.sum(D[t]))
    total_cases = round(np.sum(R[t])) + total_deaths

    return total_deaths # total_cases
