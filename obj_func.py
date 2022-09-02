
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
        self.number_of_strains = None
        self.contact_matrix = None
        self.population_sizes = None
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
    cNorm  = colors.Normalize(vmin=min_val, vmax=max_val)

    return cmx.ScalarMappable(norm=cNorm, cmap=cm)

def set_initial_cases(regions, number_of_regions, number_of_strains,
                      initial_cases_dict, population_sizes):
    """Sets initial case numbers for each region"""

    initial_cases = np.zeros((number_of_regions, number_of_strains), dtype=float)
    if initial_cases_dict is None:
        for id in len(regions):
            for s in range(number_of_strains):
                initial_cases[id][s] = 1
    elif isinstance(initial_cases_dict, list):
        for id in range(len(regions)):
            for s in range(number_of_strains):
                initial_cases[id][s] = initial_cases_dict[s] *\
                                       (population_sizes[id] / np.sum(population_sizes))
    else:
        regions_with_initial_cases = list(initial_cases_dict.keys())
        for region in regions:
            iso = region.iso
            if iso in regions_with_initial_cases:
                for s in range(number_of_strains):
                    initial_cases[region.id][s] = initial_cases_dict[iso][s]

    return initial_cases

def set_ifr(regions, ifr_by_age):
    """Sets infection fatality rate for each region"""

    num_strains = len(ifr_by_age)
    ifr = np.zeros((len(regions), num_strains), dtype=float)
    for region in regions:
        for i in range(num_strains):
            ifr[region.id][i] = np.dot(region.age_distribution, ifr_by_age[i])

    return ifr

def draw_regions(lockdown_status, border_closure_status, prevalence, day, infected,
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

    text_surf = FONT.render("Day: " + str(day), True, BLACK)
    text_rect = text_surf.get_rect(topleft=(0, surface.get_height() - (4 * font_size)))
    screen.blit(text_surf, text_rect)

    text_surf = FONT.render("Infected: " + str(infected), True, BLACK)
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
    ifr_by_age                = config['infection_fatality_rate_by_age']

    sim = Simulator(config)

    # Initialize regions
    sim.regions = initialize_regions(pop_path)

    # Add population data to regions
    add_population_data(sim.regions, pop_path)

    # Add shape data to regions
    add_shape_data(sim.regions, shp_path, points_per_polygon, display_width, display_height)

    # Build mixing matrix between regions
    if travel_enabled:
        mixing_prob_matrix = get_mixing_prob_matrix(sim.regions, airport_path, air_travel_path,
                                                    local_travel_prob_per_day, distance_threshold)
    else:
        mixing_prob_matrix = np.identity(len(sim.regions), dtype=float)

    # Build vector of contacts within each region
    sim.contact_matrix = get_contact_matrix(sim.regions, mixing_prob_matrix, contacts_per_day)

    # Build colour map for rendering
    sim.infection_cmap = get_scalar_cmap(infection_cmap, 0, 1)
    sim.vaccination_cmap = get_scalar_cmap(vaccination_cmap, 0, 1)

    # Build vector objects
    sim.number_of_regions = len(sim.regions)
    sim.number_of_strains = len(beta)
    sim.population_sizes = np.zeros((sim.number_of_regions), dtype=np.uint64)
    sim.vaccine_hesitant = np.zeros((sim.number_of_regions), dtype=np.uint64)
    for region in sim.regions:
        sim.population_sizes[region.id] = region.population_size
        vaccine_hesitancy = region.vaccine_hesitancy
        sim.vaccine_hesitant[region.id] = int(vaccine_hesitancy * region.population_size)
    sim.initial_cases = set_initial_cases(sim.regions, sim.number_of_regions, sim.number_of_strains,
                                          initial_cases_dict, sim.population_sizes)
    sim.ifr = set_ifr(sim.regions, ifr_by_age)

    return sim

def run(config, sim, lockdown_input, border_closure_input, vaccination_input):
    """Main simulation function"""

    render                = config['render']
    save_screenshot        = config['save_screenshot']
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
    number_of_regions       = sim.number_of_regions
    number_of_strains       = sim.number_of_strains
    population_sizes        = sim.population_sizes
    vaccine_hesitant        = sim.vaccine_hesitant
    initial_cases           = sim.initial_cases
    ifr                     = sim.ifr

    # Compartments (S: Susceptible and not vaccine hesitant,
    #               H: Susceptible and vaccine hesitant,
    #               I: Infected,
    #               R: Recovered,
    #               D: Dead)
    T = int(time_horizon_days * (1/step_size))
    N = population_sizes
    S = np.zeros((T, number_of_regions), dtype=np.float64)
    H = np.zeros((T, number_of_regions), dtype=np.float64)
    I = np.zeros((T, number_of_regions, number_of_strains), dtype=np.float64)
    R = np.zeros((T, number_of_regions), dtype=np.float64)
    D = np.zeros((T, number_of_regions), dtype=np.float64)

    # Initial conditions
    for k in range(number_of_regions):
        H[0][k] = vaccine_hesitant[k]
        R[0][k] = 0
        D[0][k] = 0
        for i in range(number_of_strains):
            I[0][k][i] = initial_cases[k][i]
        S[0][k] = N[k] - H[0][k] - sum([I[0][k][i] for i in range(number_of_strains)]) - R[0][k]

    # Initial interventions
    lockdown_status = np.full((number_of_regions), -1, dtype=int)
    border_closure_status = np.full((number_of_regions), -1, dtype=int)

    # Precalculate N_bar
    N_bar = np.zeros((number_of_regions, number_of_strains), dtype=float)
    for k in range(number_of_regions):
        for i in range(number_of_strains):
            N_bar[k][i] = 1 / N[k]

    if render:
        pygame.init()
        screen = pygame.display.set_mode((display_width, display_height + font_size))
        clock  = pygame.time.Clock()

    t = 0
    display_data = -1
    color_map = infection_cmap
    total_doses_administered = np.zeros((number_of_regions), dtype=np.uint64)
    done = False
    while not done:

        steps_in_a_day = int(1 / step_size)

        day = t // steps_in_a_day
        # Render
        if render:
            # Calculate deaths
            infected  = round(np.sum(I[t]))
            deaths = round(np.sum(D[t]))
            # Mouse input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        display_data *= -1
            # Calculate prevalence
            prevalence = {}
            max_vac_cov = 1.0
            for region in regions:
                if display_data == 1:
                    color_map = infection_cmap
                    normalized_prevalence = sum(I[t][region.id])/N[region.id]
                    prevalence[region] = min(normalized_prevalence, max_norm_prev)/ max_norm_prev
                if display_data == -1:
                    color_map = vaccination_cmap
                    coverage = total_doses_administered[region.id]/N[region.id]
                    prevalence[region] = min(coverage, max_vac_cov)/ max_vac_cov
            # Draw regions
            draw_regions(lockdown_status, border_closure_status, prevalence, day, infected, deaths,
                         np.sum(total_doses_administered), screen, font_size, color_map, regions)

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
            doses_administered = np.minimum(S[t], vaccination_input[day] / steps_in_a_day)
            S[t] = S[t] - doses_administered
            R[t] = R[t] + doses_administered
            total_doses_administered =\
                (total_doses_administered + doses_administered).astype(np.uint64)

            # Transmission
            A = np.matmul(contact_matrix, np.multiply(I[t], N_bar))
            S[t+1] = S[t] - step_size*np.multiply(np.matmul(A, beta), S[t])
            H[t+1] = H[t] - step_size*np.multiply(np.matmul(A, beta), H[t])
            I[t+1] = I[t] + step_size*np.multiply(A, np.tensordot(S[t] + H[t], beta, axes=0))\
                          - step_size*np.multiply(I[t], gamma)
            R[t+1] = R[t] + step_size*np.matmul(np.multiply(I[t],
                                                np.ones(np.shape(I[t])) - ifr), gamma)
            D[t+1] = D[t] + step_size*np.matmul(np.multiply(I[t], ifr), gamma)

        # Update step
        t += 1
        if t == T - 1:
            done = True

        if render:
            clock.tick(refresh_rate)

    if render:
        if save_screenshot:
            pygame.image.save(screen, screenshot_filename)
        pygame.quit()

    total_deaths = round(sum(D[t]))

    return total_deaths
