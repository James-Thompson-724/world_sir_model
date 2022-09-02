
import numpy as np
import matplotlib.pyplot as plt
import csv

def get_un_regions():
    """Creates array indicating to which UN region each country belongs"""

    isos = []
    with open("data/compiled_data.csv", newline='') as csvfile:
        next(csvfile)
        region_data = csv.reader(csvfile, delimiter=',')
        for row in region_data:
            iso = str(row[1])
            if iso == 'GB':
                iso = 'UK'
            isos.append(iso)

    un_regions = []
    for iso in isos:
        with open("data/data_un_regions.csv", newline='') as csvfile:
            region_data = csv.reader(csvfile, delimiter=',')
            for row in region_data:
                if iso == str(row[1]) or (str(row[1]) == 'GB' and iso == 'UK'):
                    un_region = str(row[4])
                    if un_region == 'Asia-Pacific Group':
                        un_regions.append(0)
                    if un_region == 'WEOG':
                        un_regions.append(1)
                    if un_region == 'GRULAC':
                        un_regions.append(2)
                    if un_region == 'African Group':
                        un_regions.append(3)
                    if un_region == 'Eastern European Group':
                        un_regions.append(4)
    un_regions = np.array(un_regions, dtype=int)

    return un_regions

def get_population_by_un_region(un_regions, N):
    """Creates array of population size for each UN region and age group"""

    population_by_un_region = np.zeros((5, number_of_age_groups), dtype=np.uint64)
    for r in range(number_of_regions):
        un_region = un_regions[r]
        population_by_un_region[un_region] = population_by_un_region[un_region] + N[r]

    return population_by_un_region

def get_doses_by_un_region(un_regions):
    """Records doses administered in each UN region in each age group each tick"""

    doses_administered_by_un_region = np.zeros((5, number_of_age_groups, T), dtype=np.uint64)
    for r in range(number_of_regions):
        un_region = un_regions[r]
        doses_administered_by_un_region[un_region] =\
            doses_administered_by_un_region[un_region] + doses_administered[r]

    return doses_administered_by_un_region

def plot_stacked_bar_chart(doses_administered_by_un_region, x_lim):
    """Plots stacked bar chart of share of doses across UN regions and age groups each tick"""

    share_per_region = np.zeros((5, number_of_age_groups, T), dtype=float)
    totals = np.sum(np.sum(doses_administered_by_un_region, axis=0), axis=0)
    for un_region in range(5):
        for a in range(number_of_age_groups):
            for t in range(T):
                if totals[t] > 0:
                    share = doses_administered_by_un_region[un_region][a][t] # / totals[t]
                    share_per_region[un_region][a][t] = share
                else:
                    share_per_region[un_region][a][t] = 0

    x = list(range(x_lim))
    y00 = share_per_region[0][0][0:x_lim]
    y01 = share_per_region[0][1][0:x_lim]
    y02 = share_per_region[0][2][0:x_lim]
    y10 = share_per_region[1][0][0:x_lim]
    y11 = share_per_region[1][1][0:x_lim]
    y12 = share_per_region[1][2][0:x_lim]
    y20 = share_per_region[2][0][0:x_lim]
    y21 = share_per_region[2][1][0:x_lim]
    y22 = share_per_region[2][2][0:x_lim]
    y30 = share_per_region[3][0][0:x_lim]
    y31 = share_per_region[3][1][0:x_lim]
    y32 = share_per_region[3][2][0:x_lim]
    y40 = share_per_region[4][0][0:x_lim]
    y41 = share_per_region[4][1][0:x_lim]
    y42 = share_per_region[4][2][0:x_lim]

    plt.figure(figsize=(13, 6))
    plt.xlim(0, x_lim)
    # plt.ylim(0, 1.0)
    plt.bar(x, y00, width=1.0, alpha=0.33, color='green')
    plt.bar(x, y01, width=1.0, alpha=0.66, bottom=y00, color='green')
    plt.bar(x, y02, width=1.0, alpha=1.00, bottom=y00+y01, color='green')
    plt.bar(x, y10, width=1.0, alpha=0.33, bottom=y00+y01+y02, color='orange')
    plt.bar(x, y11, width=1.0, alpha=0.66, bottom=y00+y01+y02+y10, color='orange')
    plt.bar(x, y12, width=1.0, alpha=1.00, bottom=y00+y01+y02+y10+y11, color='orange')
    plt.bar(x, y20, width=1.0, alpha=0.33, bottom=y00+y01+y02+y10+y11+y12, color='purple')
    plt.bar(x, y21, width=1.0, alpha=0.66, bottom=y00+y01+y02+y10+y11+y12+y20, color='purple')
    plt.bar(x, y22, width=1.0, alpha=1.00, bottom=y00+y01+y02+y10+y11+y12+y20+y21, color='purple')
    plt.bar(x, y30, width=1.0, alpha=0.33, bottom=y00+y01+y02+y10+y11+y12+y20+y21+y22, color='blue')
    plt.bar(x, y31, width=1.0, alpha=0.66, bottom=y00+y01+y02+y10+y11+y12+y20+y21+y22+y30, color='blue')
    plt.bar(x, y32, width=1.0, alpha=1.00, bottom=y00+y01+y02+y10+y11+y12+y20+y21+y22+y30+y31, color='blue')
    plt.bar(x, y40, width=1.0, alpha=0.33, bottom=y00+y01+y02+y10+y11+y12+y20+y21+y22+y30+y31+y32, color='red')
    plt.bar(x, y41, width=1.0, alpha=0.66, bottom=y00+y01+y02+y10+y11+y12+y20+y21+y22+y30+y31+y32+y40, color='red')
    plt.bar(x, y42, width=1.0, alpha=1.00, bottom=y00+y01+y02+y10+y11+y12+y20+y21+y22+y30+y31+y32+y40+y41, color='red')
    plt.legend(["Asia-Pacific, 0-17",
                "Asia-Pacific, 18-64",
                "Asia-Pacific, 65+",
                "Western Europe and Others, 0-17",
                "Western Europe and Others, 18-64",
                "Western Europe and Others, 65+",
                "Latin America and Caribbean, 0-17",
                "Latin America and Caribbean, 18-64",
                "Latin America and Caribbean, 65+",
                "Africa, 0-17",
                "Africa, 18-64",
                "Africa, 65+",
                "Eastern Europe, 0-17",
                "Eastern Europe, 18-64",
                "Eastern Europe"], loc='upper right', prop={'size': 6})
    plt.title("Share of Doses")
    plt.xlabel('Day')
    plt.xticks(ticks=[a*28*12 for a in range((x_lim // (28*12)) + 1)],
                labels=[a*28 for a in range((x_lim // (28*12)) + 1)])
    plt.show()
    # plt.savefig("bar.png")

def plot_cumulative_coverage_by_un_region(cumulative_coverage_by_un_region, x_lim):
    """Plots cumulative_coverage_by_un_region"""

    plt.figure(figsize=(8, 6))
    font = {'size' : 12}
    plt.rc('font', **font)

    plt.xlim(0, x_lim)

    color_dict = {0: 'green', 1: 'orange', 2: 'purple', 3: 'blue', 4: 'red'}
    label_dict = {0: 'Asia-Pacific, ', 1: 'Western Europe and Others, ',
                  2: 'Latin America and Caribbean, ', 3: 'Africa, ', 4: 'Eastern Europe, '}
    for un_region in range(5):
        print(label_dict[un_region] + "0-17:", cumulative_coverage_by_un_region[un_region][0][0])
        print(label_dict[un_region] + "18-64:", cumulative_coverage_by_un_region[un_region][1][0])
        print(label_dict[un_region] + "65+:", cumulative_coverage_by_un_region[un_region][2][0])
        color = color_dict[un_region]
        plt.plot(list(range(x_lim)), cumulative_coverage_by_un_region[un_region][0][0:x_lim], color,
            linewidth=1, linestyle='solid', alpha=1.0, label= label_dict[un_region] + " 0-17")
        plt.plot(list(range(x_lim)), cumulative_coverage_by_un_region[un_region][1][0:x_lim], color,
            linewidth=1, linestyle='dotted', alpha=1.0, label= label_dict[un_region] + " 18-64")
        plt.plot(list(range(x_lim)), cumulative_coverage_by_un_region[un_region][2][0:x_lim], color,
            linewidth=1, linestyle='dashed', alpha=1.0, label= label_dict[un_region] + " 65+")

    plt.ylabel('Share Vaccinated')
    plt.xlabel('Day')
    plt.xticks(ticks=[a*28*12 for a in range((x_lim // (28*12)) + 1)],
                labels=[a*28 for a in range((x_lim // (28*12)) + 1)])

    plt.legend(loc='upper right', prop={'size': 10})

    plt.grid(False)
    plt.show()
    # plt.savefig("cumulative.png")

def plot_coverage(doses_administered_by_un_region, x_lim):
    """Plots proportion of the world population vaccinated each tick"""

    plt.figure(figsize=(8, 6))
    font = {'size' : 12}
    plt.rc('font', **font)
    x_lim = int(T/2)
    plt.xlim(0, x_lim)

    coverage = np.sum(doses_administered_by_un_region, axis=0)
    cov = np.sum(coverage, axis=0) / np.sum(population_by_un_region)
    plt.plot(list(range(x_lim)), cov[0:x_lim], 'red', linewidth=1,
             linestyle='solid', alpha=1.0, label= "all")

    plt.ylabel('Share Vaccinated')
    plt.xlabel('Day')
    plt.xticks(ticks=[a*28*12 for a in range((x_lim // (28*12)) + 1)],
                labels=[a*28 for a in range((x_lim // (28*12)) + 1)])

    plt.legend(loc='upper right', prop={'size': 10})

    plt.grid(False)
    plt.show()

def plot_pie_chart(cumulative_doses_administered_by_un_region):
    """Plots pie chart of total vaccine allocation to each region and age group"""

    y = cumulative_doses_administered_by_un_region.flatten()

    plt.figure(figsize=(14, 8))

    unlabels = ["Asia-Pacific, 0-17",
                "Asia-Pacific, 18-64",
                "Asia-Pacific, 65+",
                "Western Europe and Others, 0-17",
                "Western Europe and Others, 18-64",
                "Western Europe and Others, 65+",
                "Latin America and Caribbean, 0-17",
                "Latin America and Caribbean, 18-64",
                "Latin America and Caribbean, 65+",
                "Africa, 0-17",
                "Africa, 18-64",
                "Africa, 65+",
                "Eastern Europe, 0-17",
                "Eastern Europe, 18-64",
                "Eastern Europe"]

    uncolors = [(0.0, 0.5019607843137255, 0.0, 0.33),
                (0.0, 0.5019607843137255, 0.0, 0.66),
                (0.0, 0.5019607843137255, 0.0, 1.0),
                (1.0, 0.6470588235294118, 0.0, 0.33),
                (1.0, 0.6470588235294118, 0.0, 0.66),
                (1.0, 0.6470588235294118, 0.0, 1.0),
                (0.5019607843137255, 0.0, 0.5019607843137255, 0.33),
                (0.5019607843137255, 0.0, 0.5019607843137255, 0.66),
                (0.5019607843137255, 0.0, 0.5019607843137255, 1.0),
                (0.0, 0.0, 1.0, 0.33),
                (0.0, 0.0, 1.0, 0.66),
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 0.0, 0.0, 0.33),
                (1.0, 0.0, 0.0, 0.66),
                (1.0, 0.0, 0.0, 1.0)]

    plt.pie(y, labels = None, colors = uncolors)
    plt.legend(unlabels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    # plt.show()
    plt.savefig("piechart.png")

def plot_pie_chart_age_compressed(cumulative_doses_administered_by_un_region):
    """Plots pie chart of total vaccine allocation to each region and age group"""

    y = cumulative_doses_administered_by_un_region.flatten()

    plt.figure(figsize=(14, 8))

    unlabels = ["Asia-Pacific",
                "Western Europe and Others",
                "Latin America and Caribbean",
                "Africa",
                "Eastern Europe"]

    uncolors = [(0.0, 0.5019607843137255, 0.0, 1.0),
                (1.0, 0.6470588235294118, 0.0, 1.0),
                (0.5019607843137255, 0.0, 0.5019607843137255, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 0.0, 0.0, 1.0)]

    plt.pie(y, labels = None, colors = uncolors)
    plt.legend(unlabels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    # plt.show()
    plt.savefig("piechart_age_compressed.png")

doses_administered = np.load("fitness_func_day_zero_deaths_1b/doses_administered.npy")
N = np.load("fitness_func_day_zero_deaths_1b/population_sizes_by_age.npy")

T = doses_administered.shape[0]
number_of_regions = doses_administered.shape[1]
number_of_age_groups = doses_administered.shape[2]

doses_administered = np.moveaxis(doses_administered, [0], [2])
cumulative_doses_administered = np.cumsum(doses_administered, axis=2)

un_regions = get_un_regions()

population_by_un_region = get_population_by_un_region(un_regions, N)

doses_administered_by_un_region = get_doses_by_un_region(un_regions)

cumulative_doses_administered_by_un_region = np.cumsum(doses_administered_by_un_region, axis=2)

coverage_by_un_region = np.divide(doses_administered_by_un_region,
                                  population_by_un_region[:, :, None])

cumulative_coverage_by_un_region = np.divide(cumulative_doses_administered_by_un_region,
                                             population_by_un_region[:, :, None])

coverage_by_un_region_age_compressed = np.sum(np.sum(doses_administered_by_un_region, axis=2), axis=1) /\
                                                 np.sum(population_by_un_region, axis=1)

# x_lim = int(T/6)

# plot_pie_chart(np.sum(doses_administered_by_un_region, axis=2))

# plot_pie_chart_age_compressed(np.sum(np.sum(doses_administered_by_un_region, axis=2), axis=1))

# plot_stacked_bar_chart(doses_administered_by_un_region, x_lim)

# print(population_by_un_region / np.sum(population_by_un_region, axis=1)[:, None])

# print(np.sum(coverage_by_un_region, axis=2))

# plot_cumulative_coverage_by_un_region(cumulative_coverage_by_un_region, x_lim)

# plot_coverage(doses_administered_by_un_region, x_lim)

import matplotlib.colors as colors
import matplotlib.cm as cmx
def get_scalar_cmap(cmap, min_val, max_val):
    """Constucts scalar colour map to represent disease prevalence when rendering"""

    cm = plt.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=min_val, vmax=max_val)

    return cmx.ScalarMappable(norm=cNorm, cmap=cm)
scalar_cmap = get_scalar_cmap("Blues", 0, 1)

for prev in range(100):
    prev /= 100
    colour = scalar_cmap.to_rgba(prev, bytes = True)