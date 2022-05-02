
from collections import defaultdict
import csv

# Country names and iso labels
isos_numeric = []
isos_dict = {}
names_dict = {}
with open('data_isos.csv', newline='') as csvfile:
    region_data = csv.reader(csvfile, delimiter=',')
    for row in region_data:
        isos_numeric.append(int(row[3]))
        isos_dict[int(row[3])] = str(row[1])
        names_dict[int(row[3])] = str(row[0])

# Population size and age structure
age_structure_dict = defaultdict(dict)
population_sizes = {}
with open('WPP2019_PopulationBySingleAgeSex_1950-2019.csv', newline='') as csvfile:
    next(csvfile)
    region_data = csv.reader(csvfile, delimiter=',')
    for row in region_data:
        if int(row[4]) == 2019:
            iso = int(row[0])
            start_age = int(row[7])
            age_structure_dict[iso][start_age] = float(row[11])
            if iso in population_sizes:
                population_sizes[iso] += float(row[11]) * 1000
            else:
                population_sizes[iso] = float(row[11]) * 1000

# Normalize age structure to get age distribution
sums = {}
for iso in isos_numeric:
    sums[iso] = sum(list(age_structure_dict[iso].values()))
for iso in isos_numeric:
    for start_age in age_structure_dict[iso]:
        num = age_structure_dict[iso][start_age]
        age_structure_dict[iso][start_age] = round(num / sums[iso], 6)

# Maximum vaccination coverage, using the average where data is missing
coverage_dict = {}
with open('data_vaccine_coverage.csv', newline='') as csvfile:
    region_data = csv.reader(csvfile, delimiter=',')
    for row in region_data:
        coverage_dict[str(row[1])] = float(row[2])
average_coverage = sum(list(coverage_dict.values())) / len(list(coverage_dict.values()))
coverage = {}
for iso in isos_numeric:
    if iso in population_sizes:
        if isos_dict[iso] in coverage_dict:
            coverage[iso] = coverage_dict[isos_dict[iso]]
        else:
            coverage[iso] = average_coverage

# Combine relavent data into single file to be used as simulator input
handle = open('compiled_data.csv', 'w', newline='')
writer = csv.writer(handle)
header = ['name', 'iso', 'population_size', 'vaccine_hesitant'] + [a for a in range(101)]
writer.writerow(header)
for iso in isos_numeric:
    if iso in population_sizes:
        row = [names_dict[iso], isos_dict[iso], int(population_sizes[iso]), coverage[iso]] +\
              list(age_structure_dict[iso].values())
        writer.writerow(row)
handle.close()
