
import numpy as np
import copy

number_of_regions = 5

baseline_contact_matrix = np.random.rand(number_of_regions, number_of_regions)

lockdown_status = np.array([1,1,-1,-1,-1])

border_closure_status = np.array([-1,1,-1,1,1])

lockdown_factor = 0.4

border_closure_factor = 0.1

# Update contact matrix according to non-pharmacheutical interventions (method 1)
contact_matrix_1 = copy.deepcopy(baseline_contact_matrix)
vec = (((1 + border_closure_status)/2) * border_closure_factor) +\
    (1 - ((1 + border_closure_status)/2))
mat = np.ones((number_of_regions, number_of_regions), dtype=float)
contact_matrix_1 = np.transpose(np.multiply(vec, np.transpose(contact_matrix_1)))
baseline_contact_matrix_diag = np.diagonal(baseline_contact_matrix)
new_diag = np.multiply(baseline_contact_matrix_diag * lockdown_factor, (1 + lockdown_status)/2) +\
           np.multiply(baseline_contact_matrix_diag, 1 - ((1 + lockdown_status)/2))
np.fill_diagonal(contact_matrix_1, new_diag)

# Update contact matrix according to non-pharmacheutical interventions (method 2)
contact_matrix_2 = copy.deepcopy(baseline_contact_matrix)
for i in range(number_of_regions):
    if lockdown_status[i] == 1:
        contact_matrix_2[i][i] = baseline_contact_matrix[i][i] * lockdown_factor
    if border_closure_status[i] == 1:
        for j in range(number_of_regions):
            if i != j:
                contact_matrix_2[i][j] = baseline_contact_matrix[i][j] * border_closure_factor

# Check methods give same result
for i in range(number_of_regions):
    for j in range(number_of_regions):
        assert contact_matrix_1[i][j] == contact_matrix_2[i][j]
