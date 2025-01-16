import numpy as np

def simplex(c, A, b):
    num_constraints, num_variables = A.shape
    slack_vars = np.eye(num_constraints)
    tableau = np.hstack((A, slack_vars, b.reshape(-1, 1)))
    obj_row = np.hstack((-c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, obj_row))
    num_total_vars = num_variables + num_constraints
    while True:
        # Step 1: Check for optimality (no negative values in the objective row)
        if np.all(tableau[-1, :-1] >= 0):
            break

        # Step 2: Determine the entering variable (most negative in the objective row)
        pivot_col = np.argmin(tableau[-1, :-1])

        # Step 3: Determine the leaving variable (smallest positive ratio of RHS / pivot column)
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf  # Ignore non-positive ratios
        pivot_row = np.argmin(ratios)

        if np.all(ratios == np.inf):
            raise ValueError("The problem is unbounded.")

        # Step 4: Perform the pivot operation
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    solution = np.zeros(num_total_vars)
    for i in range(num_constraints):
        basic_var_index = np.where(tableau[i, :-1] == 1)[0]
        if len(basic_var_index) == 1 and basic_var_index[0] < num_total_vars:
            solution[basic_var_index[0]] = tableau[i, -1]
    optimal_value = tableau[-1, -1]
    return solution[:num_variables], optimal_value, tableau
