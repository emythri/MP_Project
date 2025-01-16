from django.shortcuts import render
from django.http import HttpResponse
from .utils import simplex
import numpy as np

# Home view
def home(request):
    return render(request, 'home.html')

from django.shortcuts import render
from django.http import HttpResponse
from .utils import simplex
import numpy as np

# Home view
def home(request):
    return render(request, 'home.html')

# Simplex view
def simplex_view(request):
    if request.method == 'POST':
        try:
            # Generate the form for specifying variables and constraints
            if 'generate' in request.POST:
                num_vars = int(request.POST['num_vars'])
                num_constraints = int(request.POST['num_constraints'])
                return render(request, 'simplex.html', {
                    'num_vars': num_vars,
                    'num_constraints': num_constraints,
                    'var_range': range(1, num_vars + 1),
                    'constraint_range': range(1, num_constraints + 1),
                })

            # Solve the simplex problem
            elif 'solve' in request.POST:
                # Parse the number of variables and constraints
                num_vars = int(request.POST['num_vars'])
                num_constraints = int(request.POST['num_constraints'])

                # Parse the objective function coefficients
                c = np.array([float(request.POST[f'c_{i}']) for i in range(1, num_vars + 1)])

                # Parse the constraints matrix A and vector b
                A = np.array([
                    [float(request.POST[f'a_{i}_{j}']) for j in range(1, num_vars + 1)]
                    for i in range(1, num_constraints + 1)
                ])
                b = np.array([float(request.POST[f'b_{i}']) for i in range(1, num_constraints + 1)])

                # Solve using the simplex method
                solution, optimal_value, final_tableau = simplex(c, A, b)

                # Render the results
                return render(request, 'simplex.html', {
                    'num_vars': num_vars,
                    'num_constraints': num_constraints,
                    'var_range': range(1, num_vars + 1),
                    'constraint_range': range(1, num_constraints + 1),
                    'A': A.tolist(),
                    'b': b.tolist(),
                    'c': c.tolist(),
                    'solution': solution.tolist(),
                    'optimal_value': optimal_value,
                    'final_tableau': final_tableau.tolist(),
                })

        except Exception as e:
            return render(request, 'simplex.html', {
                'error': str(e)
            })

    # Default view (initial form)
    return render(request, 'simplex.html', {
        'num_vars': 1,
        'num_constraints': 1,
        'var_range': range(1, 2),
        'constraint_range': range(1, 2),
    })
