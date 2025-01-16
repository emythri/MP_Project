from django.shortcuts import render
from django.http import HttpResponse
from .utils import simplex
import numpy as np
from .forms import LPSolverForm
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for rendering plots
import matplotlib.pyplot as plt


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

#----------------------------------------

# Function to solve the linear programming problem
def solve_linear_program(c, A, b):
    """
    Solve the linear programming problem using vertex enumeration.

    Args:
        c: List of coefficients for the objective function.
        A: List of coefficients for the constraints.
        b: List of RHS values for the constraints.

    Returns:
        - List of feasible vertices
        - Optimal vertex
        - Optimal value
    """
    vertices = []
    num_constraints = len(A)
    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            A_ = np.array([A[i], A[j]])
            b_ = np.array([b[i], b[j]])
            try:
                vertex = np.linalg.solve(A_, b_)
                if all(np.dot(A, vertex) <= b) and all(vertex >= 0):
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue

    feasible_vertices = np.unique(vertices, axis=0)
    if len(feasible_vertices) > 0:
        z_values = [np.dot(c, v) for v in feasible_vertices]
        optimal_value = max(z_values)
        optimal_vertex = feasible_vertices[np.argmax(z_values)]
        return feasible_vertices, optimal_vertex, optimal_value
    return None, None, None

# Function to generate plot for the solution
def plot_results(constraints, feasible_vertices, optimal_vertex, bounds):
    """
    Generate a plot of the solution.

    Args:
        constraints: List of tuples (coefficients, RHS values).
        feasible_vertices: List of feasible vertices.
        optimal_vertex: Optimal vertex.
        bounds: Plot bounds.

    Returns:
        Base64 encoded image of the plot.
    """
    plt.figure(figsize=(10, 8))
    x = np.linspace(bounds[0], bounds[1], 400)

    # Plot each constraint
    for coeff, b in constraints:
        if coeff[1] != 0:
            y = (b - coeff[0] * x) / coeff[1]
            plt.plot(x, y, label=f"{coeff[0]}x1 + {coeff[1]}x2 ≤ {b}")
        else:
            x_val = b / coeff[0]
            plt.axvline(x_val, color='r', linestyle='--', label=f"x1 = {x_val}")

    # Highlight feasible region
    if feasible_vertices is not None and len(feasible_vertices) > 0:
        hull_vertices = np.vstack([feasible_vertices, feasible_vertices[0]])
        plt.plot(hull_vertices[:, 0], hull_vertices[:, 1], 'lightgreen', alpha=0.5)
        plt.fill(hull_vertices[:, 0], hull_vertices[:, 1], 'lightgreen', alpha=0.3, label='Feasible Region')

    # Plot feasible vertices
    for point in feasible_vertices:
        plt.plot(point[0], point[1], 'bo')

    # Plot optimal vertex
    if optimal_vertex is not None:
        plt.plot(optimal_vertex[0], optimal_vertex[1], 'ro', label='Optimal Solution')

    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linear Programming Solution")
    plt.legend()
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# Graphical method solver view
def lp_solver_view(request):
    if request.method == "POST":
        form = LPSolverForm(request.POST)
        if form.is_valid():
            # Extract coefficients for the objective function
            c = [form.cleaned_data['c1'], form.cleaned_data['c2']]
            num_constraints = form.cleaned_data['num_constraints']

            # Initialize lists for constraint coefficients and RHS values
            A = []
            b = []
            for i in range(num_constraints):
                coeff1 = float(request.POST.get(f'constraint_{i}_c1'))
                coeff2 = float(request.POST.get(f'constraint_{i}_c2'))
                rhs = float(request.POST.get(f'constraint_{i}_rhs'))
                A.append([coeff1, coeff2])
                b.append(rhs)

            # Solve the linear programming problem
            feasible_vertices, optimal_vertex, optimal_value = solve_linear_program(c, A, b)

            if feasible_vertices is not None:
                # Generate a plot of the feasible region and solution
                image = plot_results(zip(A, b), feasible_vertices, optimal_vertex, bounds=[0, max(b)])
                return render(request, "graphical.html", {
                    "form": form,
                    "num_constraints_range": range(num_constraints),
                    "optimal_vertex": optimal_vertex,
                    "optimal_value": optimal_value,
                    "image": image,
                    "A": A,
                    "b": b,
                    "c": c,
                })
            else:
                # No feasible region found
                return render(request, "graphical.html", {
                    "form": form,
                    "num_constraints_range": range(num_constraints),
                    "error": "No feasible region found.",
                     "A": A,
                    "b": b,
                    "c": c,
                })
    else:
        form = LPSolverForm()

    # Render the form with a default range for constraints (e.g., 2 constraints)
    return render(request, "graphical.html", {
        "form": form,
        "num_constraints_range": range(2)  # Default: 2 constraints
    })
