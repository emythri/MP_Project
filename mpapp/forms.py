from django import forms

class LPSolverForm(forms.Form):
    c1 = forms.FloatField(label="Objective Function Coefficient for x1", required=True)
    c2 = forms.FloatField(label="Objective Function Coefficient for x2", required=True)

    num_constraints = forms.IntegerField(
        label="Number of Constraints",
        min_value=1,
        required=True,
        initial=2
    )
