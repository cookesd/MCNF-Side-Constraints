Project intended to solve a MCNF problem with side constraints
Utilizes the pyomo package to solve the base MCNF problem
Uses gradient descent with line search (step size selection) to update dual variables:
	- Need to figure out if there's a better way, because the step size keeps coming out to 0.

TODO:
- Clean up interations with the class (what methods to call)
- Make a result class to return from the solve method
- Add visualizations of the original problem and solution (networkx)

