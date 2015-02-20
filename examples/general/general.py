
from cyres import *
import numpy as np

ff = FirstOrderFunction(1, lambda x: x[0]**8, lambda x: np.array([8*x[0]**7], dtype=np.float64))
print(ff.evaluate(np.array([2], dtype=np.float64)))
prob = GradientProblem(ff)
options = GradientProblemSolverOptions()

solver = GradientProblemSolver()
init = np.array([10], dtype=np.float64)
summary = solver.solve(options, prob, init)

print(summary.fullReport())
print(init)

