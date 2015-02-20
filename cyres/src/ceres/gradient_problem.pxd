
from libcpp cimport bool

cdef extern from "gradient_problem.h" namespace "ceres":
    cdef cppclass FirstOrderFunction:
        FirstOrderFunction()
        bool Evaluate(const double* parameters,
                      double* cost,
                      double* gradient) const
        int NumParameters() const

    cdef cppclass GradientProblem:
        GradientProblem(FirstOrderFunction* function)
        #GradientProblem(LocalParametrization* parametrization)
        #int NumParameters() const
        #int NumLocalParameters()
        bool Evaluate(const double* parameters,
                      double* cost,
                      double* gradient) const;
        bool Plus(const double* x,
                  const double* delta,
                  double* x_plus_delta) const;

