
cimport ceres
from libcpp cimport bool
include "util.pyx"

cdef object global_py_callback

cdef bool callback_wrapper(const double* x, const int size, double* func, double* grad):
    global global_py_callback
    params = array_fromaddress(<long>x, (size,))
    func_, grad_, valid = global_py_callback(params)
    
    func[0] = func_

    cdef int i = 0
    for i in range(size):
        grad[i] = grad_[i]

    return valid

cdef cppclass Callback(ceres.FirstOrderFunction):
    # This pure-C++ class is used to interface arbitrary Python functions
    # with ceres::GradientProblemSolver.
    # Because Python objects cannot be members of C++ classes, we save our
    # callback globally
    int num_params

    Callback(int num_params):
        this.num_params = num_params

    bool Evaluate(const double* parameters, double* cost, double* gradient) const:
        return callback_wrapper(parameters, this.num_params, cost, gradient)
    
    int NumParameters() const:
        return this.num_params


cdef class FirstOrderFunction:
    '''
    A FirstOrderFunction object implements the evaluation of a function
    and its gradient.
    '''
    cdef Callback* _callback

    def __init__(self):
        self._callback = new Callback(self.numParameters());

    def evaluate(self, params):
        '''
        Should return cost (scalar), gradient (vector) and isValid (boolean)
        '''
        pass

    def numParameters(self):
        pass

    def _refreshCallback(self):
        global global_py_callback
        global_py_callback = self.evaluate

cdef class GradientProblem:
    '''
    Instances of GradientProblem represent general non-linear
    optimization problems that must be solved using just the value of
    the objective function and its gradient. Unlike the Problem class,
    which can only be used to model non-linear least squares problems,
    instances of GradientProblem not restricted in the form of the
    objective function.

    Structurally GradientProblem is a composition of a
    FirstOrderFunction and optionally a LocalParameterization.

    The FirstOrderFunction is responsible for evaluating the cost and
    gradient of the objective function.

    The LocalParameterization is responsible for going back and forth
    between the ambient space and the local tangent space. (See
    local_parameterization.h for more details). When a
    LocalParameterization is not provided, then the tangent space is
    assumed to coincide with the ambient Euclidean space that the
    gradient vector lives in.

    Example usage:

    The following demonstrate the problem construction for Rosenbrock's function

      f(x,y) = (1-x)^2 + 100(y - x^2)^2;

    class Rosenbrock : public ceres::FirstOrderFunction {
     public:
      virtual ~Rosenbrock() {}

      virtual bool Evaluate(const double* parameters,
                            double* cost,
                            double* gradient) const {
        const double x = parameters[0];
        const double y = parameters[1];

        cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
        if (gradient != NULL) {
          gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
          gradient[1] = 200.0 * (y - x * x);
        }
        return true;
      };

      virtual int NumParameters() const { return 2; };
    };

    ceres::GradientProblem problem(new Rosenbrock());
    '''
    cdef ceres.GradientProblem* _problem
    cdef FirstOrderFunction _function

    def __init__(self, FirstOrderFunction function):
        self._function = function
        cdef ceres.FirstOrderFunction* callback = <ceres.FirstOrderFunction*>function._callback
        self._problem = new ceres.GradientProblem(callback)

    def _refreshCallback(self):
        self._function._refreshCallback()

