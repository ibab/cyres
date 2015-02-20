#ifndef WRAPPER_H_
#define WRAPPER_H_

#include <ceres/ceres.h>

typedef double (*EvalFunc)(const double* const);
typedef double* (*EvalGrad)(const double* const);

class Callback: public::ceres::FirstOrderFunction {
    public:
        Callback(int num_params, EvalFunc func, EvalGrad grad) {
            this->func = func;
            this->grad = grad;
            this->num_params = num_params;
        }

        virtual ~Callback() {}

        virtual bool Evaluate(const double* parameters, double* cost, double* gradient) const {
            cost[0] = this->func(parameters);
            double* new_grad = this->grad(parameters);
            for (int i=0; i<this->num_params; i++) {
                gradient[i] = new_grad[i];
            }
            return true;
        }

        virtual int NumParameters() const {
            return this->num_params;
        }

    private:
        int num_params;
        EvalFunc func;
        EvalGrad grad;
};


#endif
