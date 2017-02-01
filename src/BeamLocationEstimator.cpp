#include "BeamLocationEstimator.h"

namespace mcopt
{
    double BeamLocationEstimator::linearFunction(const double z, const double slope, const double intercept) const
    {
        return intercept + slope * z;
    }

    arma::vec BeamLocationEstimator::linearFunction(const arma::vec& z, const double slope, const double intercept) const
    {
        return intercept + slope * z;
    }

    double BeamLocationEstimator::findX(const double z) const
    {
        return linearFunction(z, xslope, xint);
    }

    arma::vec BeamLocationEstimator::findX(const arma::vec& z) const
    {
        return linearFunction(z, xslope, xint);
    }

    double BeamLocationEstimator::findY(const double z) const
    {
        return linearFunction(z, yslope, yint);
    }

    arma::vec BeamLocationEstimator::findY(const arma::vec& z) const
    {
        return linearFunction(z, yslope, yint);
    }
}
