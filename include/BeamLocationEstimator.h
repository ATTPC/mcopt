#ifndef BEAMLOCATIONESTIMATOR_H
#define BEAMLOCATIONESTIMATOR_H

#include <armadillo>

namespace mcopt
{
    class BeamLocationEstimator
    {
    public:
        BeamLocationEstimator(const double xslope_, const double xint_, const double yslope_, const double yint_)
        : xslope(xslope_), xint(xint_), yslope(yslope_), yint(yint_) {}

        double findX(const double z) const;
        arma::vec findX(const arma::vec& z) const;

        double findY(const double z) const;
        arma::vec findY(const arma::vec& z) const;

    private:
        double linearFunction(const double z, const double slope, const double intercept) const;
        arma::vec linearFunction(const arma::vec& z, const double slope, const double intercept) const;

        double xslope;
        double xint;
        double yslope;
        double yint;
    };
}

#endif /* end of include guard: BEAMLOCATIONESTIMATOR_H */
