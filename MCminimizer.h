#ifndef MCOPT_MCMINIMIZER_H
#define MCOPT_MCMINIMIZER_H

#include <armadillo>
#include <vector>
#include "Tracker.h"
#include "Track.h"

namespace mcopt
{
    arma::vec dropNaNs(const arma::vec& data);

    typedef std::tuple<arma::vec, arma::mat, arma::vec, arma::vec> MCminimizeResult;

    class MCminimizer
    {
    public:
        MCminimizer(const unsigned massNum, const unsigned chargeNum, const std::vector<double>& eloss,
                    const arma::vec3& efield, const arma::vec3& bfield)
            : efield(efield), bfield(bfield), tracker(Tracker(massNum, chargeNum, eloss, efield, bfield)) {}

        static arma::mat makeParams(const arma::vec& ctr, const arma::vec& sigma, const unsigned numSets,
                                    const arma::vec& mins, const arma::vec& maxes);
        static arma::mat findDeviations(const arma::mat& simtrack, const arma::mat& expdata);
        double runTrack(const arma::vec& p, const arma::mat& trueValues) const;
        MCminimizeResult minimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& trueValues,
                                  const unsigned numIters, const unsigned numPts, const double redFactor) const;

    private:
        arma::vec3 efield;
        arma::vec3 bfield;
        Tracker tracker;
    };
}

#endif /* end of include guard: MCOPT_MCMINIMIZER_H */
