#ifndef MCOPT_MCMINIMIZER_H
#define MCOPT_MCMINIMIZER_H

#include <armadillo>
#include <vector>
#include <tuple>
#include "Tracker.h"
#include "Track.h"
#include "EventGen.h"

namespace mcopt
{
    arma::vec dropNaNs(const arma::vec& data);

    class MCminimizeResult
    {
    public:
        arma::vec ctr;
        arma::mat allParams;
        arma::vec minPosChis;
        arma::vec minEnChis;
        arma::vec goodParamIdx;
    };

    class MCminimizer
    {
    public:
        MCminimizer(const Tracker& tracker, const EventGenerator& evtgen)
            : tracker(tracker), evtgen(evtgen) {}

        static arma::mat makeParams(const arma::vec& ctr, const arma::vec& sigma, const unsigned numSets,
                                    const arma::vec& mins, const arma::vec& maxes);
        static arma::mat findPositionDeviations(const arma::mat& simPos, const arma::mat& expPos);
        arma::vec findEnergyDeviation(const arma::mat& simPos, const arma::vec& simEn, const arma::vec& expMesh) const;
        arma::mat prepareSimulatedTrackMatrix(const arma::mat& simtrack) const;
        std::tuple<double, double> runTrack(const arma::vec& params, const arma::mat& expPos, const arma::vec& expMesh) const;
        MCminimizeResult minimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& expPos,
                                  const arma::vec& expMesh, const unsigned numIters, const unsigned numPts,
                                  const double redFactor) const;

    private:
        Tracker tracker;
        EventGenerator evtgen;
    };
}

#endif /* end of include guard: MCOPT_MCMINIMIZER_H */
