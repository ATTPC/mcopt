#ifndef MCOPT_MCMINIMIZER_H
#define MCOPT_MCMINIMIZER_H

#include <armadillo>
#include <vector>
#include <tuple>
#include <algorithm>
#include "Tracker.h"
#include "Track.h"
#include "EventGen.h"

namespace mcopt
{
    arma::vec dropNaNs(const arma::vec& data);
    arma::vec replaceNaNs(const arma::vec& data, const double replacementValue);

    class Chi2Set
    {
    public:
        double posChi2;
        double enChi2;
    };

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
        arma::vec findHitPatternDeviation(const arma::mat& simPos, const arma::vec& simEn, const arma::vec& expHits) const;
        double findTotalSignalChi(const std::map<pad_t, arma::vec>& simEvt, const std::map<pad_t, arma::vec>& expEvt) const;
        Chi2Set runTrack(const arma::vec& params, const arma::mat& expPos, const arma::vec& expHits) const;
        arma::mat runTracks(const arma::mat& params, const arma::mat& expPos, const arma::vec& expHits) const;
        MCminimizeResult minimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& expPos,
                                  const arma::vec& expMesh, const unsigned numIters, const unsigned numPts,
                                  const double redFactor) const;

    private:
        Tracker tracker;
        EventGenerator evtgen;
    };
}

#endif /* end of include guard: MCOPT_MCMINIMIZER_H */
