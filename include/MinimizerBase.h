#ifndef MCOPT_MINIMIZER_BASE
#define MCOPT_MINIMIZER_BASE

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
        double vertChi2;

        double sum() const { return posChi2 + enChi2 + vertChi2; }
        constexpr int numChis() const { return 3; }
        arma::rowvec asRow() const { return arma::rowvec {posChi2, enChi2, vertChi2}; }
    };

    class MinimizerBase
    {
    public:
        MinimizerBase(const Tracker& tracker, const EventGenerator& evtgen)
            : tracker(tracker), evtgen(evtgen) {}

        // Objective function components
        static arma::mat findPositionDeviations(const arma::mat& simPos, const arma::mat& expPos);
        arma::vec findEnergyDeviation(const arma::mat& simPos, const arma::vec& simEn, const arma::vec& expMesh) const;
        arma::vec findHitPatternDeviation(const arma::mat& simPos, const arma::vec& simEn, const arma::vec& expHits) const;
        double findVertexDeviationFromOrigin(const double x0, const double y0) const;
        double findTotalSignalChi(const std::map<pad_t, arma::vec>& simEvt, const std::map<pad_t, arma::vec>& expEvt) const;

        // Track runners
        Chi2Set runTrack(const arma::vec& params, const arma::mat& expPos, const arma::vec& expHits) const;
        arma::mat runTracks(const arma::mat& params, const arma::mat& expPos, const arma::vec& expHits) const;

    private:
        Tracker tracker;
        EventGenerator evtgen;
    };
}

#endif /* end of include guard: MCOPT_MINIMIZER_BASE */
