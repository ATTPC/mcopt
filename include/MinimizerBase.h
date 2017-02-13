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
        double posChi2 = 0;
        double enChi2 = 0;
        double vertChi2 = 0;

        double sum() const { return posChi2 + enChi2 + vertChi2; }
        constexpr size_t numChis() const { return 3; }
        arma::rowvec asRow() const { return arma::rowvec {posChi2, enChi2, vertChi2}; }
    };

    class MinimizerBase
    {
    public:
        MinimizerBase(const Tracker* tracker_, const EventGenerator* evtgen_)
            : tracker(tracker_), evtgen(evtgen_) {}

        // Functions to calculate deviations for chi2 components
        arma::mat findPositionDeviations(const arma::mat& simPos, const arma::mat& expPos) const;
        arma::vec findHitPatternDeviation(const arma::mat& simPos, const arma::vec& simEn, const arma::vec& expHits) const;
        double findVertexDeviationFromOrigin(const double x0, const double y0) const;

        // Functions to calculate chi2 components
        double findPosChi2(const arma::mat& simPos, const arma::mat& expPos) const;
        double findEnChi2(const arma::mat& simPos, const arma::vec& simEn, const arma::vec& expHits) const;
        double findVertChi2(const double x0, const double y0) const;

        // Track runners
        Chi2Set runTrack(const arma::vec& params, const arma::mat& expPos, const arma::vec& expHits) const;
        arma::mat runTracks(const arma::mat& params, const arma::mat& expPos, const arma::vec& expHits) const;

        // Flags to enable/disable objective function components
        bool posChi2Enabled = true;
        bool enChi2Enabled = true;
        bool vertChi2Enabled = true;

        // Chi2 normalization constants
        double posChi2Norm = 0.5e-2;
        double enChi2NormFraction = 0.10;
        double vertChi2Norm = 0.5e-4;

    private:
        const Tracker* const tracker;
        const EventGenerator* const evtgen;
    };
}

#endif /* end of include guard: MCOPT_MINIMIZER_BASE */
