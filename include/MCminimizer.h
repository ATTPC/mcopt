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
        double vertChi2;
    };

    class MCminimizeResult
    {
    public:
        MCminimizeResult() = default;
        MCminimizeResult(const arma::uword numVars, const arma::uword numPts, const arma::uword numIters, const arma::uword numChis)
        : ctr(arma::vec(numVars)), allParams(arma::mat(numPts * numIters, numVars)),
          minChis(arma::mat(numIters, numChis)), goodParamIdx(arma::vec(numIters)) {}

        arma::vec ctr;           /// The minimized set of parameters (i.e. the result)
        arma::mat allParams;     /// The full set of generated parameters, for testing.
        arma::mat minChis;       /// The matrix of chi2 values. Rows are iterations, columns are chi2 variables.
        arma::vec goodParamIdx;  /// Indices into allParams that give the params corresponding to minChis.
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
        double findVertexDeviationFromOrigin(const double x0, const double y0) const;
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
