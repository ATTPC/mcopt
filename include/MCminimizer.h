#ifndef MCOPT_MCMINIMIZER_H
#define MCOPT_MCMINIMIZER_H

#include <armadillo>
#include <vector>
#include <tuple>
#include <algorithm>
#include "Tracker.h"
#include "Track.h"
#include "EventGen.h"
#include "MinimizerBase.h"
#include "BeamLocationEstimator.h"

namespace mcopt
{
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

    class MCminimizer : public MinimizerBase
    {
    public:
        MCminimizer(const Tracker* tracker, const EventGenerator* evtgen)
            : MinimizerBase(tracker, evtgen) {}

        static arma::mat makeParams(const arma::vec& ctr, const arma::vec& sigma, const unsigned numSets,
                                    const arma::vec& mins, const arma::vec& maxes,
                                    const BeamLocationEstimator& beamloc);
        MCminimizeResult minimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& expPos,
                                  const arma::vec& expMesh, const unsigned numIters, const unsigned numPts,
                                  const double redFactor, const BeamLocationEstimator& beamloc) const;
    };
}

#endif /* end of include guard: MCOPT_MCMINIMIZER_H */
