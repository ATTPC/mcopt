#ifndef MCOPT_ANNEALER_H
#define MCOPT_ANNEALER_H

#include <armadillo>
#include "MinimizerBase.h"
#include "EventGen.h"
#include "Tracker.h"
#include <vector>
#include <random>
#include <tuple>
#include <cassert>
#include <algorithm>
#include <omp.h>

namespace mcopt
{
    enum class AnnealStopReason {converged, maxIters, tooManyCalls};

    class AnnealResult
    {
    public:
        AnnealResult() = default;
        AnnealResult(const std::vector<arma::vec>& ctrs_in, const std::vector<Chi2Set>& chis_in,
                     const AnnealStopReason& reason_in, int numCalls_in);

        double getFinalChiTotal() const;

        arma::mat ctrs;           /// The minimized set of parameters (i.e. the result)
        arma::mat chis;       /// The matrix of chi2 values. Rows are iterations, columns are chi2 variable

        AnnealStopReason stopReason;
        int numCalls;
    };

    struct AnnealerState
    {
        std::vector<arma::vec> ctrs;
        std::vector<Chi2Set> chis;
        double temp;
        arma::vec sigma;
        int numCalls = 0;

        arma::mat expPos;
        arma::vec expHits;
    };

    class AnnealerReachedMaxCalls : public std::exception
    {
    public:
        const char* what() const noexcept override { return "Annealer reached the maximum number of calls."; }
    };

    class Annealer : public MinimizerBase
    {
    public:
        Annealer(const Tracker& tracker_, const EventGenerator& evtgen_, const double T0_, const double coolRate_,
                 const int numIters_, const int maxCallsPerIter_)
            : MinimizerBase(tracker_, evtgen_), T0(T0_), coolRate(coolRate_), numIters(numIters_),
              maxCallsPerIter(maxCallsPerIter_), multiMinimizeNumTrials(20) {}

        arma::vec randomStep(const arma::vec& ctr, const arma::vec& sigma) const;
        bool solutionIsBetter(const double newChi, const double oldChi, const double T) const;
        void findNextPoint(AnnealerState& state) const;

        AnnealResult minimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& expPos,
                              const arma::vec& expHits) const;

        AnnealResult multiMinimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& expPos,
                                   const arma::vec& expHits) const;

        // Annealing parameters
        double T0;
        double coolRate;
        int numIters;
        int maxCallsPerIter;
        size_t multiMinimizeNumTrials;
    };
}

#endif /* defined MCOPT_ANNEALER_H */
