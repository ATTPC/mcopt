#ifndef MCOPT_ANNEALER_H
#define MCOPT_ANNEALER_H

#include <armadillo>
#include <MinimizerBase.h>
#include <EventGen.h>
#include <Tracker.h>
#include <vector>
#include <random>
#include <tuple>

namespace mcopt
{
    enum class AnnealStopReason {converged, maxIters, tooManyCalls};

    class AnnealResult
    {
    public:
        AnnealResult() = default;
        AnnealResult(const std::vector<arma::vec>& ctrs_in, const std::vector<Chi2Set>& chis_in,
                     const AnnealStopReason& reason_in)
            : stopReason(reason_in)
        {
            ctrs = arma::mat(ctrs_in.size(), ctrs_in.at(0).n_elem);
            chis = arma::mat(chis_in.size(), chis_in.at(0).numChis());

            for (arma::uword i = 0; i < ctrs_in.size(); i++) {
                ctrs.row(i) = ctrs_in.at(i).t();
                chis.row(i) = chis_in.at(i).asRow();
            }
        }

        arma::mat ctrs;           /// The minimized set of parameters (i.e. the result)
        arma::mat chis;       /// The matrix of chi2 values. Rows are iterations, columns are chi2 variable
        AnnealStopReason stopReason;
    };

    class AnnealerReachedMaxCalls : public std::exception
    {
    public:
        const char* what() const noexcept override { return "Annealer reached the maximum number of calls."; }
    };

    class Annealer : public MinimizerBase
    {
    public:
        Annealer(const Tracker& tracker, const EventGenerator& evtgen)
            : MinimizerBase(tracker, evtgen), randomEngine(std::random_device()()) {}

        arma::vec randomStep(const arma::vec& ctr, const arma::vec& sigma) const;
        bool solutionIsBetter(const double newChi, const double oldChi, const double T);
        std::tuple<arma::vec, Chi2Set>
        findNextPoint(const arma::vec& lastCtr, const double lastChi, const arma::vec& sigma, const double T,
                      const arma::mat& expPos, const arma::vec& expHits, const int maxCallsPerIter);

        AnnealResult minimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& expPos,
                const arma::vec& expHits, const double T0, const double coolRate,
                const int numIters, const int maxCallsPerIter);

    private:
        std::mt19937 randomEngine;
    };
}

#endif /* defined MCOPT_ANNEALER_H */
