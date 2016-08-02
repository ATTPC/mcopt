#include <Annealer.h>

namespace mcopt {

    arma::vec Annealer::randomStep(const arma::vec& ctr, const arma::vec& sigma) const
    {
        return ctr + (arma::randu<arma::vec>(arma::size(sigma)) - 0.5) * sigma;
    }

    bool Annealer::solutionIsBetter(const double newChi, const double oldChi, const double T)
    {
        bool isBetter;

        if (newChi < oldChi) {
            isBetter = true;  // Solution is good if the result was better than the last iteration
        }
        else {
            // If not, keep the solution according to some probability
            std::uniform_real_distribution<double> uniDistr(0, 1);
            double energy = std::exp(-(newChi - oldChi) / T);
            isBetter = energy > uniDistr(randomEngine);
        }

        return isBetter;
    }

    std::tuple<arma::vec, Chi2Set>
    Annealer::findNextPoint(const arma::vec& lastCtr, const double lastChi, const arma::vec& sigma, const double T,
                            const arma::mat& expPos, const arma::vec& expHits, const int maxCallsPerIter)
    {
        arma::vec ctr;
        Chi2Set trialChis;
        bool foundGoodPoint = false;

        for (int iterCalls = 0; iterCalls < maxCallsPerIter && !foundGoodPoint; iterCalls++) {
            // Make a trial step and evaluate the objective function
            arma::vec ctr = lastCtr + randomStep(lastCtr, sigma);

            Chi2Set trialChis;
            try {
                trialChis = runTrack(ctr, expPos, expHits);
            }
            catch (const std::exception&) {
                continue;  // Eat the exception and give up on this trial
            }

            double trialTotal = trialChis.sum();

            // Now determine whether to keep this step
            foundGoodPoint = solutionIsBetter(trialTotal, lastChi, T);
        }

        if (foundGoodPoint) {
            return std::make_tuple(ctr, trialChis);
        }
        else {
            throw AnnealerReachedMaxCalls();
        }
    }

    AnnealResult Annealer::minimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& expPos,
            const arma::vec& expHits, const double T0, const double coolRate,
            const int numIters, const int maxCallsPerIter)
    {
        int totNumCalls = 0;

        std::vector<arma::vec> ctrs {ctr0};
        std::vector<Chi2Set> chis {runTrack(ctr0, expPos, expHits)};
        totNumCalls++;

        for (int iter = 0; iter < numIters; iter++) {
            // Cool the system
            const double redFactor = std::pow(coolRate, iter);
            const double T = T0 * redFactor;
            arma::vec sigma = sigma0 * redFactor;

            // Results from the last iteration, for comparison
            const arma::vec lastCtr = ctrs.back();
            const double lastTotal = chis.back().sum();

            try {
                arma::vec newCtr;
                Chi2Set newChis;

                std::tie(newCtr, newChis) = findNextPoint(lastCtr, lastTotal, sigma, T, expPos, expHits, maxCallsPerIter);

                ctrs.push_back(newCtr);
                chis.push_back(newChis);
            }
            catch (const AnnealerReachedMaxCalls&) {
                return AnnealResult(ctrs, chis, AnnealStopReason::tooManyCalls);
            }
        }

        return AnnealResult(ctrs, chis, AnnealStopReason::maxIters);
    }
}
