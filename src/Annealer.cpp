#include <Annealer.h>

namespace mcopt {

    arma::vec Annealer::randomStep(const arma::vec& ctr, const arma::vec& sigma) const
    {
        assert(ctr.n_elem == sigma.n_elem);
        return ctr + (arma::randu<arma::vec>(arma::size(sigma)) - 0.5) % sigma;
    }

    bool Annealer::solutionIsBetter(const double newChi, const double oldChi, AnnealerState& state) const
    {
        bool isBetter;

        if (newChi < oldChi) {
            isBetter = true;  // Solution is good if the result was better than the last iteration
        }
        else {
            // If not, keep the solution according to some probability
            std::uniform_real_distribution<double> uniDistr(0, 1);
            double energy = std::exp(-(newChi - oldChi) / state.temp);
            isBetter = energy > uniDistr(state.randomEngine);
        }

        return isBetter;
    }

    void Annealer::findNextPoint(AnnealerState& state) const
    {
        arma::vec ctr;
        Chi2Set trialChis;
        bool foundGoodPoint = false;

        const arma::vec& lastCtr = state.ctrs.back();
        const double lastTotalChi = state.chis.back().sum();

        for (int iterCalls = 0; iterCalls < maxCallsPerIter && !foundGoodPoint; iterCalls++) {
            // Make a trial step and evaluate the objective function
            ctr = randomStep(lastCtr, state.sigma);

            try {
                trialChis = runTrack(ctr, state.expPos, state.expHits);
                state.numCalls++;
            }
            catch (const std::exception&) {
                continue;  // Eat the exception and give up on this trial
            }

            double trialTotalChi = trialChis.sum();

            // Now determine whether to keep this step
            foundGoodPoint = solutionIsBetter(trialTotalChi, lastTotalChi, state);
        }

        if (foundGoodPoint) {
            state.ctrs.push_back(ctr);
            state.chis.push_back(trialChis);
            return;
        }
        else {
            throw AnnealerReachedMaxCalls();
        }
    }

    AnnealResult Annealer::minimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& expPos,
                                    const arma::vec& expHits) const
    {
        assert(ctr0.n_elem > 0);
        assert(ctr0.n_elem == sigma0.n_elem);

        AnnealerState state;
        state.temp = T0;
        state.sigma = sigma0;
        state.expPos = expPos;
        state.expHits = expHits;

        state.ctrs.push_back(ctr0);
        state.chis.push_back(runTrack(ctr0, expPos, expHits));
        state.numCalls++;

        for (int iter = 0; iter < numIters; iter++) {
            try {
                findNextPoint(state);
            }
            catch (const AnnealerReachedMaxCalls&) {
                return AnnealResult(state.ctrs, state.chis, AnnealStopReason::tooManyCalls, state.numCalls);
            }

            // Cool the system before the next iteration
            state.temp *= coolRate;
            state.sigma *= coolRate;
        }

        return AnnealResult(state.ctrs, state.chis, AnnealStopReason::maxIters, state.numCalls);
    }
}
