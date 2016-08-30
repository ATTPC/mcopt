#include <Annealer.h>

namespace mcopt {
    AnnealResult::AnnealResult(const std::vector<arma::vec>& ctrs_, const std::vector<Chi2Set>& chis_,
                               const AnnealStopReason& reason_, int numCalls_)
        : stopReason(reason_), numCalls(numCalls_)
    {
        ctrs = arma::mat(ctrs_.size(), ctrs_.at(0).n_elem);
        chis = arma::mat(chis_.size(), chis_.at(0).numChis());

        for (arma::uword i = 0; i < ctrs_.size(); i++) {
            ctrs.row(i) = ctrs_.at(i).t();
            chis.row(i) = chis_.at(i).asRow();
        }
    }

    double AnnealResult::getFinalChiTotal() const
    {
        return arma::accu(chis.tail_rows(1));
    }

    arma::vec Annealer::randomStep(const arma::vec& ctr, const arma::vec& sigma) const
    {
        assert(ctr.n_elem == sigma.n_elem);
        // return ctr + (arma::randu<arma::vec>(arma::size(sigma)) - 0.5) % sigma;
        return ctr + (arma::randn<arma::vec>(arma::size(sigma)) % sigma);
    }

    bool Annealer::solutionIsBetter(const double newChi, const double oldChi, const double T) const
    {
        bool isBetter;

        if (newChi < oldChi) {
            isBetter = true;  // Solution is good if the result was better than the last iteration
        }
        else {
            // If not, keep the solution according to some probability
            static std::mt19937 randomEngine {std::random_device()()};

            std::uniform_real_distribution<double> uniDistr(0, 1);
            double energy = std::exp(-(newChi - oldChi) / T);
            isBetter = energy > uniDistr(randomEngine);
        }

        return isBetter;
    }

    struct Trial
    {
        arma::vec ctr;
        Chi2Set chi;
        bool hadError = false;

        bool operator<(const Trial& other) const { return !hadError && chi.sum() < other.chi.sum(); }
    };

    void Annealer::findNextPoint(AnnealerState& state) const
    {
        const arma::vec& lastCtr = state.ctrs.back();
        const double lastTotalChi = state.chis.back().sum();

        const size_t numThreads = static_cast<size_t>(omp_get_max_threads());

        for (int iterCalls = 0; iterCalls < maxCallsPerIter; iterCalls += numThreads) {
            std::vector<Trial> trials (numThreads);

            #pragma omp parallel
            {
                const size_t thnum = static_cast<size_t>(omp_get_thread_num());
                Trial& thTrial = trials.at(thnum);
                thTrial.ctr = randomStep(lastCtr, state.sigma);
                try {
                    thTrial.chi = runTrack(thTrial.ctr, state.expPos, state.expHits);
                }
                catch (const std::exception&) {
                    thTrial.hadError = true;
                }
            }

            state.numCalls += numThreads;

            auto bestTrialIter = std::min_element(trials.begin(), trials.end());
            if (solutionIsBetter(bestTrialIter->chi.sum(), lastTotalChi, state.temp)) {
                state.ctrs.push_back(bestTrialIter->ctr);
                state.chis.push_back(bestTrialIter->chi);
                return;
            }
        }

        // If we reach this point, we didn't find a good solution, so fail
        throw AnnealerReachedMaxCalls();
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

    AnnealResult Annealer::multiMinimize(const arma::vec& ctr0, const arma::vec& sigma0, const arma::mat& expPos,
                                         const arma::vec& expHits) const
    {
        std::vector<AnnealResult> results (multiMinimizeNumTrials);

        #pragma omp parallel for schedule(static, 1)
        for (size_t trial = 0; trial < multiMinimizeNumTrials; trial++) {
            results.at(trial) = minimize(ctr0, sigma0, expPos, expHits);
        }

        auto comp = [](const AnnealResult& a, const AnnealResult& b) {
            return a.getFinalChiTotal() < b.getFinalChiTotal();
        };

        auto minIter = std::min_element(results.begin(), results.end(), comp);

        return *minIter;
    }
}
