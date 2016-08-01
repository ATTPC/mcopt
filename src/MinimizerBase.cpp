#include "MinimizerBase.h"

namespace mcopt
{
    arma::vec dropNaNs(const arma::vec& data)
    {
        if (!data.has_nan()) {
            return data;
        }
        arma::vec res (arma::size(data));
        arma::uword dataIter = 0;
        arma::uword resIter = 0;

        for (; dataIter < data.n_rows && resIter < res.n_rows; dataIter++) {
            if (!std::isnan(data(dataIter))) {
                res(resIter) = data(dataIter);
                resIter++;
            }
        }

        if (resIter == 0) {
            res.clear();
            return res;
        }
        else {
            return res.rows(0, resIter-1);
        }
    }

    arma::vec replaceNaNs(const arma::vec& data, const double replacementValue)
    {
        if (!data.has_nan()) {
            return data;
        }

        arma::vec res (arma::size(data));

        for (arma::uword i = 0; i < data.n_elem; i++) {
            double v = data(i);
            res(i) = std::isnan(v) ? replacementValue : v;
        }

        return res;
    }

    arma::mat MinimizerBase::findPositionDeviations(const arma::mat& simPos, const arma::mat& expPos)
    {
        // ASSUMPTION: matrices must be sorted in increasing Z order.
        // Assume also that the matrices are structured as:
        //     (x, y, z, ...)

        arma::vec xInterp;
        arma::vec yInterp;

        arma::interp1(simPos.col(2), simPos.col(0), expPos.col(2), xInterp);
        arma::interp1(simPos.col(2), simPos.col(1), expPos.col(2), yInterp);

        arma::mat result (xInterp.n_rows, 2);
        result.col(0) = (xInterp - expPos.col(0)) / 0.5e-2;
        result.col(1) = (yInterp - expPos.col(1)) / 0.5e-2;

        return result;
    }

    arma::vec MinimizerBase::findEnergyDeviation(const arma::mat& simPos, const arma::vec& simEn,
                                               const arma::vec& expMesh) const
    {
        arma::vec simMesh = evtgen.makeMeshSignal(simPos, simEn);
        double sigma = expMesh.max() * 0.10;
        return (simMesh - expMesh) / sigma;
    }

    arma::vec MinimizerBase::findHitPatternDeviation(const arma::mat& simPos, const arma::vec& simEn,
                                                   const arma::vec& expHits) const
    {
        arma::vec simHits = evtgen.makeHitPattern(simPos, simEn);
        double sigma = expHits.max() * 0.10;
        return (simHits - expHits) / sigma;
    }

    double MinimizerBase::findTotalSignalChi(const std::map<pad_t, arma::vec>& simEvt,
                                           const std::map<pad_t, arma::vec>& expEvt) const
    {
        auto simIter = simEvt.cbegin();
        auto expIter = expEvt.cbegin();

        // Merging operation is (sim - exp)^2 for each TB

        double accum = 0;

        while (simIter != simEvt.cend() && expIter != expEvt.cend()) {
            if (simIter->first < expIter->first) {
                // This trace is only in the sim track
                accum += arma::sum(arma::square(simIter->second));
                simIter++;
            }
            else if (expIter->first < simIter->first) {
                // This trace is only in the exp track
                accum += arma::sum(arma::square(expIter->second));
                expIter++;
            }
            else {
                // This trace is in both tracks
                accum += arma::sum(arma::square(simIter->second - expIter->second));
                simIter++;
                expIter++;
            }
        }
        // Clean up any traces remaining in the other map after one map reaches the end
        for (; simIter != simEvt.cend(); simIter++) {
            accum += arma::sum(arma::square(simIter->second));
        }
        for (; expIter != expEvt.cend(); expIter++) {
            accum += arma::sum(arma::square(expIter->second));
        }

        return accum;
    }

    double MinimizerBase::findVertexDeviationFromOrigin(const double x0, const double y0) const
    {
        return (x0*x0 + y0*y0) / 0.5e-4;
    }

    Chi2Set MinimizerBase::runTrack(const arma::vec& params, const arma::mat& expPos,
                                                     const arma::vec& expHits) const
    {
        Track tr = tracker.trackParticle(params(0), params(1), params(2), params(3), params(4), params(5));
        if (tr.numPts() < 2) {
            throw TrackingFailed("Track was too short");
        }

        arma::mat simPos = tr.getPositionMatrix();
        arma::vec simEn = tr.getEnergyVector();

        double posChi2 = 0;
        double enChi2 = 0;
        double vertChi2 = 0;

        arma::mat posDevs = findPositionDeviations(simPos, expPos);
        if (!posDevs.is_empty()) {
            double clampMax = 100.0;
            arma::vec validPosDevs = replaceNaNs(arma::sum(arma::square(posDevs), 1), clampMax);
            posChi2 = arma::sum(arma::clamp(validPosDevs, 0, clampMax)) / validPosDevs.n_elem;
        }
        else {
            posChi2 = 200;
        }

        arma::vec enDevs = findHitPatternDeviation(simPos, simEn, expHits);
        arma::vec validEnDevs = dropNaNs(arma::square(enDevs));
        arma::uvec nonzeroLocs = arma::find(expHits > 0);
        enChi2 = arma::sum(arma::clamp(validEnDevs(nonzeroLocs), 0, 100)) / nonzeroLocs.n_elem;

        vertChi2 = findVertexDeviationFromOrigin(params(0), params(1));

        return Chi2Set {posChi2, enChi2, vertChi2};
    }

    arma::mat MinimizerBase::runTracks(const arma::mat& params, const arma::mat& expPos, const arma::vec& expHits) const
    {
        arma::mat chis (params.n_rows, 3);

        #pragma omp parallel for schedule(static) shared(chis)
        for (unsigned j = 0; j < params.n_rows; j++) {
            arma::vec p = arma::conv_to<arma::colvec>::from(params.row(j));
            try {
                auto chiset = runTrack(p, expPos, expHits);
                chis(j, 0) = chiset.posChi2;
                chis(j, 1) = chiset.enChi2;
                chis(j, 2) = chiset.vertChi2;
            }
            catch (...) {
                chis(j, 0) = arma::datum::nan;
                chis(j, 1) = arma::datum::nan;
                chis(j, 2) = arma::datum::nan;
            }
        }

        return chis;
    }
}
