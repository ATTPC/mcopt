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

    arma::mat MinimizerBase::findPositionDeviations(const arma::mat& simPos, const arma::mat& expPos) const
    {
        // ASSUMPTION: matrices must be sorted in increasing Z order.
        // Assume also that the matrices are structured as:
        //     (x, y, z, ...)

        arma::vec xInterp;
        arma::vec yInterp;

        arma::interp1(simPos.col(2), simPos.col(0), expPos.col(2), xInterp);
        arma::interp1(simPos.col(2), simPos.col(1), expPos.col(2), yInterp);

        arma::mat result (xInterp.n_rows, 2);
        result.col(0) = (xInterp - expPos.col(0)) / posChi2Norm;
        result.col(1) = (yInterp - expPos.col(1)) / posChi2Norm;

        return result;
    }

    arma::vec MinimizerBase::findHitPatternDeviation(const arma::mat& simPos, const arma::vec& simEn,
                                                     const arma::vec& expHits) const
    {
        arma::vec simHits = evtgen->makeHitPattern(simPos, simEn);
        double sigma = expHits.max() * enChi2NormFraction;
        return (simHits - expHits) / sigma;
    }

    double MinimizerBase::findPosChi2(const arma::mat& simPos, const arma::mat& expPos) const
    {
        double posChi2 = 0;
        arma::mat posDevs = findPositionDeviations(simPos, expPos);
        if (!posDevs.is_empty()) {
            double clampMax = 100.0;
            arma::vec validPosDevs = replaceNaNs(arma::sum(arma::square(posDevs), 1), clampMax);
            posChi2 = arma::sum(arma::clamp(validPosDevs, 0, clampMax)) / validPosDevs.n_elem;
        }
        else {
            posChi2 = 200;
        }

        return posChi2;
    }

    double MinimizerBase::findEnChi2(const arma::mat& simPos, const arma::vec& simEn, const arma::vec& expHits) const
    {
        double enChi2 = 0;

        arma::vec enDevs = findHitPatternDeviation(simPos, simEn, expHits);
        arma::vec validEnDevs = dropNaNs(arma::square(enDevs));
        arma::uvec nonzeroLocs = arma::find(expHits > 0);
        enChi2 = arma::sum(arma::clamp(validEnDevs(nonzeroLocs), 0, 100)) / nonzeroLocs.n_elem;

        return enChi2;
    }

    double MinimizerBase::findVertChi2(const double x0, const double y0) const
    {
        return (x0*x0 + y0*y0) / vertChi2Norm;
    }

    Chi2Set MinimizerBase::runTrack(const arma::vec& params, const arma::mat& expPos,
                                                     const arma::vec& expHits) const
    {
        Track tr = tracker->trackParticle(params(0), params(1), params(2), params(3), params(4), params(5));
        if (tr.numPts() < 2) {
            throw TrackingFailed("Track was too short");
        }

        arma::mat simPos = tr.getPositionMatrix();
        arma::vec simEn = tr.getEnergyVector();

        Chi2Set chis;

        if (posChi2Enabled) {
            chis.posChi2 = findPosChi2(simPos, expPos);
        }
        if (enChi2Enabled) {
            chis.enChi2 = findEnChi2(simPos, simEn, expHits);
        }
        if (vertChi2Enabled) {
            chis.vertChi2 = findVertChi2(params(0), params(1));
        }

        return chis;
    }

    arma::mat MinimizerBase::runTracks(const arma::mat& params, const arma::mat& expPos, const arma::vec& expHits) const
    {
        arma::mat chis (params.n_rows, Chi2Set::numChis());

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
