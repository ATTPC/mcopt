#include "MCminimizer.h"

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

    arma::mat MCminimizer::findPositionDeviations(const arma::mat& simPos, const arma::mat& expPos)
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

    arma::vec MCminimizer::findEnergyDeviation(const arma::mat& simPos, const arma::vec& simEn,
                                               const arma::vec& expMesh) const
    {
        arma::vec simMesh = evtgen.makeMeshSignal(simPos, simEn);
        double sigma = expMesh.max() * 0.10;
        return (simMesh - expMesh) / sigma;
    }

    std::tuple<double, double> MCminimizer::runTrack(const arma::vec& params, const arma::mat& expPos,
                                                     const arma::vec& expMesh) const
    {
        arma::vec3 thisBfield = {0, 0, params(6)};

        Track tr = tracker.trackParticle(params(0), params(1), params(2), params(3), params(4), params(5), thisBfield);
        arma::mat simPos = tr.getPositionMatrix();
        arma::vec simEn = tr.getEnergyVector();

        double posChi2 = 0;
        double enChi2 = 0;

        arma::mat posDevs = findPositionDeviations(simPos, expPos);
        if (!posDevs.is_empty()) {
            double clampMax = 3;
            arma::vec validPosDevs = replaceNaNs(arma::sum(arma::square(posDevs), 1), clampMax);
            posChi2 = arma::sum(arma::clamp(validPosDevs, 0, clampMax)) / validPosDevs.n_elem;
        }
        else {
            posChi2 = 200;
        }

        arma::vec enDevs = findEnergyDeviation(simPos, simEn, expMesh);
        arma::vec validEnDevs = dropNaNs(arma::square(enDevs));
        enChi2 = arma::sum(arma::clamp(validEnDevs, 0, 100)) / validEnDevs.n_elem;

        return std::make_tuple(posChi2, enChi2);
    }

    arma::mat MCminimizer::makeParams(const arma::vec& ctr, const arma::vec& sigma, const unsigned numSets,
                                      const arma::vec& mins, const arma::vec& maxes)
    {
        const arma::uword numVars = ctr.n_rows;
        assert(sigma.n_rows == numVars);
        assert(mins.n_rows == numVars);
        assert(maxes.n_rows == numVars);

        arma::mat params = arma::randu(numSets, numVars);

        for (arma::uword i = 0; i < numVars; i++) {
            params.col(i) = arma::clamp(ctr(i) + (params.col(i) - 0.5) * sigma(i),
                                        mins(i), maxes(i));
        }

        return params;
    }

    MCminimizeResult MCminimizer::minimize(const arma::vec& ctr0, const arma::vec& sigma0,
                                           const arma::mat& expPos, const arma::vec& expMesh,
                                           const unsigned numIters, const unsigned numPts, const double redFactor) const
    {
        arma::uword numVars = ctr0.n_rows;

        arma::vec mins = ctr0 - sigma0 / 2;
        if (mins(3) < 0) {
            mins(3) = 0;  // Prevent energy from being negative
        }
        arma::vec maxes = ctr0 + sigma0 / 2;
        arma::vec ctr = ctr0;
        arma::vec sigma = sigma0;

        arma::mat allParams(numPts * numIters, numVars);
        arma::vec minPosChis(numIters);
        arma::vec minEnChis(numIters);
        arma::vec goodParamIdx(numIters);

        for (unsigned i = 0; i < numIters; i++) {
            arma::mat params = makeParams(ctr, sigma, numPts, mins, maxes);
            arma::vec posChi2s (numPts, arma::fill::zeros);
            arma::vec enChi2s (numPts, arma::fill::zeros);

            #pragma omp parallel for schedule(static)
            for (unsigned j = 0; j < numPts; j++) {
                arma::vec p = arma::conv_to<arma::colvec>::from(params.row(j));
                double posChi2, enChi2;
                try {
                    std::tie(posChi2, enChi2) = runTrack(p, expPos, expMesh);
                }
                catch (...) {
                    posChi2 = arma::datum::nan;
                    enChi2 = arma::datum::nan;
                }
                posChi2s(j) = posChi2;
                enChi2s(j) = enChi2;
            }

            arma::vec totChis = posChi2s + enChi2s;

            arma::uword minChiLoc = 0;
            totChis.min(minChiLoc);

            ctr = arma::conv_to<arma::colvec>::from(params.row(minChiLoc));
            sigma *= redFactor;

            allParams.rows(i*numPts, (i+1)*numPts-1) = params;
            minPosChis(i) = posChi2s(minChiLoc);
            minEnChis(i) = enChi2s(minChiLoc);
            goodParamIdx(i) = minChiLoc + i*numPts;
        }

        MCminimizeResult res;
        res.ctr = std::move(ctr);
        res.allParams = std::move(allParams);
        res.minPosChis = std::move(minPosChis);
        res.minEnChis = std::move(minEnChis);
        res.goodParamIdx = std::move(goodParamIdx);

        return res;
    }
}
