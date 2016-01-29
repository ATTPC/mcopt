#include "MCminimizer.h"

namespace mcopt
{
    arma::vec dropNaNs(const arma::vec& data)
    {
        if (!data.has_nan()) {
            return data;
        }
        arma::vec res (data.n_rows, data.n_cols);
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
        result.col(0) = (xInterp - expPos.col(0)) / 0.5e-2;  // Valid?
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

    double MCminimizer::runTrack(const arma::vec& params, const arma::mat& expPos, const arma::vec& expMesh) const
    {
        arma::vec3 thisBfield = {0, 0, params(6)};

        Track tr = tracker.trackParticle(params(0), params(1), params(2), params(3), params(4), params(5), thisBfield);
        arma::mat simPos = tr.getPositionMatrix();
        arma::vec simEn = tr.getEnergyVector();

        double zlenSim = simPos.col(2).max() - simPos.col(2).min();
        double zlenExp = expPos.col(2).max() - expPos.col(2).min();

        double posChi2 = 0;
        double enChi2 = 0;
        double chi2 = 0;

        if (simPos.n_rows > 10 and (zlenSim - zlenExp) >= -0.05) {
            arma::mat posDevs = findPositionDeviations(simPos, expPos);
            arma::vec validPosDevs = dropNaNs(arma::sum(arma::square(posDevs), 1));
            posChi2 = !validPosDevs.is_empty() ? arma::median(validPosDevs) : 200;

            arma::vec enDevs = findEnergyDeviation(simPos, simEn, expMesh);
            arma::vec validEnDevs = dropNaNs(arma::square(enDevs));
            enChi2 = !validEnDevs.is_empty() ? arma::mean(validEnDevs) : 200;

            // chi2 = posChi2 + enChi2;
            chi2 = posChi2;
        }
        else {
            chi2 = 100;
        }

        return chi2;
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
        arma::vec minChis(numIters);
        arma::vec goodParamIdx(numIters);

        for (unsigned i = 0; i < numIters; i++) {
            arma::mat params = makeParams(ctr, sigma, numPts, mins, maxes);
            arma::vec chi2s (numPts, arma::fill::zeros);

            #pragma omp parallel for schedule(static)
            for (unsigned j = 0; j < numPts; j++) {
                arma::vec p = arma::conv_to<arma::colvec>::from(params.row(j));
                double chi2;
                try {
                    chi2 = runTrack(p, expPos, expMesh);
                }
                catch (const std::exception&) {
                    chi2 = arma::datum::nan;
                }
                chi2s(j) = chi2;
            }

            arma::uword minChiLoc = 0;
            double minChi = chi2s.min(minChiLoc);

            ctr = arma::conv_to<arma::colvec>::from(params.row(minChiLoc));
            sigma *= redFactor;

            allParams.rows(i*numPts, (i+1)*numPts-1) = params;
            minChis(i) = minChi;
            goodParamIdx(i) = minChiLoc + i*numPts;
        }
        return std::make_tuple(ctr, allParams, minChis, goodParamIdx);
    }
}
