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

    arma::mat MCminimizer::findDeviations(const arma::mat& simtrack, const arma::mat& expdata)
    {
        // ASSUMPTION: matrices must be sorted in increasing Z order.
        // Assume also that the matrices are structured as:
        //     (x, y, z, de, ...)

        arma::vec xInterp;
        arma::vec yInterp;
        arma::vec enInterp;

        arma::interp1(simtrack.col(2), simtrack.col(0), expdata.col(2), xInterp);
        arma::interp1(simtrack.col(2), simtrack.col(1), expdata.col(2), yInterp);
        arma::interp1(simtrack.col(2), simtrack.col(3), expdata.col(2), enInterp);

        arma::mat result (xInterp.n_rows, 3);
        result.col(0) = xInterp - expdata.col(0);
        result.col(1) = yInterp - expdata.col(1);
        result.col(2) = enInterp - expdata.col(3);

        return result;
    }

    arma::mat MCminimizer::prepareSimulatedTrackMatrix(const arma::mat& simtrack) const
    {
        // Simulation has columns (x, y, z, time, enu, azi, pol).
        // Need to output (x, y, z, de)

        arma::mat result (simtrack.n_rows, 4, arma::fill::zeros);

        result.cols(0, 2) = simtrack.cols(0, 2);
        result.col(3).tail(result.n_rows-1) = -arma::diff(simtrack.col(4)) * 1e6 / ioniz;

        return result;
    }

    double MCminimizer::runTrack(const arma::vec& p, const arma::mat& trueValues) const
    {
        arma::vec3 thisBfield = {0, 0, p(6)};

        Track tr = tracker.trackParticle(p(0), p(1), p(2), p(3), p(4), p(5), thisBfield);
        arma::mat simtrack = tr.getMatrix();

        double zlenSim = simtrack.col(2).max() - simtrack.col(2).min();
        double zlenTrue = trueValues.col(2).max() - trueValues.col(2).min();

        double chi2 = 0;
        if (simtrack.n_rows > 10 and (zlenSim - zlenTrue) >= -0.05) {
            arma::mat simtrackMat = prepareSimulatedTrackMatrix(simtrack);
            arma::mat devs = findDeviations(simtrackMat, trueValues);
            arma::vec validDevs = dropNaNs(arma::sum(arma::square(devs), 1));
            if (!validDevs.is_empty()) {
                chi2 = arma::median(validDevs);
            }
            else {
                chi2 = 200;
            }
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
                                           const arma::mat& trueValues, const unsigned numIters, const unsigned numPts,
                                           const double redFactor) const
    {
        arma::uword numVars = ctr0.n_rows;

        arma::vec mins = ctr0 - sigma0 / 2;
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
                    chi2 = runTrack(p, trueValues);
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
