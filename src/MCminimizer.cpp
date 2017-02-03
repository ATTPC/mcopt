#include "MCminimizer.h"

namespace mcopt
{
    arma::mat MCminimizer::makeParams(const arma::vec& ctr, const arma::vec& sigma, const unsigned numSets,
                                      const arma::vec& mins, const arma::vec& maxes,
                                      const BeamLocationEstimator& beamloc)
    {
        const arma::uword numVars = ctr.n_rows;
        assert(sigma.n_rows == numVars);
        assert(mins.n_rows == numVars);
        assert(maxes.n_rows == numVars);

        arma::mat params = arma::randu(numSets, numVars);

        // Columns are x, y, z, en, azi, pol. Clamp z, en, azi, pol to sigma, so iterate from 2 to N.
        for (arma::uword i = 2; i < numVars; i++) {
            params.col(i) = arma::clamp(ctr(i) + (params.col(i) - 0.5) * sigma(i),
                                        mins(i), maxes(i));
        }

        // Now fix x and y from z
        params.col(0) = beamloc.findX(params.col(2));
        params.col(1) = beamloc.findY(params.col(2));

        return params;
    }

    MCminimizeResult MCminimizer::minimize(const arma::vec& ctr0, const arma::vec& sigma0,
                                           const arma::mat& expPos, const arma::vec& expHits,
                                           const unsigned numIters, const unsigned numPts, const double redFactor,
                                           const BeamLocationEstimator& beamloc) const
    {
        const arma::uword numVars = ctr0.n_rows;
        const arma::uword numChis = Chi2Set::numChis();

        assert(ctr0.n_elem == sigma0.n_elem);

        arma::vec mins = ctr0 - sigma0 / 2;
        if (mins(3) < 0) {
            mins(3) = 0;  // Prevent energy from being negative
        }
        arma::vec maxes = ctr0 + sigma0 / 2;
        arma::vec ctr = ctr0;
        arma::vec sigma = sigma0;

        MCminimizeResult res (numVars, numPts, numIters, numChis);

        for (unsigned i = 0; i < numIters; i++) {
            arma::mat params = makeParams(ctr, sigma, numPts, mins, maxes, beamloc);

            arma::mat chis = runTracks(params, expPos, expHits);

            arma::vec totChis = arma::sum(chis, 1);
            assert(totChis.n_rows == params.n_rows);

            arma::uword minChiLoc = 0;
            totChis.min(minChiLoc);

            ctr = arma::conv_to<arma::colvec>::from(params.row(minChiLoc));
            sigma *= redFactor;

            res.allParams.rows(i*numPts, (i+1)*numPts-1) = params;
            res.minChis.row(i) = chis.row(minChiLoc);
            res.goodParamIdx(i) = minChiLoc + i*numPts;
        }

        res.ctr = ctr;

        return res;
    }
}
