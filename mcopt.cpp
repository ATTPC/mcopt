#include <cmath>
#include <cassert>
#include "mcopt.h"

static const double C_LGT = 299792458.0;
static const double E_CHG = 1.602176565e-19;
static const double P_MC2 = 938.272046;
static const double P_KG = 1.672621777e-27;
static const double POS_STEP = 1e-3;

static inline double threshold(const double value, const double threshmin)
{
    if (value < threshmin) {
        return threshmin;
    }
    else {
        return value;
    }
}

static inline double betaFactor(const double en, const double massNum)
{
    return (sqrt(en)*sqrt(en + 2*P_MC2*massNum)) / (en + massNum*P_MC2);
}

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

void Track::append(const double xi, const double yi, const double zi, const double timei,
                   const double enui, const double azii, const double poli)
{
    x.push_back(xi);
    y.push_back(yi);
    z.push_back(zi);
    time.push_back(timei);
    enu.push_back(enui);
    azi.push_back(azii);
    pol.push_back(poli);
}

arma::mat Track::getMatrix() const
{
    arma::mat result (numPts(), 7);
    result.col(0) = arma::vec(x);
    result.col(1) = arma::vec(y);
    result.col(2) = arma::vec(z);
    result.col(3) = arma::vec(time);
    result.col(4) = arma::vec(enu);
    result.col(5) = arma::vec(azi);
    result.col(6) = arma::vec(pol);

    return result;
}

size_t Track::numPts() const
{
    size_t N = x.size();

    assert(y.size() == N);
    assert(z.size() == N);
    assert(time.size() == N);
    assert(enu.size() == N);
    assert(azi.size() == N);
    assert(pol.size() == N);

    return N;
}

void MCminimizer::updateState(State& st, const double tstep) const
{
    return updateState(st, tstep, this->bfield);
}

void MCminimizer::updateState(State& st, const double tstep, const arma::vec3& bfield) const
{
    arma::vec3 mom_si = st.mom * 1e6 * E_CHG / C_LGT;

    double beta = betaFactor(st.en, massNum);
    if (beta < 1e-8) {
        st.pos = {0, 0, 0};
        st.mom = {0, 0, 0};
        st.en = 0;
        st.de = 0;
        return;
    }

    double gamma = 1 / sqrt(1 - beta*beta);

    arma::vec3 vel = mom_si / (gamma * massNum * P_KG);
    arma::vec3 force = chargeNum * E_CHG * (efield + arma::cross(vel, bfield));

    arma::vec3 new_mom_si = mom_si + force * tstep;
    arma::vec3 new_mom = new_mom_si * 1e-6 / E_CHG * C_LGT;

    // NOTE: I'm assuming the Lorentz force doesn't change the energy appreciably.

    arma::vec3 new_vel = new_mom_si / (gamma * massNum * P_KG);
    arma::vec3 new_pos = st.pos + vel * tstep;

    assert(st.en >= 0);  // This is assumed by the cast to unsigned long in the next line.
    size_t elossIdx = static_cast<size_t>(lround(st.en * 1000));  // Convert to keV, the index units

    double stopping;
    try {
        stopping = eloss.at(elossIdx);
    }
    catch (const std::out_of_range&) {
        throw TrackingFailed("Energy loss index out of range.");
    }
    double dpos = tstep * beta * C_LGT;
    double de = threshold(stopping*dpos, 0);
    assert(de >= 0);

    if (stopping <= 0 || de < 1e-10) {
        // position remains the same
        st.mom = {0, 0, 0};
        st.en = 0;
        st.de = de;
        return;
    }
    else {
        double en = threshold(st.en - de, 0);
        assert(en >= 0);

        double new_mom_mag = sqrt(pow(en + massNum*P_MC2, 2) - pow(massNum*P_MC2, 2));
        double old_mom_mag = arma::norm(new_mom);  // "old" means before eloss, in this case
        double mom_correction_factor = new_mom_mag / old_mom_mag;

        new_mom *= mom_correction_factor;
        new_mom_si *= mom_correction_factor;

        st.pos = new_pos;
        st.mom = new_mom;
        st.en = en;
        st.de = de;
        return;
    }
}

Track MCminimizer::trackParticle(const double x0, const double y0, const double z0,
                                 const double enu0,  const double azi0, const double pol0) const
{
    return trackParticle(x0, y0, z0, enu0, azi0, pol0, this->bfield);
}

Track MCminimizer::trackParticle(const double x0, const double y0, const double z0,
                                 const double enu0,  const double azi0, const double pol0,
                                 const arma::vec3& bfield) const
{
    const unsigned long maxIters = 10000;

    Track tr;
    State st;

    const double en0 = enu0 * massNum;
    double current_time = 0;

    double mom_mag = sqrt(pow(en0 + massNum*P_MC2, 2) - pow(massNum*P_MC2, 2));
    arma::vec::fixed<3> mom = {mom_mag * cos(azi0) * sin(pol0),
                               mom_mag * sin(azi0) * sin(pol0),
                               mom_mag * cos(pol0)};

    st.pos = {x0, y0, z0};
    st.mom = mom;
    st.en = en0;
    st.de = 0;

    tr.append(st.pos(0), st.pos(1), st.pos(2), current_time, st.en / massNum, azi0, pol0);

    for (unsigned long i = 1; i < maxIters; i++) {
        double beta = betaFactor(st.en, massNum);
        if (st.en < 1e-3 || beta < 1e-6) {
            break;
        }
        double tstep = POS_STEP / (beta * C_LGT);

        updateState(st, tstep, bfield);

        double azi = atan2(st.mom(1), st.mom(0));
        double pol = atan2(sqrt(pow(st.mom(0), 2) + pow(st.mom(1), 2)), st.mom(2));

        current_time += tstep;

        tr.append(st.pos(0), st.pos(1), st.pos(2), current_time, st.en / massNum, azi, pol);

        // double rad = arma::norm(st.pos);
        if (st.pos(2) < 0 || st.pos(2) > 1) { // || rad > 0.275) {
            break;
        }
    }

    return tr;
}

arma::mat MCminimizer::findDeviations(const arma::mat& simtrack, const arma::mat& expdata)
{
    // ASSUMPTION: matrices must be sorted in increasing Z order.

    arma::vec xInterp;
    arma::vec yInterp;

    arma::interp1(simtrack.col(2), simtrack.col(0), expdata.col(2), xInterp);
    arma::interp1(simtrack.col(2), simtrack.col(1), expdata.col(2), yInterp);

    return arma::join_horiz(xInterp - expdata.col(0), yInterp - expdata.col(1));
}

double MCminimizer::runTrack(const arma::vec& p, const arma::mat& trueValues) const
{
    arma::vec3 thisBfield = {0, 0, p(6)};

    Track tr = trackParticle(p(0), p(1), p(2), p(3), p(4), p(5), thisBfield);
    arma::mat simtrack = tr.getMatrix();

    double zlenSim = simtrack.col(2).max() - simtrack.col(2).min();
    double zlenTrue = trueValues.col(2).max() - trueValues.col(2).min();

    double chi2 = 0;
    if (simtrack.n_rows > 10 and (zlenSim - zlenTrue) >= -0.05) {
        arma::mat devs = findDeviations(simtrack, trueValues);
        arma::vec temp = dropNaNs(arma::sum(arma::square(devs), 1));
        if (!temp.is_empty()) {
            chi2 = arma::median(temp);
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

PadPlane::PadPlane(const arma::Mat<uint16_t>& lt, const double xLB, const double xDelta, const double yLB, const double yDelta)
: xLowerBound(xLB), yLowerBound(yLB), xDelta(xDelta), yDelta(yDelta), lookupTable(lt)
{
    xUpperBound = xLowerBound + lookupTable.n_cols * xDelta;
    yUpperBound = yLowerBound + lookupTable.n_rows * yDelta;
}

uint16_t PadPlane::getPadNumberFromCoordinates(const double x, const double y) const
{
    double xPos = std::round((x - xLowerBound) / xDelta);
    double yPos = std::round((y - yLowerBound) / yDelta);

    if (xPos < 0 || yPos < 0 || xPos >= lookupTable.n_cols || yPos >= lookupTable.n_rows) {
        return 20000;
    }
    return lookupTable(xPos, yPos);
}

arma::mat calibrate(const Track& tr, const arma::vec vd, const double clock)
{
    arma::mat trMat = tr.getMatrix();
    arma::mat pos = trMat.cols(0, 2);
    arma::mat result = pos + pos.col(2) * -vd.t() / clock * 10;
    result.col(2) -= pos.col(2);

    return result;
}

arma::mat uncalibrate(const Track& tr, const arma::vec vd, const double clock, const int offset)
{
    arma::mat trMat = tr.getMatrix();
    arma::mat pos = trMat.cols(0, 2);

    auto tbs = pos.col(2) * clock / (10 * -vd(2)) + offset;

    arma::mat result = pos - tbs * -vd.t() / clock * 10;
    result.col(2) = tbs;

    return result;
}

std::set<uint16_t> findHitPads(const PadPlane& pads, const Track& tr, const arma::vec& vd, const double clock)
{
    arma::mat uncal = uncalibrate(tr, vd, clock);
    std::set<uint16_t> result;

    for (arma::uword i = 0; i < uncal.n_rows; i++) {
        uint16_t pad = pads.getPadNumberFromCoordinates(uncal(i, 0), uncal(i, 1));
        if (pad != 20000) {
            result.insert(pad);
        }
    }

    return result;
}
