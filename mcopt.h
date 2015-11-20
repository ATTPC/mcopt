#ifndef MCOPT_H
#define MCOPT_H

#include "arma_include.h"
// #include <armadillo>
#include <vector>
#include <string>
#include <tuple>

struct State
{
    arma::vec::fixed<3> pos;
    arma::vec::fixed<3> mom;
    double en;
    double de;
};
typedef struct State State;

struct Conditions
{
    double massNum;
    double chargeNum;
    std::vector<double> eloss;
    arma::vec::fixed<3> efield;
    arma::vec::fixed<3> bfield;
};
typedef struct Conditions Conditions;

struct Track
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> time;
    std::vector<double> enu;
    std::vector<double> azi;
    std::vector<double> pol;
};
typedef struct Track Track;

class TrackingFailed : public std::exception
{
public:
    TrackingFailed(std::string m) : msg(m) {}
    const char* what() const noexcept { return msg.c_str(); }

private:
    std::string msg;
};

typedef std::tuple<arma::vec, arma::mat, arma::vec, arma::vec> MCminimizeResult;

void updateState(State& st, const Conditions& cond, const double tstep);
Track trackParticle(const double x0, const double y0, const double z0,
                    const double enu0,  const double azi0, const double pol0, const Conditions& cond);
arma::mat makeParams(const arma::vec ctr, const arma::vec sigma, const unsigned numSets,
                     const arma::vec mins, const arma::vec maxes);
double runTrack(const arma::vec& p, const arma::mat& trueValues, const Conditions& condBase);
arma::mat findDeviations(const arma::mat& simtrack, const arma::mat& expdata);
MCminimizeResult MCminimize(const arma::vec& ctr0, const arma::vec& sigma0,
                            const arma::mat& trueValues, const Conditions& cond,
                            const unsigned numIters, const unsigned numPts,
                            const double redFactor);

#endif /* def MCOPT_H */
