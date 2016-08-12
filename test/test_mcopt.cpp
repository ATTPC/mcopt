#include "catch.hpp"

#include "mcopt.h"
#include <armadillo>
#include <cmath>

TEST_CASE("Calculated deviations are correct", "[deviations]")
{
    std::vector<double> eloss = arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(100000));
    std::vector<double> enVsZ = arma::conv_to<std::vector<double>>::from(arma::vec(1000, arma::fill::ones));

    unsigned massNum = 1;
    unsigned chargeNum = 1;
    arma::vec3 efield {0, 0, 1e3};
    arma::vec3 bfield {0, 0, 1};
    double ioniz = 10;
    arma::vec3 vd {0, 0, 10};
    double gain = 1;
    double tilt = 0;
    double shape = 200e-9;
    double clock = 12.5e6;
    double diffSigma = 0.5e-3;

    mcopt::Gas gas (eloss, enVsZ);
    mcopt::Tracker tracker (massNum, chargeNum, gas, efield, bfield);

    arma::Mat<mcopt::pad_t> mockLUT =
        arma::conv_to<arma::Mat<mcopt::pad_t>>::from(arma::round(arma::randu<arma::mat>(5600, 5600) * 10000));
    mcopt::PadPlane pads (mockLUT, -0.280, 0.0001, -0.280, 0.0001, 0);
    mcopt::EventGenerator evtgen (pads, vd, clock, shape, massNum, ioniz, gain, tilt, diffSigma);

    mcopt::MCminimizer minimizer (tracker, evtgen);

    arma::mat A (20, 4);

    A.col(0) = arma::linspace<arma::vec>(0, 20, 20);
    A.col(1) = arma::linspace<arma::vec>(0, 20, 20);
    A.col(2) = arma::linspace<arma::vec>(0, 20, 20);
    A.col(3) = arma::linspace<arma::vec>(0, 20, 20);

    SECTION("Two equal arrays have zero deviation")
    {
        arma::mat B = A;
        arma::mat devs = minimizer.findPositionDeviations(A, B);

        INFO("A = " << A);
        INFO("B = " << B);
        INFO("devs = " << devs);
        REQUIRE(arma::accu(arma::abs(devs)) < 1e-4);
    }

    SECTION("Add a constant to column 0")
    {
        const double c = 100;
        arma::mat B = A;
        B.col(0) += c;
        arma::mat devs = minimizer.findPositionDeviations(A, B);

        INFO("c = " << c);
        INFO("A = " << A);
        INFO("B = " << B);
        INFO("devs = " << devs);

        REQUIRE(arma::all(devs.col(0) * B.col(0).max() + c < 1e-6));
        REQUIRE(arma::all(devs.col(1) < 1e-6));
    }

    SECTION("Add a constant to column 1")
    {
        const double c = 100;
        arma::mat B = A;
        B.col(1) += c;
        arma::mat devs = minimizer.findPositionDeviations(A, B);

        INFO("c = " << c);
        INFO("A = " << A);
        INFO("B = " << B);
        INFO("devs = " << devs);

        REQUIRE(arma::all(devs.col(0) < 1e-6));
        REQUIRE(arma::all(devs.col(1) * B.col(1).max() + c < 1e-6));
    }
}

TEST_CASE("Minimizer works", "[minimizer]")
{
    arma::arma_rng::set_seed(12345);

    arma::mat expPos = arma::randu<arma::mat>(100, 4);
    arma::vec expMesh = arma::randu<arma::vec>(10240);

    std::vector<double> eloss = arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(100000));
    std::vector<double> enVsZ = arma::conv_to<std::vector<double>>::from(arma::vec(1000, arma::fill::ones));

    unsigned massNum = 1;
    unsigned chargeNum = 1;
    arma::vec3 efield {0, 0, 1e3};
    arma::vec3 bfield {0, 0, 1};
    double ioniz = 10;
    arma::vec3 vd {0, 0, 10};
    double gain = 1;
    double tilt = 0;
    double shape = 200e-9;
    double clock = 12.5e6;
    double diffSigma = 0.5e-3;

    arma::vec ctr0 = {0, 0, 0.9, 1, 0, arma::datum::pi, 0};
    arma::vec sigma = {0, 0, 0.001, 0.5, 0.2, 0.2, 0.1};

    mcopt::Gas gas (eloss, enVsZ);
    mcopt::Tracker tracker (massNum, chargeNum, gas, efield, bfield);

    arma::Mat<mcopt::pad_t> mockLUT =
        arma::conv_to<arma::Mat<mcopt::pad_t>>::from(arma::round(arma::randu<arma::mat>(5600, 5600) * 10000));
    mcopt::PadPlane pads (mockLUT, -0.280, 0.0001, -0.280, 0.0001, 0);
    mcopt::EventGenerator evtgen (pads, vd, clock, shape, massNum, ioniz, gain, tilt, diffSigma);

    SECTION("Minimizer doesn't throw")
    {
        mcopt::MCminimizer minimizer (tracker, evtgen);

        REQUIRE_NOTHROW(
            minimizer.minimize(ctr0, sigma, expPos, expMesh, 2, 50, 0.8);
        );
    }

    SECTION("Minimizer doesn't throw when eloss is tiny")
    {
        eloss = arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(100));
        mcopt::Gas gas (eloss, enVsZ);
        mcopt::Tracker tracker (massNum, chargeNum, gas, efield, bfield);
        mcopt::MCminimizer minimizer (tracker, evtgen);

        ctr0(3) = 10;  // raise the energy

        REQUIRE_NOTHROW(
            minimizer.minimize(ctr0, sigma, expPos, expMesh, 2, 50, 0.8);
        );
    }
}

TEST_CASE("dropNaNs function works", "[dropNaNs]")
{
    auto nan = arma::datum::nan;

    SECTION("NaNs are removed when present")
    {
        arma::vec testData = {nan, 1, 2, 3, 4, nan, 5, 6, nan, 7, 8, 9, nan, nan, nan};
        arma::vec answer = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        arma::vec result = mcopt::dropNaNs(testData);
        CAPTURE(result);

        REQUIRE(arma::accu(arma::abs(result - answer)) < 1e-6);
        REQUIRE_FALSE(result.has_nan());
    }

    SECTION("Nothing happens when NaNs are not present")
    {
        arma::vec testData = {1, 2, 3, 4, 5, 6};
        arma::vec result = mcopt::dropNaNs(testData);
        CAPTURE(result);

        REQUIRE(arma::accu(arma::abs(testData - result)) < 1e-6);
    }

    SECTION("Get empty vector when input is all NaNs")
    {
        arma::vec testData = {nan, nan, nan, nan};
        arma::vec result = mcopt::dropNaNs(testData);
        CAPTURE(result);

        REQUIRE(result.is_empty());
    }
}

TEST_CASE("replaceNaNs function works", "[replaceNaNs]")
{
    auto nan = arma::datum::nan;
    double rvl = 100;  // The replacement value

    SECTION("NaNs are replaced when present")
    {
        arma::vec testData = {nan, 1, 2, 3, 4, nan, 5, 6, nan, 7, 8, 9, nan, nan, nan};
        arma::vec answer =   {rvl, 1, 2, 3, 4, rvl, 5, 6, rvl, 7, 8, 9, rvl, rvl, rvl};
        arma::vec result = mcopt::replaceNaNs(testData, rvl);
        CAPTURE(result);

        REQUIRE(arma::accu(arma::abs(result - answer)) < 1e-6);
        REQUIRE_FALSE(result.has_nan());
    }

    SECTION("Nothing happens when NaNs are not present")
    {
        arma::vec testData = {1, 2, 3, 4, 5, 6};
        arma::vec result = mcopt::replaceNaNs(testData, rvl);
        CAPTURE(result);

        REQUIRE(arma::accu(arma::abs(testData - result)) < 1e-6);
    }

    SECTION("Get vector full of replacement value when input is all NaNs")
    {
        arma::vec testData = {nan, nan, nan, nan};
        arma::vec answer   = {rvl, rvl, rvl, rvl};
        arma::vec result = mcopt::replaceNaNs(testData, rvl);
        CAPTURE(result);

        REQUIRE(arma::accu(arma::abs(result - answer)) < 1e-6);
        REQUIRE_FALSE(result.has_nan());
    }
}

TEST_CASE("Parameter generator works", "[makeParams]")
{
    arma::arma_rng::set_seed(12345);
    arma::vec ctr = {1, 2, 3, 4, 5, 6};
    arma::vec sig = {1, 2, 3, 4, 5, 6};
    unsigned numSets = 1000;

    SECTION("Set of parameters follows uniform distribution")
    {
        arma::vec mins = {0, 0, 0, 0, 0, 0};
        arma::vec maxes = {100, 100, 100, 100, 100, 100};

        arma::mat params = mcopt::MCminimizer::makeParams(ctr, sig, numSets, mins, maxes);

        for (arma::uword j = 0; j < params.n_cols; j++) {
            double mean = arma::mean(params.col(j));
            double min = arma::min(params.col(j));
            double max = arma::max(params.col(j));

            CHECK(min < mean);
            CHECK(mean < max);
            CHECK(fabs(min - (ctr(j) - sig(j)/2)) < 1e-1);
            CHECK(fabs(max - (ctr(j) + sig(j)/2)) < 1e-1);
        }
    }
}
