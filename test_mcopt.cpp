#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "mcopt.h"
#include <armadillo>

TEST_CASE("Calculated deviations are correct", "[deviations]")
{
    arma::mat A (20, 3);

    A.col(0) = arma::linspace<arma::vec>(0, 20, 20);
    A.col(1) = arma::linspace<arma::vec>(0, 20, 20);
    A.col(2) = arma::linspace<arma::vec>(0, 20, 20);

    SECTION("Two equal arrays have zero deviation")
    {
        arma::mat B = A;
        arma::mat devs = MCminimizer::findDeviations(A, B);

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
        arma::mat devs = MCminimizer::findDeviations(A, B);

        INFO("c = " << c);
        INFO("A = " << A);
        INFO("B = " << B);
        INFO("devs = " << devs);

        REQUIRE(arma::all(devs.col(0) + c < 1e-6));
        REQUIRE(arma::all(devs.col(1) < 1e-6));
    }

    SECTION("Add a constant to column 1")
    {
        const double c = 100;
        arma::mat B = A;
        B.col(1) += c;
        arma::mat devs = MCminimizer::findDeviations(A, B);

        INFO("c = " << c);
        INFO("A = " << A);
        INFO("B = " << B);
        INFO("devs = " << devs);

        REQUIRE(arma::all(devs.col(0) < 1e-6));
        REQUIRE(arma::all(devs.col(1) + c < 1e-6));
    }
}

TEST_CASE("Minimizer works", "[minimizer]")
{
    arma::arma_rng::set_seed(12345);

    arma::mat trueValues = arma::randu<arma::mat>(100, 3);

    std::vector<double> eloss = arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(100000));

    unsigned massNum = 1;
    unsigned chargeNum = 1;
    arma::vec3 efield {0, 0, 1e3};
    arma::vec3 bfield {0, 0, 1};

    arma::vec ctr0 = {0, 0, 0.9, 1, 0, arma::datum::pi, 0};
    arma::vec sigma = {0, 0, 0.001, 0.5, 0.2, 0.2, 0.1};

    SECTION("Minimizer doesn't throw")
    {
        MCminimizer minimizer {massNum, chargeNum, eloss, efield, bfield};

        REQUIRE_NOTHROW(
            minimizer.minimize(ctr0, sigma, trueValues, 2, 50, 0.8);
        );
    }

    SECTION("Minimizer doesn't throw when eloss is tiny")
    {
        eloss = arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(100));
        MCminimizer minimizer {massNum, chargeNum, eloss, efield, bfield};

        ctr0(3) = 10;  // raise the energy

        REQUIRE_NOTHROW(
            minimizer.minimize(ctr0, sigma, trueValues, 2, 50, 0.8);
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
        arma::vec result = dropNaNs(testData);
        CAPTURE(result);

        REQUIRE(arma::accu(arma::abs(result - answer)) < 1e-6);
        REQUIRE_FALSE(result.has_nan());
    }

    SECTION("Nothing happens when NaNs are not present")
    {
        arma::vec testData = {1, 2, 3, 4, 5, 6};
        arma::vec result = dropNaNs(testData);
        CAPTURE(result);

        REQUIRE(arma::accu(arma::abs(testData - result)) < 1e-6);
    }

    SECTION("Get empty vector when input is all NaNs")
    {
        arma::vec testData = {nan, nan, nan, nan};
        arma::vec result = dropNaNs(testData);
        CAPTURE(result);

        REQUIRE(result.is_empty());
    }
}

TEST_CASE("Calibration and uncalibration work", "[eventGenerator]")
{
    Track tr;
    for (int i = 0; i < 512; i++) {
        tr.append(-200+i, -200+i, i, 0, 0, 0, 0);
    }
    double clock = 10;
    arma::vec vd = {0, 0.5, 1};

    SECTION("Calibration works")
    {
        arma::mat orig_data = tr.getMatrix().cols(0, 2);
        arma::mat cal = calibrate(tr, vd, clock);

        REQUIRE(orig_data.n_rows == cal.n_rows);
        REQUIRE(cal.n_cols == 3);

        for (arma::uword i = 0; i < cal.n_rows; i++) {
            // We need a minus here because of the sign of vd...
            double xExp = orig_data(i, 0) + (-vd(0)) * orig_data(i, 2) / clock * 10;
            double yExp = orig_data(i, 1) + (-vd(1)) * orig_data(i, 2) / clock * 10;
            double zExp =                 + (-vd(2)) * orig_data(i, 2) / clock * 10;

            CAPTURE(i);
            CAPTURE(orig_data(i, 0));
            CAPTURE(orig_data(i, 1));
            CAPTURE(orig_data(i, 2));
            CAPTURE(xExp);
            CAPTURE(yExp);
            CAPTURE(zExp);
            CAPTURE(cal(i, 0));
            CAPTURE(cal(i, 1));
            CAPTURE(cal(i, 2));

            REQUIRE(std::abs(cal(i, 0) - xExp) < 1e-6);
            REQUIRE(std::abs(cal(i, 1) - yExp) < 1e-6);
            REQUIRE(std::abs(cal(i, 2) - zExp) < 1e-6);
        }
    }

    SECTION("Round-trip operation is the identity")
    {
        auto cal = calibrate(tr, vd, clock);
        Track tr2;
        for (arma::uword i = 0; i < cal.n_rows; i++) {
            tr2.append(cal(i, 0), cal(i, 1), cal(i, 2), 0, 0, 0, 0);
        }
        auto uncal = uncalibrate(tr2, vd, clock);

        arma::mat expected = tr.getMatrix().cols(0, 2);
        CAPTURE(expected);
        CAPTURE(uncal);

        REQUIRE(arma::accu(arma::abs(uncal - expected)) < 1e-6);
    }
}
