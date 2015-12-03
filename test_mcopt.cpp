#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "mcopt.h"
#include "arma_include.h"

TEST_CASE("Calculated deviations are correct", "[deviations]")
{
    arma::mat A (20, 3);

    A.col(0) = arma::linspace<arma::vec>(0, 20, 20);
    A.col(1) = arma::linspace<arma::vec>(0, 20, 20);
    A.col(2) = arma::linspace<arma::vec>(0, 20, 20);

    SECTION("Two equal arrays have zero deviation")
    {
        arma::mat B = A;
        arma::mat devs = findDeviations(A, B);

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
        arma::mat devs = findDeviations(A, B);

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
        arma::mat devs = findDeviations(A, B);

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

    Conditions cond;
    cond.massNum = 1;
    cond.chargeNum = 1;
    cond.eloss = eloss;
    cond.efield = arma::vec3({0, 0, 1e3});
    cond.bfield = arma::vec3({0, 0, 1});

    arma::vec ctr0 = {0, 0, 0.9, 1, 0, arma::datum::pi, 0};
    arma::vec sigma = {0, 0, 0.001, 0.5, 0.2, 0.2, 0.1};

    SECTION("Minimizer doesn't throw")
    {
        REQUIRE_NOTHROW(
            MCminimize(ctr0, sigma, trueValues, cond, 2, 50, 0.8);
        );
    }

    SECTION("Minimizer doesn't throw when eloss is tiny")
    {
        eloss = arma::conv_to<std::vector<double>>::from(arma::randu<arma::vec>(100));
        cond.eloss = eloss;

        ctr0(3) = 10;  // raise the energy

        REQUIRE_NOTHROW(
            MCminimize(ctr0, sigma, trueValues, cond, 2, 50, 0.8);
        );
    }
}
