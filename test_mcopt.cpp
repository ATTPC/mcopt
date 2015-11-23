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
