#include "catch.hpp"

#include "mcopt.h"
#include <armadillo>
#include <cmath>

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
