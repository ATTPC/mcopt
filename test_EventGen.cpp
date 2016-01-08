#include "catch.hpp"
#include "EventGen.h"
#include <armadillo>

TEST_CASE("Calibration and uncalibration work", "[eventGenerator]")
{
    mcopt::Track tr;
    for (int i = 0; i < 512; i++) {
        tr.append(-200+i, -200+i, i, 0, 0, 0, 0);
    }
    double clock = 10;
    arma::vec vd = {0, 0.5, 1};

    SECTION("Calibration works")
    {
        arma::mat orig_data = tr.getMatrix().cols(0, 2);
        arma::mat cal = mcopt::calibrate(tr, vd, clock);

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
        auto cal = mcopt::calibrate(tr, vd, clock);
        mcopt::Track tr2;
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

TEST_CASE("Square wave function works", "[trigger]")
{
    int size = 512;
    double height = 20.5;
    int leftEdge = 30;
    int width = 20;

    arma::vec wave = mcopt::squareWave(size, leftEdge, width, height);

    SECTION("Values are correct")
    {
        for (arma::uword i = 0; i < size; i++) {
            if (i < leftEdge || i >= leftEdge + width) {
                REQUIRE(wave(i) - 0 < 1e-6);
            }
            else {
                REQUIRE(wave(i) - height < 1e-6);
            }
        }
    }
}
