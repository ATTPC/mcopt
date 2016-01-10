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
    arma::uword size = 512;
    double height = 20.5;
    arma::uword leftEdge = 30;
    arma::uword width = 20;

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

TEST_CASE("Trigger class works", "[trigger]")
{
    unsigned padThreshMSB = 1;
    unsigned padThreshLSB = 2;
    double trigWidth = 200e-9;
    unsigned multThresh = 10000;
    unsigned multWindow = 300;
    double writeCk = 12.5e6;
    double gain = 120e-15;
    double discrFrac = 0.175;

    mcopt::PadMap padmap;
    padmap.insert(0, 0, 0, 0, 0);
    padmap.insert(1, 0, 0, 0, 1);

    mcopt::Trigger trig (padThreshMSB, padThreshLSB, trigWidth, multThresh, multWindow,
                         writeCk, gain, discrFrac, padmap);

    SECTION("Pad threshold is right")
    {
        double threshSetting = (padThreshMSB << 4) + padThreshLSB;
        CAPTURE(threshSetting);
        double fullScale = gain / 4096 / 1.602176565e-19;;
        CAPTURE(fullScale);
        double expectedThresh = (threshSetting / 128) * discrFrac * 4096 * fullScale;
        CAPTURE(trig.getPadThresh());
        CAPTURE(expectedThresh);
        REQUIRE(std::abs(expectedThresh - trig.getPadThresh()) < 1e-6);
    }

    SECTION("Trigger signals are valid")
    {
        std::map<mcopt::pad_t, mcopt::Peak> peaks;
        peaks.emplace(0, mcopt::Peak{10, 1});
        peaks.emplace(1, mcopt::Peak{10, 20000});

        CAPTURE(trig.getPadThresh());

        auto res = trig.findTriggerSignals(peaks);

        SECTION("Signals below pad threshold do not trigger")
        {
            const arma::vec& v = res.at(0);
            CAPTURE(arma::nonzeros(v));
            REQUIRE(arma::accu(v) < 1e-6);
        }

        SECTION("Signals above pad threshold do trigger")
        {
            const arma::vec& v = res.at(1);
            CAPTURE(arma::nonzeros(v));
            REQUIRE(arma::accu(v) > 0);
        }

        SECTION("Trigger pulse shape is right")
        {
            const arma::vec& pulse = res.at(1);
            arma::vec nz = arma::nonzeros(pulse);
            double width = nz.n_rows;
            CAPTURE(width);
            unsigned long expWidth = std::lround(trigWidth * writeCk);
            CAPTURE(expWidth);
            REQUIRE(std::abs(width - expWidth) < 1e-6);

            for (auto v : nz) {
                REQUIRE(std::abs(v - 48) < 1e-6);
            }
        }
    }
}
