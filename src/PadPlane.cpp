#include "PadPlane.h"

namespace mcopt {

    PadPlane::PadPlane(const arma::Mat<uint16_t>& lt_, const double xLB_, const double xDelta_,
                       const double yLB_, const double yDelta_, const double rotAngle_)
    : xLowerBound(xLB_), yLowerBound(yLB_), xDelta(xDelta_), yDelta(yDelta_), lookupTable(lt_),
      sinRotAngle(sin(rotAngle_)), cosRotAngle(cos(rotAngle_)), padCoords(generatePadCoordinates(rotAngle_))
    {
        xUpperBound = xLowerBound + lookupTable.n_cols * xDelta;
        yUpperBound = yLowerBound + lookupTable.n_rows * yDelta;
    }

    uint16_t PadPlane::getPadNumberFromCoordinates(const double x, const double y) const
    {
        const double cosAng =  cosRotAngle;  // cos(-rotAngle)
        const double sinAng = -sinRotAngle;  // sin(-rotAngle)
        double rotX = cosAng * x - sinAng * y;
        double rotY = sinAng * x + cosAng * y;

        double xPos = std::round((rotX - xLowerBound) / xDelta);
        double yPos = std::round((rotY - yLowerBound) / yDelta);

        if (xPos < 0 || yPos < 0 || xPos >= lookupTable.n_cols || yPos >= lookupTable.n_rows) {
            return 20000;
        }
        return lookupTable(static_cast<arma::uword>(xPos), static_cast<arma::uword>(yPos));
    }

    static inline std::vector<std::vector<double>> create_triangle(const double x_offset, const double y_offset,
                                                                   const double side, const double orient)
    {
        return {{x_offset, y_offset},
                {x_offset + side / 2, y_offset + orient * side * std::sqrt(3.) / 2.},
                {x_offset + side, y_offset}};
    }

    static inline double roundToEven(const double v)
    {
        return v >= 0.0 ? std::floor(v + 0.5) : std::ceil(v - 0.5);
    }

    std::vector<std::vector<std::vector<double>>>
    PadPlane::generatePadCoordinates(const double rotation_angle)
    {
        double small_z_spacing = 2 * 25.4 / 1000;
        double small_tri_side = 184. * 25.4 / 1000;
        double umega_radius = 10826.772 * 25.4 / 1000;
        double beam_image_radius = 4842.52 * 25.4 / 1000;
        size_t pad_index = 0;

        double small_x_spacing = 2 * small_z_spacing / std::sqrt(3);
        double small_y_spacing = small_x_spacing * std::sqrt(3);
        double dotted_s_tri_side = 4. * small_x_spacing + small_tri_side;
        double dotted_s_tri_hi = dotted_s_tri_side * std::sqrt(3) / 2;
        double dotted_l_tri_side = 2. * dotted_s_tri_side;
        double dotted_l_tri_hi = dotted_l_tri_side * std::sqrt(3) / 2;
        double large_x_spacing = small_x_spacing;
        double large_y_spacing = small_y_spacing;
        double large_tri_side = dotted_l_tri_side - 4 * large_x_spacing;
        double large_tri_hi = dotted_l_tri_side * std::sqrt(3) / 2;
        double row_len_s = std::pow(2, std::ceil(std::log(beam_image_radius / dotted_s_tri_side) / std::log(2)));
        double row_len_l = std::floor(umega_radius / dotted_l_tri_hi);

        double xoff = 0;
        double yoff = 0;

        // Create half a circle

        std::vector<std::vector<std::vector<double>>> pads (10240,
            std::vector<std::vector<double>>(3, std::vector<double>(2, 0))); // Initializes the results

        for (int j = 0; j < row_len_l; j++) {
            double pads_in_half_hex = 0;
            double pads_in_hex = 0;
            double row_length = std::abs(std::sqrt(std::pow(umega_radius, 2) - std::pow(j * dotted_l_tri_hi + dotted_l_tri_hi / 2.0, 2)));

            if (j < row_len_s / 2.0) {
                pads_in_half_hex = (2 * row_len_s - 2 * j) / 4.0;
                pads_in_hex = 2 * row_len_s - 1 - 2 * j;
            }

            double pads_in_half_row = row_length / dotted_l_tri_side;
            int pads_out_half_hex = static_cast<int>(roundToEven(2 * (pads_in_half_row - pads_in_half_hex)));
            int pads_in_row = static_cast<int>(2 * pads_out_half_hex + 4 * pads_in_half_hex - 1);

            int ort = 1;

            for (int i = 0; i < pads_in_row; i++) {
                if (i == 0) {
                    if (j % 2 == 0) {
                        ort = -1;
                    }
                    if (((pads_in_row - 1) / 2) % 2 == 1) {
                        ort = -ort;
                    }
                }
                else {
                    ort = -ort;
                }

                double pad_x_off = -(pads_in_half_hex + pads_out_half_hex / 2.0) * dotted_l_tri_side
                                   + i * dotted_l_tri_side / 2 + 2 * large_x_spacing + xoff;

                if ((i < pads_out_half_hex) || (i > pads_in_hex + pads_out_half_hex - 1) || (j > row_len_s / 2.0 - 1)) {
                    // Outside hex
                    double pad_y_off = j * dotted_l_tri_hi + large_y_spacing + yoff;
                    if (ort == -1) {
                        pad_y_off += large_tri_hi;
                    }
                    pads.at(pad_index) = create_triangle(pad_x_off, pad_y_off, large_tri_side, ort);
                    pad_index += 1;
                }
                else {
                    // inside hex, make 4 small triangles
                    double pad_y_off = j * dotted_l_tri_hi + large_y_spacing + yoff;

                    if (ort == -1) {
                        pad_y_off = j * dotted_l_tri_hi + 2 * dotted_s_tri_hi - small_y_spacing + yoff;
                    }

                    pads.at(pad_index) = create_triangle(pad_x_off, pad_y_off, small_tri_side, ort);
                    pad_index += 1;

                    double tmp_pad_x_off = pad_x_off + dotted_s_tri_side / 2.0;
                    double tmp_pad_y_off = pad_y_off + ort * dotted_s_tri_hi - 2 * ort * small_y_spacing;
                    pads.at(pad_index) = create_triangle(tmp_pad_x_off, tmp_pad_y_off, small_tri_side, -ort);
                    pad_index += 1;

                    tmp_pad_y_off = pad_y_off + ort * dotted_s_tri_hi;
                    pads.at(pad_index) = create_triangle(tmp_pad_x_off, tmp_pad_y_off, small_tri_side, ort);
                    pad_index += 1;

                    tmp_pad_x_off = pad_x_off + dotted_s_tri_side;
                    pads.at(pad_index) = create_triangle(tmp_pad_x_off, pad_y_off, small_tri_side, ort);
                    pad_index += 1;
                }
            }
        }

        // Create symmetric pads
        for (size_t i = 0; i < pad_index; i++) {
            pads.at(pad_index + i).at(0) = {pads.at(i).at(0).at(0), -pads.at(i).at(0).at(1)};
            pads.at(pad_index + i).at(1) = {pads.at(i).at(1).at(0), -pads.at(i).at(1).at(1)};
            pads.at(pad_index + i).at(2) = {pads.at(i).at(2).at(0), -pads.at(i).at(2).at(1)};
        }

        const double cosang = std::cos(rotation_angle);
        const double sinang = std::sin(rotation_angle);

        for (auto& pad : pads) {
            for (auto& point : pad) {
                auto x = point.at(0);
                auto y = point.at(1);
                point.at(0) = cosang * x - sinang * y;
                point.at(1) = sinang * x + cosang * y;
            }
        }

        return pads;
    }

    std::vector<double> PadPlane::getPadCenter(const size_t padNum) const
    {
        const auto& pad = padCoords.at(padNum);
        const auto& v1 = pad.at(0);
        const auto& v2 = pad.at(1);
        const auto& v3 = pad.at(2);

        return {(v1.at(0) + v2.at(0) + v3.at(0)) / 3, (v1.at(1) + v2.at(1) + v3.at(1)) / 3};
    }

#ifdef HAVE_HDF5
    arma::Mat<uint16_t> readLUT(const std::string& path)
    {
        H5::H5File file (path.c_str(), H5F_ACC_RDONLY);
        H5::DataSet ds = file.openDataSet("LUT");

        H5::DataSpace filespace = ds.getSpace();
        int ndims = filespace.getSimpleExtentNdims();
        if (ndims != 2) {
            throw std::runtime_error("LUT in HDF5 file had the wrong number of dimensions");
        }
        hsize_t dims[2] = {1, 1};
        filespace.getSimpleExtentDims(dims);

        H5::DataSpace memspace (ndims, dims);

        arma::Mat<uint16_t> res (dims[0], dims[1]);

        ds.read(res.memptr(), H5::PredType::NATIVE_UINT16, memspace, filespace);

        // NOTE: Armadillo stores data in column-major order, while HDF5 uses
        // row-major ordering. Above, we read the data directly from HDF5 into
        // the arma matrix, so it was implicitly transposed. The next function
        // fixes this problem.
        arma::inplace_trans(res);
        return res;
    }
#endif /* defined HAVE_HDF5 */
}
