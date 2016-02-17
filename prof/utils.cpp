#include "utils.h"

std::vector<double> readEloss(const std::string& path)
{
    H5::H5File file (path.c_str(), H5F_ACC_RDONLY);
    H5::DataSet ds = file.openDataSet("eloss");

    H5::DataSpace filespace = ds.getSpace();
    int ndims = filespace.getSimpleExtentNdims();
    assert(ndims == 1);
    hsize_t dim;
    filespace.getSimpleExtentDims(&dim);

    H5::DataSpace memspace (ndims, &dim);

    std::vector<double> res (dim);

    ds.read(res.data(), H5::PredType::NATIVE_DOUBLE, memspace, filespace);

    filespace.close();
    memspace.close();
    ds.close();
    file.close();
    return res;
}

arma::Mat<uint16_t> readLUT(const std::string& path)
{
    H5::H5File file (path.c_str(), H5F_ACC_RDONLY);
    H5::DataSet ds = file.openDataSet("LUT");

    H5::DataSpace filespace = ds.getSpace();
    int ndims = filespace.getSimpleExtentNdims();
    assert(ndims == 2);
    hsize_t dims[2] = {1, 1};
    filespace.getSimpleExtentDims(dims);

    H5::DataSpace memspace (ndims, dims);

    arma::Mat<uint16_t> res (dims[0], dims[1]);

    ds.read(res.memptr(), H5::PredType::NATIVE_UINT16, memspace, filespace);

    filespace.close();
    memspace.close();
    ds.close();
    file.close();

    // NOTE: Armadillo stores data in column-major order, while HDF5 uses
    // row-major ordering. Above, we read the data directly from HDF5 into
    // the arma matrix, so it was implicitly transposed. The next function
    // fixes this problem.
    arma::inplace_trans(res);
    return res;
}
