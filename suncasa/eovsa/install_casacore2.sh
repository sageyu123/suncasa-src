#!/bin/bash
# Install casacore v2.4.1 (last v2 release) to /usr/local/casacore2
# so that the EOVSA WSClean compatibility wrapper can find libcasa_*.so.2
#
# Run as root or with sudo on the pipeline server:
#   sudo bash install_casacore2.sh

set -e

INSTALL_PREFIX="/usr/local/casacore2"
BUILD_DIR="/tmp/casacore2_build"
CASACORE_VERSION="v2.4.1"

echo "=== Installing casacore ${CASACORE_VERSION} to ${INSTALL_PREFIX} ==="

# Install build dependencies
apt-get update
apt-get install -y \
    cmake \
    g++ \
    gfortran \
    flex \
    bison \
    libblas-dev \
    liblapack-dev \
    libcfitsio-dev \
    wcslib-dev \
    libfftw3-dev \
    libhdf5-dev \
    libboost-python-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libreadline-dev \
    libncurses-dev \
    python3-dev \
    python3-numpy \
    git

# Clean previous build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Clone casacore v2.4.1
echo "=== Cloning casacore ${CASACORE_VERSION} ==="
git clone --depth 1 --branch "${CASACORE_VERSION}" \
    https://github.com/casacore/casacore.git

cd casacore

# Build
mkdir build && cd build
echo "=== Configuring ==="
cmake .. \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON=OFF \
    -DBUILD_PYTHON3=OFF \
    -DBUILD_TESTING=OFF \
    -DUSE_OPENMP=ON \
    -DUSE_HDF5=ON \
    -DUSE_FFTW3=ON \
    -DDATA_DIR=/usr/share/casacore/data

echo "=== Building (this may take 10-20 minutes) ==="
make -j$(nproc)

echo "=== Installing to ${INSTALL_PREFIX} ==="
make install

# Verify
echo ""
echo "=== Installed libraries ==="
ls -la "${INSTALL_PREFIX}/lib/"libcasa_*.so.2*

echo ""
echo "=== Done. Now test with: ==="
echo "  LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib /usr/local/bin/wsclean-eovsa-bin --version"
echo ""
echo "Or use the wrapper script: /usr/local/bin/wsclean-eovsa"
echo "Legacy binary expected by the wrapper: /usr/local/bin/wsclean-eovsa-bin"

# Clean up build dir
rm -rf "${BUILD_DIR}"
