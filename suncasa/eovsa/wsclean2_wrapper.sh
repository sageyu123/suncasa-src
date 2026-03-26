#!/bin/bash
# Wrapper script to run the legacy WSClean binary with casacore v2 libraries
# Install to /usr/local/bin/wsclean-eovsa:
#   sudo cp wsclean2_wrapper.sh /usr/local/bin/wsclean-eovsa
#   sudo chmod +x /usr/local/bin/wsclean-eovsa

CASACORE2_LIB="/usr/local/casacore2/lib"
WSCLEAN_BIN="/usr/local/bin/wsclean-eovsa-bin"

if [ ! -d "${CASACORE2_LIB}" ]; then
    echo "Error: casacore2 not found at ${CASACORE2_LIB}" >&2
    echo "Run install_casacore2.sh first." >&2
    exit 1
fi

if [ ! -x "${WSCLEAN_BIN}" ]; then
    echo "Error: legacy WSClean binary not found at ${WSCLEAN_BIN}" >&2
    exit 1
fi

# Fail fast if the wrapper accidentally points at itself.
if [ "${WSCLEAN_BIN}" = "$0" ] || [ "$(readlink -f "${WSCLEAN_BIN}")" = "$(readlink -f "$0")" ]; then
    echo "Error: wrapper is pointing at itself: ${WSCLEAN_BIN}" >&2
    echo "Reinstall the wrapper and keep the legacy binary as /usr/local/bin/wsclean-eovsa-bin." >&2
    exit 1
fi

# Prepend casacore2 libs so .so.2 is found before .so.4
export LD_LIBRARY_PATH="${CASACORE2_LIB}:${LD_LIBRARY_PATH}"

exec "${WSCLEAN_BIN}" "$@"
