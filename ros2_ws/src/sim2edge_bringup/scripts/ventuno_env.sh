# Source this before building or running the stack on the Ventuno Q.
#
#     source src/sim2edge_bringup/scripts/ventuno_env.sh [--base]
#
# Every line here is load-bearing and none of them is guessable. The reasoning
# is in docs/daily/2026_08_04_log.md; the short version is below.

# ── QNN SDK ───────────────────────────────────────────────────────────────────
export QNN_SDK_ROOT=${QNN_SDK_ROOT:-/opt/qcom/aistack/qairt/2.48.0.260626}

# The `aarch64-oe-linux-gcc11.2` directory must come FIRST.
#
# `aarch64-ubuntu-gcc9.4` is the obvious choice on Ubuntu and it is the wrong
# one: its libQnnHtp.so rejects this silicon and it ships no V75 stub at all.
# The symptom is
#     <E> Dsp startup: Unsupported SoC model (SnapdragonModel): 82
#     <E> Invalid dsp arch. Cannot determine stub
# where the 82 is hardware auto-detection, not the --soc-model argument -- so
# changing --soc-model does not change it. 82 is QNN_SOC_MODEL_QCS8300, and QNN
# has no QCS8275 entry, so this chip correctly reports its QCS8300-family
# number and the Ubuntu variant's DSP startup table simply lacks it.
#
# The gcc9.4 directory stays on the path second because some tools (notably
# qnn-platform-validator) only exist there.
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2:$QNN_SDK_ROOT/lib/aarch64-ubuntu-gcc9.4${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Semicolon-separated, and unset by default on this board.
export ADSP_LIBRARY_PATH="$QNN_SDK_ROOT/lib/hexagon-v75/unsigned;/usr/lib/rfsa/adsp;/dsp"

# ── ExecuTorch ────────────────────────────────────────────────────────────────
# `build-x86` is a native aarch64 build. The directory name is misleading and is
# inherited from the board's bring-up; do not "fix" it.
export EXECUTORCH_INSTALL_DIR=${EXECUTORCH_INSTALL_DIR:-/opt/executorch/build-x86}

# libqnn_executorch_backend.so is a shared library and lives here. The installed
# binaries carry it in their RPATH (CMAKE_INSTALL_RPATH_USE_LINK_PATH), so this
# is belt and braces -- but a build tree that has not been installed, or a
# binary run straight out of build/, still needs it.
export LD_LIBRARY_PATH=$EXECUTORCH_INSTALL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# The AOT toolchain, for lowering a checkpoint on the board. ExecuTorch is built
# at /opt/executorch but never installed into ~/.venv/executorch, and PYTHONPATH
# is the whole fix -- nothing needs rebuilding.
export PYTHONPATH=/opt${PYTHONPATH:+:$PYTHONPATH}

# ── Create 3 over the USB gadget ──────────────────────────────────────────────
# The board has several NICs and Wi-Fi holds the default route, so Fast DDS
# joins discovery multicast on Wi-Fi and never hears the base -- even though
# tcpdump shows the base's SPDP arriving on usb0. ROS_DISCOVERY_SERVER must stay
# unset: this base uses simple/multicast discovery.
#
# Disabled by default because it also restricts discovery to loopback and usb0,
# which is not what you want while testing over ssh from another machine. Pass
# ``--base`` when driving the real base. Resolve the profile relative to this
# file so this works both from src/ and install/ before an overlay is sourced.
if [ "${1:-}" = "--base" ]; then
    _VENTUNO_ENV_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    export ROS_DOMAIN_ID=0
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export FASTRTPS_DEFAULT_PROFILES_FILE="$_VENTUNO_ENV_DIR/../config/fastdds_usb0.xml"
    unset ROS_DISCOVERY_SERVER
elif [ -n "${1:-}" ]; then
    echo "ventuno_env.sh: unknown argument: $1" >&2
    return 2 2>/dev/null || exit 2
fi

echo "ventuno env: QNN $QNN_SDK_ROOT"
echo "             ExecuTorch $EXECUTORCH_INSTALL_DIR"
echo "             validate the runtime before trusting any number:"
echo "               \$QNN_SDK_ROOT/bin/aarch64-ubuntu-gcc9.4/qnn-platform-validator --backend dsp --coreVersion --testBackend"
echo "             expect 'Hexagon Architecture V75' and 'Unit Test ... Passed'."
echo "             An HTP mismatch returns garbage, not an error."
if [ "${1:-}" = "--base" ]; then
    echo "             Create 3 DDS pinned to usb0: $FASTRTPS_DEFAULT_PROFILES_FILE"
fi
