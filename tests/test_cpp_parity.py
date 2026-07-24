"""Python vs C++ preprocessing parity, run as a test.

This is the automated form of the mandatory pre-robot gate in §6.2 of the
implementation plan. A calibration/runtime preprocessing mismatch was the single
most expensive failure mode in both reference projects, and its defining
property is that it produces confident wrong actions rather than an error. The
only defence is to compare the two implementations mechanically, on adversarial
data, every time the code changes.

The C++ side is compiled here directly with g++ rather than through colcon: the
preprocessing translation unit depends on nothing but the standard library, so
the gate runs on any machine with a compiler and does not need ROS, ExecuTorch,
or the board.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from sim2edge.spec import DEFAULT_SPEC  # noqa: E402

S = DEFAULT_SPEC
PKG = REPO / "ros_ws/src/nav_policy_node"

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None,
    reason="g++ is required to build the C++ preprocessing for the parity gate")


@pytest.fixture(scope="module")
def parity_binary(tmp_path_factory):
    """Compile preprocess.cpp + preprocess_parity.cpp exactly as the node does."""
    out = tmp_path_factory.mktemp("parity") / "preprocess_parity"
    result = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
         "-I", str(PKG / "include"),
         str(PKG / "src/preprocess.cpp"),
         str(PKG / "src/preprocess_parity.cpp"),
         "-o", str(out)],
        capture_output=True, text=True)
    assert result.returncode == 0, f"C++ build failed:\n{result.stderr}"
    return out


def test_preprocessing_implementations_agree(parity_binary, tmp_path):
    """The gate itself.

    The samples deliberately include the cases that break naive
    implementations -- NaN and negative depth, all-invalid scans, values outside
    the clip range, and a scan reported over [0, 2pi) instead of [-pi, pi).
    Parity on clean data would prove almost nothing; the implementations diverge
    precisely on how they handle garbage.
    """
    from parity_check import (
        check_preprocessing,
        make_samples,
        python_reference,
        read_outputs,
        write_inputs,
    )

    samples = make_samples(24, S, seed=3)
    in_path = tmp_path / "in.bin"
    out_path = tmp_path / "cpp.bin"
    write_inputs(str(in_path), samples, S)

    result = subprocess.run([str(parity_binary), str(in_path), str(out_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    py = python_reference(samples, S)
    cpp = read_outputs(str(out_path), S)
    assert len(py) == len(cpp)

    for name in S.obs.actor_inputs:
        worst = max(float(np.max(np.abs(a[name].ravel() - b[name].ravel())))
                    for a, b in zip(py, cpp))
        assert worst <= 1e-5, (
            f"'{name}' differs by {worst:.3e} between "
            f"sim2edge/preprocess.py and preprocess.cpp")

    assert check_preprocessing(samples, str(out_path), S)


def test_cpp_compiles_the_current_spec_fingerprint(parity_binary, tmp_path):
    """The compiled-in fingerprint must match the current spec, or the ROS node
    would refuse to load a correctly exported model."""
    in_path = tmp_path / "in.bin"
    from parity_check import make_samples, write_inputs

    write_inputs(str(in_path), make_samples(1, S, seed=0), S)
    result = subprocess.run(
        [str(parity_binary), str(in_path), str(tmp_path / "out.bin")],
        capture_output=True, text=True)
    assert S.fingerprint() in result.stdout, (
        f"C++ reports a different fingerprint than {S.fingerprint()}; "
        f"run tools/gen_spec_header.py")
