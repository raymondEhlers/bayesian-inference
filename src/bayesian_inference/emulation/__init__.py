"""
Emulation module for Bayesian Inference.

This module provides functionality to train and call emulators for a given analysis run.

The main functionalities are:
 - fit_emulators() performs PCA, fits an emulator to each PC, and writes the emulator to file
 - predict() construct mean, std of emulator for a given set of parameter values

A configuration class EmulationConfig provides simple access to emulation settings.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""
from __future__ import annotations

from bayesian_inference.emulation.base import (  # noqa: F401
    EmulationConfig,
    GroupedEmulationConfig,
    fit_emulators,
    predict,
)
