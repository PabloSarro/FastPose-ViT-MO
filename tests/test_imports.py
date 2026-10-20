"""Smoke test: the project modules and bop_toolkit import in this environment."""

import importlib

import pytest

MODULES = [
    "src.camera",
    "src.utils",
    "src.datasets",
    "src.engine",
    "src.losses",
    "src.metrics",
    "src.models",
    "src.vit_models",
    "src.optimizer",
    "src.scheduler",
    "bop_toolkit_lib.inout",
    "bop_toolkit_lib.misc",
    "bop_toolkit_lib.pose_error",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)
