# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Pytest configuration for E2E tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options for E2E tests."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests that require API access",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test requiring API access"
    )


def pytest_collection_modifyitems(config, items):
    """Skip E2E tests unless --run-e2e is specified."""
    if config.getoption("--run-e2e"):
        # --run-e2e given: do not skip e2e tests
        return
    
    skip_e2e = pytest.mark.skip(reason="E2E tests require --run-e2e flag")
    for item in items:
        # Skip only tests that don't have skipif(False) - CLI tests can run without API
        if "e2e" in item.keywords or "tests/e2e" in str(item.fspath):
            # Don't skip CLI tests
            if "CLI" not in item.nodeid:
                item.add_marker(skip_e2e)
