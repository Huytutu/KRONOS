"""Shared pytest configuration — register custom marks."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU and real model weights")
