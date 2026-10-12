import argparse
from pathlib import Path

import pytest

from solar_seg.eval import _parse_data_roots


def test_parse_data_roots_single():
    roots = _parse_data_roots(["bdappv=data/processed/bdappv"])
    assert list(roots.keys()) == ["bdappv"]
    assert roots["bdappv"] == Path("data/processed/bdappv").resolve()


def test_parse_data_roots_multiple():
    roots = _parse_data_roots(
        ["bdappv=data/processed/bdappv", "bradbury=data/processed/bradbury"]
    )
    assert list(roots.keys()) == ["bdappv", "bradbury"]
    assert roots["bradbury"] == Path("data/processed/bradbury").resolve()


def test_parse_data_roots_path_with_equals():
    roots = _parse_data_roots(["weird=a=b/c"])
    assert roots["weird"] == Path("a=b/c").resolve()


def test_parse_data_roots_invalid():
    with pytest.raises(argparse.ArgumentTypeError, match="Expected NAME=PATH"):
        _parse_data_roots(["no-equals-sign"])
