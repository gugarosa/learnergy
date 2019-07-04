import os

import pytest

from recogners.utils import loader


def test_load_opf():
    opf_file = os.path.abspath('./tests/data/opf_format.txt')

    try:
        X, Y = loader.load_opf('')
    except:
        X, Y = loader.load_opf(opf_file)

    assert len(X) == 5
    assert len(Y) == 5
