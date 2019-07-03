import os

import pytest

from recogners.datasets import opf


def test_opf_dataset():
    opf_file = os.path.abspath('./tests/recogners/datasets/opf_data.txt')

    new_opf_dataset = opf.OPFDataset(path=opf_file)

    assert len(new_opf_dataset) == 5
    assert len(new_opf_dataset.X) == 5
    assert len(new_opf_dataset.Y) == 5
