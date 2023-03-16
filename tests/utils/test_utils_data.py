from typing import Literal

import pytest

import numpy as np

from moscot.utils.data import (
    apoptosis_markers,
    proliferation_markers,
    transcription_factors,
)


class TestTranscriptionFactor:
    @pytest.mark.parametrize("organism", ["human", "mouse", "drosophila"])
    def test_load_data(self, organism: Literal["human", "mouse", "drosophila"]):
        tfs = transcription_factors(organism=organism)
        assert isinstance(tfs, list)
        assert len(tfs) > 0
        is_str = [isinstance(el, str) for el in tfs]
        assert np.sum(is_str) == len(tfs)


class TestMarkerGenes:
    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_proliferation_markers(self, organism: Literal["human", "mouse"]):
        mgs = proliferation_markers(organism=organism)
        assert isinstance(mgs, list)
        assert len(mgs) > 0
        is_str = [isinstance(el, str) for el in mgs]
        assert np.sum(is_str) == len(mgs)

    @pytest.mark.parametrize("organism", ["human", "mouse"])
    def test_apoptosis_markers(self, organism: Literal["human", "mouse"]):
        mgs = apoptosis_markers(organism=organism)
        assert isinstance(mgs, list)
        assert len(mgs) > 0
        is_str = [isinstance(el, str) for el in mgs]
        assert np.sum(is_str) == len(mgs)
