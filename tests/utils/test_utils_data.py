from typing import Literal

import pytest

import numpy as np

from moscot.utils._data import MarkerGenes, TranscriptionFactors


class TestTranscriptionFactor:
    @pytest.mark.parameterize("organism", ["human", "mouse", "drosophila"])
    def test_load_data(self, organism: Literal["human", "mouse", "drosophila"]):
        tfs = TranscriptionFactors.transcription_factors(organism=organism)
        assert isinstance(tfs, list)
        assert len(tfs) > 0
        is_str = [isinstance(el, str) for el in tfs]
        assert np.sum(is_str) == len(tfs)


class TestMarkerGenes:
    @pytest.mark.parameterize("organism", ["human", "mouse"])
    def test_proliferation_markers(self, organism: Literal["human", "mouse"]):
        mgs = MarkerGenes.proliferation_markers(organism=organism)
        assert isinstance(mgs, list)
        assert len(mgs) > 0
        is_str = [isinstance(el, str) for el in mgs]
        assert np.sum(is_str) == len(mgs)

    @pytest.mark.parameterize("organism", ["human", "mouse"])
    def test_apoptosis_markers(self, organism: Literal["human", "mouse"]):
        mgs = MarkerGenes.apoptosis_markers(organism=organism)
        assert isinstance(mgs, list)
        assert len(mgs) > 0
        is_str = [isinstance(el, str) for el in mgs]
        assert np.sum(is_str) == len(mgs)
