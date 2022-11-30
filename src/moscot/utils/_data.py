from typing import Literal, Mapping, Sequence
import os

import pandas as pd


class TranscriptionFactors:
    """
    Taken from https://resources.aertslab.org/cistarget/tf_lists/.

    modified on webpage: 2022-04-27
    accessed: 30/11/2022.

    """

    data_dir = "../data"
    _transcription_factors: Mapping[str, Sequence[str]] = {
        "human": os.path.join(data_dir, "allTFs_hg38.txt"),
        "mouse": os.path.join(data_dir, "allTFs_mm.txt"),
        "drosophila": os.path.join(data_dir, "allTFs_dmel.txt"),
    }

    @classmethod
    def _load(cls, organism: Literal["human", "mouse", "drosophila"]) -> Sequence[str]:
        return pd.read_csv(cls._transcription_factors[organism], sep=" ", header=None)[0].values

    @classmethod
    def transcription_factors(cls, organism: Literal["human", "mouse", "drosophila"]) -> Sequence[str]:
        """Get transcription factors for ``organism``."""
        try:
            return cls._load(organism)
        except KeyError:
            raise NotImplementedError(f"Transcription factors for `{organism}` are not yet implemented.") from None
