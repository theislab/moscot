from pathlib import Path
from typing import List, Literal

__all__ = ["transcription_factors", "proliferation_markers", "apoptosis_markers"]


def transcription_factors(organism: Literal["human", "mouse", "drosophila"]) -> List[str]:
    """TODO.

    Taken from `here <https://resources.aertslab.org/cistarget/tf_lists/>`_.
    """
    if organism == "human":
        fname = "allTFs_hg38.txt"
    elif organism == "mouse":
        fname = "allTFs_mm.txt"
    elif organism == "drosophila":
        fname = "allTFs_dmel.txt"
    else:
        raise NotImplementedError(f"Transcription factors for `{organism!r}` are not yet implemented.")

    with open(Path(__file__).parent / "_data" / fname) as fin:
        return sorted(tf.strip() for tf in fin.readlines())


def proliferation_markers(organism: Literal["human", "mouse"]) -> List[str]:
    """TODO."""
    if organism not in ("human", "mouse"):
        raise NotImplementedError(f"Proliferation markers for `{organism!r}` are not yet implemented.")

    fname = f"{organism}_proliferation.txt"
    with open(Path(__file__).parent / "_data" / fname) as fin:
        return sorted(tf.strip() for tf in fin.readlines())


def apoptosis_markers(organism: Literal["human", "mouse"]) -> List[str]:
    """TODO."""
    if organism not in ("human", "mouse"):
        raise NotImplementedError(f"Apoptosis markers for `{organism!r}` are not yet implemented.")

    fname = f"{organism}_apoptosis.txt"
    with open(Path(__file__).parent / "_data" / fname) as fin:
        return sorted(tf.strip() for tf in fin.readlines())
