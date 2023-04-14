from pathlib import Path
from typing import List, Literal

__all__ = ["transcription_factors", "proliferation_markers", "apoptosis_markers"]


def transcription_factors(organism: Literal["human", "mouse", "drosophila"]) -> List[str]:
    """Get transcription factors for a selected organism.

    The data was taken from this `source <https://resources.aertslab.org/cistarget/tf_lists/>`_.

    Parameters
    ----------
    organism
        Organism for which to select the transcription factors.

    Returns
    -------
    Transcription factors for ``organism``.
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
    """Get proliferation markers for a selected organism.

    Parameters
    ----------
    organism
        Organism for which to select the marker genes. Human markers come from :cite:`tirosh:16:science`,
        mouse markers come from :cite:`tirosh:16:nature`.

    Returns
    -------
    Proliferation markers for ``organism``.
    """
    if organism not in ("human", "mouse"):
        raise NotImplementedError(f"Proliferation markers for `{organism!r}` are not yet implemented.")

    fname = f"{organism}_proliferation.txt"
    with open(Path(__file__).parent / "_data" / fname) as fin:
        return sorted(tf.strip() for tf in fin.readlines())


def apoptosis_markers(organism: Literal["human", "mouse"]) -> List[str]:
    """Get apoptosis markers for a selected organism.

    Parameters
    ----------
    organism
        Organism for which to select the marker genes. Human markers come from
        `Hallmark Apoptosis, MSigDB <https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_APOPTOSIS>`_,
        mouse markers come from
        `Hallmark P53 Pathway, MSigDB <https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_P53_PATHWAY>`_.

    Returns
    -------
    Apoptosis markers for ``organism``.
    """
    if organism not in ("human", "mouse"):
        raise NotImplementedError(f"Apoptosis markers for `{organism!r}` are not yet implemented.")

    fname = f"{organism}_apoptosis.txt"
    with open(Path(__file__).parent / "_data" / fname) as fin:
        return sorted(tf.strip() for tf in fin.readlines())
