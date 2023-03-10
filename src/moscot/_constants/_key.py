import contextlib
from typing import Any, Callable, List, Optional, Set

from anndata import AnnData

import numpy as np


class cprop:
    def __init__(self, f: Callable[..., str]):
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> str:
        return self.f(owner)


class Key:
    """Class which manages keys in :class:`anndata.AnnData`."""

    class obs:
        pass

    class obsm:
        @cprop
        def spatial(cls) -> str:
            return "spatial"

    class uns:
        @cprop
        def spatial(cls) -> str:
            return Key.obsm.spatial

        @classmethod
        def nhood_enrichment(cls, cluster: str) -> str:
            return f"{cluster}_nhood_enrichment"


class RandomKeys:
    """
    Create random keys inside an :class:`~anndata.AnnData` object.

    Parameters
    ----------
    adata
        Annotated data object.
    n
        Number of keys, If `None`, create just 1 keys.
    where
        Attribute of ``adata``. If `'obs'`, also clean up `'{key}_colors'` for each generated key.

    """

    def __init__(self, adata: AnnData, n: Optional[int] = None, where: str = "obs"):
        self._adata = adata
        self._where = where
        self._n = n or 1
        self._keys: List[str] = []

    def _generate_random_keys(self):
        def generator() -> str:
            return f"RNG_COL_{np.random.RandomState().randint(2 ** 16)}"

        where = getattr(self._adata, self._where)
        names: List[str] = []
        seen: Set[str] = set(where.keys())

        while len(names) != self._n:
            name = generator()
            if name not in seen:
                seen.add(name)
                names.append(name)

        return names

    def __enter__(self):
        self._keys = self._generate_random_keys()
        return self._keys

    def __exit__(self, exc_type, exc_val, exc_tb):
        with contextlib.suppress(KeyError):
            for key in self._keys:
                df = getattr(self._adata, self._where)
                df.drop(key, axis="columns", inplace=True)
                if self._where == "obs":
                    del self._adata.uns[f"{key}_colors"]
