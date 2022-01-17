from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sized, Tuple, Union, Literal, Optional, Sequence
from operator import ge, le
from itertools import product

import pandas as pd
import networkx as nx

import numpy as np
import numpy.typing as npt

from anndata import AnnData

Item_t = Tuple[Any, Any]
Value_t = Tuple[npt.ArrayLike, npt.ArrayLike]


__all__ = ("SubsetPolicy", "OrderedPolicy", "PairwisePolicy", "SequentialPolicy", "TriangularPolicy")


class SubsetPolicy:
    class Category:
        def __init__(self, cats: Sequence[Any]):
            assert len(cats) > 1, "TODO: too few categories"
            self._i2c = tuple(cats)
            self._c2i = dict(zip(cats, range(len(cats))))
            self._next_cat = dict(zip(cats[:-1], cats[1:]))
            self._prev_cat = {v: k for k, v in self._next_cat.items()}

        def _move(self, curr: Any, *, forward: bool = True):
            direction = self._next_cat if forward else self._prev_cat
            if curr not in direction:
                raise IndexError("TODO: out-of-range")

            return direction[curr]

        def next(self, curr: Any) -> Any:
            return self._move(curr, forward=True)

        def prev(self, curr: Any) -> Any:
            return self._move(curr, forward=False)

        @property
        def categories(self) -> Tuple[Any]:
            return self._i2c

        def __len__(self) -> int:
            return len(self._i2c)

        def __getitem__(self, ix: int) -> Any:
            return self._i2c[ix]

        def __invert__(self) -> "SubsetPolicy.Category":
            return SubsetPolicy.Category(self[::-1])

    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], key: Optional[str] = None):
        self._data: pd.Series = pd.Series(adata.obs[key] if isinstance(adata, AnnData) else adata)
        self._subset: Optional[Dict[Tuple[Any, Any], Tuple[npt.ArrayLike, npt.ArrayLike]]] = None
        self._cat = self.Category(self._data.cat.categories)

    @abstractmethod
    def _create_subset(self, *args: Any, **kwargs: Any) -> Sequence[Item_t]:
        pass

    def subset(self, *args: Any, **kwargs: Any) -> "SubsetPolicy":
        subset = self._create_subset(*args, **kwargs)
        if not len(subset):
            raise ValueError("TODO: empty subset")
        if not all(isinstance(s, Sized) and len(s) == 2 for s in subset):
            raise ValueError("TODO: not all values are two-pair")
        # TODO(michalk8): make unique, but order-preserving
        self._subset = subset
        return self

    @classmethod
    def create(cls, kind: Literal["TODO"], adata: AnnData, key: Optional[str] = None) -> "SubsetPolicy":
        if kind == "sequential":
            return SequentialPolicy(adata, key=key)
        if kind == "pairwise":
            return PairwisePolicy(adata, key=key)
        if kind == "triu":
            return TriangularPolicy(adata, key=key, upper=True)
        if kind == "tril":
            return TriangularPolicy(adata, key=key, upper=False)
        if kind == "explicit":
            return ExplicitPolicy(adata, key=key)

        raise NotImplementedError(kind)

    def chain(self, start: Optional[Any] = None, end: Optional[Any] = None) -> List[Item_t]:
        start = self._cat[0] if start is None else start
        end = self._cat[-1] if end is None else end
        if start == end:
            raise ValueError("TODO: start is the same as end.")
        if self._subset is None:
            raise ValueError("TODO: initialize the subset first")

        G = nx.DiGraph() if isinstance(self, OrderedPolicy) else nx.Graph()
        G.add_edges_from(self._subset)
        path = nx.shortest_path(G, start, end)

        return list(zip(path[:-1], path[1:]))

    def mask(self, discard_empty: bool = True) -> Dict[Item_t, Value_t]:
        res = {}
        for a, b in self._subset:
            mask_a = self._data == a
            if discard_empty and not np.sum(mask_a):
                continue
            mask_b = self._data == b
            if discard_empty and not np.sum(mask_b):
                continue
            res[a, b] = mask_a, mask_b

        if not len(res):
            # can only happen when `discard_empty=True`
            raise ValueError("TODO: All masks were discarded.")

        return res


class OrderedPolicy(SubsetPolicy, ABC):
    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], key: Optional[str] = None):
        super().__init__(adata, key=key)
        # TODO(michalk8): verify whether they can be ordered (only numeric?) + warn (or just raise)


class PairwisePolicy(SubsetPolicy):
    def _create_subset(self, *_: Any, **__: Any) -> Sequence[Item_t]:
        return [(a, b) for a, b in zip(self._cat[:-1], self._cat[1:])]


class SequentialPolicy(OrderedPolicy):
    def _create_subset(self, *args: Any, **kwargs: Any) -> Sequence[Item_t]:
        return [(c, self._cat.next(c)) for c in self._cat[:-1]]


class TriangularPolicy(OrderedPolicy):
    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], key: Optional[str] = None, upper: bool = True):
        super().__init__(adata, key=key)
        self._compare = le if upper else ge

    def _create_subset(self, *_: Any, **__: Any) -> Sequence[Item_t]:
        return [(a, b) for a, b in product(self._cat, self._cat) if self._compare(a, b)]


class ExplicitPolicy(SubsetPolicy):
    def _create_subset(self, subset: Sequence[Item_t], **_: Any) -> Sequence[Item_t]:
        # pass-through, all checks are done by us later
        return subset

    def chain(
        self, start: Optional[Any] = None, end: Optional[Any] = None, interp_step: Optional[Union[int, float]] = None
    ) -> List[Item_t]:
        if not interp_step:
            return super().chain(start, end)

        start = self._cat[0] if start is None else start
        end = self._cat[-1] if end is None else end
        G = nx.DiGraph()  # TODO: if data not ordered, raise

        if isinstance(interp_step, int):
            G.add_edges_from(
                [(a, b) for a, b in self._subset if abs(self._cat._c2i[a] - self._cat._c2i[b]) <= interp_step]
            )
        elif isinstance(interp_step, float):
            # TODO: assert self._data is numeric
            G.add_edges_from([(a, b) for a, b in self._subset if abs(b - a) <= interp_step])
        else:
            raise TypeError("TODO: wrong interpolation type")

        # TODO: catch the error and print a nice message
        path = nx.shortest_path(G, start, end)
        return list(zip(path[:-1], path[1:]))
