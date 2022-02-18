from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sized, Tuple, Union, Literal, Optional, Sequence
from operator import gt, lt
from itertools import product

import pandas as pd
import networkx as nx

import numpy as np
import numpy.typing as npt

from anndata import AnnData

Item_t = Tuple[Any, Any]
Value_t = Tuple[npt.ArrayLike, npt.ArrayLike]


__all__ = (
    "SubsetPolicy",
    "OrderedPolicy",
    "PairwisePolicy",
    "StarPolicy",
    "ExternalStarPolicy",
    "SequentialPolicy",
    "TriangularPolicy",
    "ExplicitPolicy",
)


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
        self._subset: Optional[List[Item_t]] = None
        self._cat = self.Category(self._data.cat.categories)

    @abstractmethod
    def _create_subset(self, *args: Any, **kwargs: Any) -> Sequence[Item_t]:
        pass

    @abstractmethod
    def plan(self, **kwargs: Any) -> Dict[Item_t, List[Item_t]]:
        pass

    def __call__(self, *args: Any, filter: Optional[Sequence[Any]] = None, **kwargs: Any) -> "SubsetPolicy":
        subset = self._create_subset(*args, **kwargs)
        if not all(isinstance(s, Sized) and len(s) == 2 for s in subset):
            raise ValueError("TODO: not all values are two-pair")
        if filter is not None:
            subset = [(a, b) for a, b in subset if (a in filter and b in filter) or (a, b) in filter]
        if not len(subset):
            raise ValueError("TODO: empty subset")
        # TODO(michalk8): make unique, but order-preserving
        self._subset = subset
        return self

    @classmethod
    def create(
        cls,
        kind: Literal["sequential", "pairwise", "star", "triu", "tril", "explicit"],
        adata: AnnData,
        key: Optional[str] = None,
    ) -> "SubsetPolicy":
        if kind == "sequential":
            return SequentialPolicy(adata, key=key)
        if kind == "pairwise":
            return PairwisePolicy(adata, key=key)
        if kind == "star":
            return StarPolicy(adata, key=key)
        if kind == "external_star":
            return ExternalStarPolicy(adata, key=key)
        if kind == "triu":
            return TriangularPolicy(adata, key=key, upper=True)
        if kind == "tril":
            return TriangularPolicy(adata, key=key, upper=False)
        if kind == "explicit":
            return ExplicitPolicy(adata, key=key)

        raise NotImplementedError(kind)

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

    @property
    def _default_plan(self) -> Dict[Tuple[Any, Any], List[Any]]:
        return {s: [s] for s in self._subset}


class SimplePlanFilterMixin:
    def plan(self, filter: Optional[Sequence[Item_t]] = None, **_: Any) -> Dict[Item_t, List[Item_t]]:
        if filter is None:
            return self._default_plan
        return {s: [s] for s in self._subset if s in filter}


class OrderedPolicy(SubsetPolicy, ABC):
    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], key: Optional[str] = None):
        super().__init__(adata, key=key)
        # TODO(michalk8): verify whether they can be ordered (only numeric?) + warn (or just raise)

    def plan(self, start: Optional[Any] = None, end: Optional[Any] = None, **_: Any) -> Dict[Item_t, List[Item_t]]:
        if self._subset is None:
            raise RuntimeError("TODO: init subset first")
        if start is None and end is None:
            return self._default_plan

        G = nx.DiGraph()
        G.add_edges_from(self._subset)

        if start is None:
            paths = nx.single_target_shortest_path(G, end)
            return {(n, path[-1][-1]): list(zip(path[:-1], path[1:])) for n, path in paths.items() if n != end}
        if end is None:
            paths = nx.single_source_shortest_path(G, start)
            return {(path[0][0], n): list(zip(path[:-1], path[1:])) for n, path in paths.items() if n != start}
        if start != end:
            path = nx.shortest_path(G, start, end)
            return {(start, end): list(zip(path[:-1], path[1:]))}

        raise ValueError("TODO: start is the same as end.")


class PairwisePolicy(SimplePlanFilterMixin, SubsetPolicy):
    def _create_subset(self, *_: Any, **__: Any) -> Sequence[Item_t]:
        return [(a, b) for a, b in zip(self._cat[:-1], self._cat[1:])]


class StarPolicy(SubsetPolicy):
    def _create_subset(self, reference: Any, **kwargs: Any) -> Sequence[Item_t]:
        if reference not in self._cat:
            raise ValueError(f"TODO: Reference {reference} not in {self._cat}")
        return [(c, reference) for c in self._cat if c != reference]

    def plan(self, filter: Optional[Sequence[Any]] = None, **_: Any) -> Dict[Item_t, List[Item_t]]:
        if filter is None:
            return self._default_plan
        return {s: [s] for s in self._subset if s[0] in filter}


class ExternalStarPolicy(StarPolicy):
    _SENTINEL = object()

    def _create_subset(self, **kwargs: Any) -> Sequence[Item_t]:
        return [(c, self._SENTINEL) for c in self._cat if c != self._SENTINEL]

    def mask(self, discard_empty: bool = True) -> Dict[Item_t, Value_t]:
        return super().mask(discard_empty=False)


class SequentialPolicy(OrderedPolicy):
    def _create_subset(self, *_: Any, **__: Any) -> Sequence[Item_t]:
        return [(c, self._cat.next(c)) for c in self._cat[:-1]]


class TriangularPolicy(OrderedPolicy):
    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], key: Optional[str] = None, upper: bool = True):
        super().__init__(adata, key=key)
        self._compare = lt if upper else gt

    def _create_subset(self, *_: Any, **__: Any) -> Sequence[Item_t]:
        return [(a, b) for a, b in product(self._cat, self._cat) if self._compare(a, b)]


class ExplicitPolicy(SimplePlanFilterMixin, SubsetPolicy):
    def _create_subset(self, subset: Sequence[Item_t], **_: Any) -> Sequence[Item_t]:
        if subset is None:
            raise ValueError("TODO: specify subset for explicit policy.")
        # pass-through, all checks are done by us later
        return subset
