from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union, Generic, TypeVar, Hashable, Iterable, Optional, Sequence
from operator import gt, lt
from itertools import product

from typing_extensions import Literal
import pandas as pd
import networkx as nx

import numpy as np

from anndata import AnnData

from moscot._types import ArrayLike

Value_t = Tuple[ArrayLike, ArrayLike]
Axis_t = Literal["obs", "var"]
Policy_t = Literal[
    "sequential",
    "pairwise",
    "star",
    "external_star",
    "triu",
    "tril",
    "explicit",
]


__all__ = [
    "SubsetPolicy",
    "OrderedPolicy",
    "PairwisePolicy",
    "StarPolicy",
    "ExternalStarPolicy",
    "SequentialPolicy",
    "TriangularPolicy",
    "ExplicitPolicy",
    "DummyPolicy",
]


K = TypeVar("K", bound=Hashable)


class FormatterMixin(ABC):
    @abstractmethod
    def _format(self, value: Any, *, is_source: bool) -> Any:
        pass


class SubsetPolicy(Generic[K]):
    """Policy class."""

    def __init__(
        self, adata: Union[AnnData, pd.Series, pd.Categorical], key: Optional[str] = None, axis: Axis_t = "obs"
    ):
        if isinstance(adata, AnnData):
            # TODO(michalk8): raise nicer KeyError (giovp) this way we can solve for full anndata with key=None
            if key is not None:
                if axis == "obs":
                    self._data = pd.Series(adata.obs[key])
                elif axis == "var":
                    self._data = pd.Series(adata.var[key])
                else:
                    raise ValueError(f"TODO: wrong axis `{axis}`")
            else:
                raise ValueError(f"TODO: wrong axis `{axis}`")
        else:
            self._data = adata
        if not hasattr(self._data, "cat"):
            self._data = self._data.astype("category")  # TODO(@MUCDK): catch conversion error
        self._axis = axis
        self._graph: Sequence[Tuple[K, K]] = []
        self._cat = tuple(self._data.cat.categories)
        self._subset_key: Optional[str] = key

    @abstractmethod
    def _create_graph(self, **kwargs: Any) -> Sequence[Tuple[K, K]]:
        pass

    def plan(self, forward: bool = True, **kwargs: Any) -> Sequence[Tuple[K, K]]:
        plan = self._plan(**kwargs)
        return plan if forward else plan[::-1]

    @abstractmethod
    def _plan(self, **kwargs: Any) -> Sequence[Tuple[K, K]]:
        pass

    def __call__(self, filter: Optional[Sequence[Tuple[K, K]]] = None, **kwargs: Any) -> "SubsetPolicy[K]":
        graph = self._create_graph(**kwargs)
        if filter is not None:
            graph = self._filter_graph(graph, filter=filter)
        if not len(graph):
            raise ValueError("TODO: empty graph")
        self._graph = graph
        return self

    def _filter_graph(self, graph: Sequence[Tuple[K, K]], filter: Sequence[Tuple[K, K]]) -> Sequence[Tuple[K, K]]:
        return [i for i in graph if i in filter]

    @classmethod
    def create(
        cls,
        kind: Policy_t,
        adata: Union[AnnData, pd.Series, pd.Categorical],
        **kwargs: Any,
    ) -> "SubsetPolicy[K]":
        if kind == "sequential":
            return SequentialPolicy(adata, **kwargs)
        if kind == "pairwise":
            return PairwisePolicy(adata, **kwargs)
        if kind == "star":
            return StarPolicy(adata, **kwargs)
        if kind == "external_star":
            return ExternalStarPolicy(adata, **kwargs)
        if kind == "triu":
            return TriangularPolicy(adata, **kwargs, upper=True)
        if kind == "tril":
            return TriangularPolicy(adata, **kwargs, upper=False)
        if kind == "explicit":
            return ExplicitPolicy(adata, **kwargs)

        raise NotImplementedError(kind)

    def create_mask(self, value: Union[K, Sequence[K]], *, allow_empty: bool = False) -> ArrayLike:
        if isinstance(value, str) or not isinstance(value, Iterable):
            mask = self._data == value
        else:
            mask = self._data.isin(value)
        if not allow_empty and not np.sum(mask):
            raise ValueError("TODO: empty mask...")
        return np.asarray(mask)

    def create_masks(self, discard_empty: bool = True) -> Dict[Tuple[K, K], Tuple[ArrayLike, ArrayLike]]:
        res = {}
        for a, b in self._graph:
            try:
                mask_a = self.create_mask(a, allow_empty=not discard_empty)
                mask_b = self.create_mask(b, allow_empty=not discard_empty)
                res[a, b] = mask_a, mask_b
            except ValueError as e:
                if "TODO: empty mask" not in str(e):
                    raise

        if not res:
            # can only happen when `discard_empty=True`
            raise ValueError("TODO: All masks were discarded.")

        return res

    @property
    def axis(self) -> Axis_t:
        return self._axis


class OrderedPolicy(SubsetPolicy[K], ABC):
    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], **kwargs: Any):
        super().__init__(adata, **kwargs)
        # TODO(michalk8): verify whether they can be ordered (only numeric?) + warn (or just raise)

    def _plan(self, start: Optional[K] = None, end: Optional[K] = None, **_: Any) -> Sequence[Tuple[K, K]]:
        if self._graph is None:
            raise RuntimeError("TODO: run graph creation first")
        if start is None and end is None:
            return self._graph
        # TODO: add Graph for undirected
        G = nx.DiGraph()
        G.add_edges_from(self._graph)

        if start == end:
            raise ValueError("TODO: start is the same as end.")
        if start is None or end is None:
            raise ValueError("TODO: start or end is None")

        path = nx.shortest_path(G, start, end)
        return list(zip(path[:-1], path[1:]))


class SimplePlanPolicy(SubsetPolicy[K], ABC):
    def _plan(self, **_: Any) -> Sequence[Tuple[K, K]]:
        return self._graph


class PairwisePolicy(SimplePlanPolicy[K]):
    def _create_graph(self, *_: Any, **__: Any) -> Sequence[Tuple[K, K]]:
        return [(a, b) for a, b in product(self._cat, self._cat) if a != b]


class StarPolicy(SimplePlanPolicy[K]):
    def _create_graph(self, reference: K, **kwargs: Any) -> Sequence[Tuple[K, K]]:  # type: ignore[override]
        if reference not in self._cat:
            raise ValueError(f"TODO: Reference {reference} not in {self._cat}")
        return [(c, reference) for c in self._cat if c != reference]

    def _filter_graph(
        self, graph: Sequence[Tuple[K, K]], filter: Sequence[Union[K, Tuple[K, K]]]
    ) -> Sequence[Tuple[K, K]]:
        filter = [src[0] if isinstance(src, tuple) else src for src in filter]
        return [(src, ref) for src, ref in graph if src in filter]


class ExternalStarPolicy(FormatterMixin, StarPolicy[K]):
    _SENTINEL = object()

    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], tgt_name: str = "ref", **kwargs: Any):
        super().__init__(adata, **kwargs)
        self._tgt_name = tgt_name

    def _format(self, value: K, *, is_source: bool) -> Union[str, K]:
        if is_source:
            return value
        if value is self._SENTINEL:
            return self._tgt_name
        raise ValueError("TODO.")

    def _create_graph(self, **_: Any) -> Sequence[Tuple[K, object]]:  # type: ignore[override]
        return [(c, self._SENTINEL) for c in self._cat if c != self._SENTINEL]

    def create_masks(self, discard_empty: bool = True) -> Dict[Tuple[K, K], Tuple[ArrayLike, ArrayLike]]:
        return super().create_masks(discard_empty=False)


class SequentialPolicy(OrderedPolicy[K]):
    def _create_graph(self, *_: Any, **__: Any) -> Sequence[Tuple[K, K]]:
        return list(zip(self._cat[:-1], self._cat[1:]))


class TriangularPolicy(OrderedPolicy[K]):
    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], upper: bool = True, **kwargs: Any):
        super().__init__(adata, **kwargs)
        self._compare = lt if upper else gt

    def _create_graph(self, **__: Any) -> Sequence[Tuple[K, K]]:
        return [(a, b) for a, b in product(self._cat, self._cat) if self._compare(a, b)]


class ExplicitPolicy(SimplePlanPolicy[K]):
    def _create_graph(self, subset: Sequence[Tuple[K, K]], **_: Any) -> Sequence[Tuple[K, K]]:  # type: ignore[override]
        if subset is None:
            raise ValueError("TODO: specify subset for explicit policy.")
        # pass-through, all checks are done by us later
        return subset


class DummyPolicy(FormatterMixin, SubsetPolicy[str]):
    _SENTINEL = object()

    def __init__(
        self,
        adata: Union[AnnData, pd.Series, pd.Categorical],
        src_name: str = "src",
        tgt_name: str = "ref",
        **kwargs: Any,
    ):
        super().__init__(pd.Series([self._SENTINEL] * len(adata)), **kwargs)
        self._src_name = src_name
        self._tgt_name = tgt_name

    def _create_graph(self, **__: Any) -> Sequence[Tuple[object, object]]:  # type: ignore[override]
        return [(self._SENTINEL, self._SENTINEL)]

    def _plan(self, **_: Any) -> List[Tuple[str, str]]:
        return [(self._src_name, self._tgt_name)]

    def _format(self, _: Any, *, is_source: bool) -> str:
        return self._src_name if is_source else self._tgt_name
