import abc
import contextlib
import itertools
import operator
from typing import (
    Any,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import networkx as nx
import numpy as np
import pandas as pd

from anndata import AnnData

from moscot import _constants
from moscot._types import ArrayLike, Policy_t

__all__ = [
    "SubsetPolicy",
    "OrderedPolicy",
    "StarPolicy",
    "ExternalStarPolicy",
    "SequentialPolicy",
    "TriangularPolicy",
    "ExplicitPolicy",
    "DummyPolicy",
    "FormatterMixin",
    "create_policy",
]


K = TypeVar("K", bound=Hashable)


class FormatterMixin(abc.ABC):
    @abc.abstractmethod
    def _format(self, value: Any, *, is_source: bool) -> Any:
        pass


class SubsetPolicy(Generic[K], abc.ABC):
    """Policy class."""

    def __init__(
        self,
        adata: Union[AnnData, pd.Series, pd.Categorical],
        key: Optional[str] = None,
        verify_integrity: bool = True,
    ):
        try:
            self._data = pd.Series(adata.obs[key]) if isinstance(adata, AnnData) else adata
        except KeyError:
            raise KeyError(f"Unable to find data in `adata.obs[{key!r}]`.") from None
        self._data = self._data.astype("category")  # TODO(@MUCDK): catch conversion error
        self._graph: Set[Tuple[K, K]] = set()
        self._cat = tuple(self._data.cat.categories)
        self._subset_key: Optional[str] = key

        if verify_integrity and len(self._cat) < 2:
            raise ValueError(f"Policy must contain at least `2` different values, found `{len(self._cat)}`.")

    @abc.abstractmethod
    def _create_graph(self, **kwargs: Any) -> Set[Tuple[K, K]]:
        pass

    def plan(
        self,
        filter: Optional[Sequence[Tuple[K, K]]] = None,  # noqa: A002
        explicit_steps: Optional[Sequence[Tuple[K, K]]] = None,
        **kwargs: Any,
    ) -> Sequence[Tuple[K, K]]:
        """Compute all pairs of distributions belonging to the policy.

        Parameters
        ----------
        %(fitler)s
        %(explicit_steps)s
        %(kwargs_plan)s

        Returns
        -------
        Sequence containing pairs of distributions.
        """
        if explicit_steps is not None:
            G = nx.DiGraph()
            G.add_edges_from(explicit_steps)
            if not set(G.nodes).issubset(self._cat):
                raise ValueError(
                    f"Explicitly specified steps `{set(explicit_steps)}` must be a subset of `{self._cat}`."
                )
            if not nx.has_path(G, explicit_steps[0][0], explicit_steps[-1][1]):
                raise ValueError("Explicitly specified steps do not form a connected path.")
            return explicit_steps
        plan = self._plan(**kwargs)
        # TODO(michalk8): ensure unique
        if filter is not None:
            plan = self._filter_plan(plan, filter=filter)
        if not len(plan):
            raise ValueError("Unable to create a plan, no steps were selected after filtering.")
        return plan

    @abc.abstractmethod
    def _plan(self, **kwargs: Any) -> Sequence[Tuple[K, K]]:
        pass

    def __call__(self, **kwargs: Any) -> "SubsetPolicy[K]":
        graph = self._create_graph(**kwargs)
        if not len(graph):
            raise ValueError("The policy graph is empty.")
        self._graph = graph
        return self

    def _filter_plan(
        self, plan: Sequence[Tuple[K, K]], filter: Sequence[Tuple[K, K]]  # noqa: A002
    ) -> Sequence[Tuple[K, K]]:
        return [step for step in plan if step in filter]

    def create_mask(self, value: Union[K, Sequence[K]], *, allow_empty: bool = False) -> ArrayLike:
        if isinstance(value, str) or not isinstance(value, Iterable):
            mask = self._data == value
        else:
            mask = self._data.isin(value)
        if not allow_empty and not np.sum(mask):
            raise ValueError("Unable to construct an empty mask, use `allow_empty=True` to override.")
        return np.asarray(mask)

    def create_masks(self, discard_empty: bool = True) -> Dict[Tuple[K, K], Tuple[ArrayLike, ArrayLike]]:
        res = {}
        for a, b in self._graph:
            try:
                mask_a = self.create_mask(a, allow_empty=not discard_empty)
                mask_b = self.create_mask(b, allow_empty=not discard_empty)
                res[a, b] = mask_a, mask_b
            except ValueError as e:
                if "Unable to construct an empty mask" not in str(e):
                    raise

        if not res:
            # can only happen when `discard_empty=True`
            raise ValueError("All empty masks were discarded.")

        return res

    def add_node(self, node: Tuple[K, K], only_existing: bool = False) -> None:
        """
        Add a node to the policy.

        Parameters
        ----------
        %(node)s
        %(only_existing)s

        Returns
        -------
        None
        """
        if self._graph is None:
            raise RuntimeError("Construct the policy graph first.")
        src, tgt = node
        if src == tgt:
            raise ValueError(f"Unable to add `{src, tgt}` node, self connections are disallowed.")
        if only_existing and (src not in self._cat or tgt not in self._cat):
            raise ValueError(
                f"Unable to add `{src}` or `{tgt}` node(s) that are not already present "
                f"in the policy graph, use `only_existing=False` to override."
            )
        self._graph.add(node)

    def remove_node(self, node: Tuple[K, K]) -> None:
        """
        Remove a node from the policy graph.

        Parameters
        ----------
        %(node)s

        Returns
        -------
        None
        """
        if self._graph is None:
            raise RuntimeError("Construct the policy graph first.")
        with contextlib.suppress(KeyError):
            self._graph.remove(node)


class OrderedPolicy(SubsetPolicy[K], abc.ABC):
    """
    Policy taking into account the order of nodes.

    Parameters
    ----------
    %(adata_policy)s
    %(kwargs_policy)s

    Returns
    -------
    None
    """

    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], **kwargs: Any):
        super().__init__(adata, **kwargs)
        # TODO(michalk8): verify whether they can be ordered (only numeric?) + warn (or just raise)

    def _plan(
        self, forward: bool = True, start: Optional[K] = None, end: Optional[K] = None, **_: Any
    ) -> Sequence[Tuple[K, K]]:
        if self._graph is None:
            raise RuntimeError("Construct the policy graph first.")
        if start is None and end is None:
            start, end = self._cat[0], self._cat[-1]
        if start is None:
            start = self._cat[0]
        if end is None:
            end = self._cat[-1]
        # TODO: add Graph for undirected
        G = nx.DiGraph()
        G.add_edges_from(self._graph)

        if start == end:
            raise ValueError(f"Start node `{start}` is the same as the end node `{end}`.")
        if start is None or end is None:
            raise ValueError("Both start and end node are `None`.")

        path = nx.shortest_path(G, start, end)
        path = list(zip(path[:-1], path[1:]))

        return path if forward else path[::-1]


class SimplePlanPolicy(SubsetPolicy[K], abc.ABC):
    def _plan(self, **_: Any) -> Sequence[Tuple[K, K]]:
        return list(self._graph)


class StarPolicy(SimplePlanPolicy[K]):
    def _create_graph(self, reference: K, **kwargs: Any) -> Set[Tuple[K, K]]:  # type: ignore[override]
        if reference not in self._cat:
            raise ValueError(f"Reference `{reference}` is not in valid nodes: `{self._cat}`.")
        return {(c, reference) for c in self._cat if c != reference}

    def _filter_plan(
        self, plan: Sequence[Tuple[K, K]], filter: Sequence[Union[K, Tuple[K, K]]]  # noqa: A002
    ) -> Sequence[Tuple[K, K]]:
        filter = [src[0] if isinstance(src, tuple) else src for src in filter]  # noqa: A001
        return [(src, ref) for src, ref in plan if src in filter]

    @property
    def reference(self) -> K:
        for _, ref in self._graph:
            return ref
        raise ValueError("Graph is empty.")

    def add_node(self, node: Union[K, Tuple[K, K]], only_existing: bool = False) -> None:
        if not isinstance(node, tuple):
            node = (node, self.reference)
        return super().add_node(node, only_existing=only_existing)  # type: ignore[arg-type]

    def remove_node(self, node: Union[K, Tuple[K, K]]) -> None:
        if not isinstance(node, tuple):
            node = (node, self.reference)
        return super().remove_node(node)  # type: ignore[arg-type]


class ExternalStarPolicy(FormatterMixin, StarPolicy[K]):
    """Policy with one central node connected to all other nodes."""

    _SENTINEL = object()

    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], tgt_name: str = "ref", **kwargs: Any):
        super().__init__(adata, **kwargs)
        self._tgt_name: K = tgt_name  # type: ignore [assignment]

    def _format(self, value: K, *, is_source: bool) -> K:
        if is_source:
            return value
        if value is self._SENTINEL:
            return self._tgt_name
        raise ValueError(f"Expected value to be `{self._SENTINEL}`, found `{value}`.")

    def _create_graph(self, **_: Any) -> Set[Tuple[K, object]]:  # type: ignore[override]
        return {(c, self._SENTINEL) for c in self._cat if c != self._SENTINEL}

    def _plan(self, **_: Any) -> Sequence[Tuple[K, K]]:
        return [(src, self._format(tgt, is_source=False)) for (src, tgt) in self._graph]

    def add_node(self, node: Union[K, Tuple[K, K]], only_existing: bool = False) -> None:
        """Add a node to the policy."""
        if isinstance(node, tuple):
            _, tgt = node
        if tgt is self._tgt_name:
            return None
        return super().add_node(node, only_existing=only_existing)  # type: ignore[arg-type]

    def create_masks(self, discard_empty: bool = True) -> Dict[Tuple[K, K], Tuple[ArrayLike, ArrayLike]]:
        return super().create_masks(discard_empty=False)


class SequentialPolicy(OrderedPolicy[K]):
    """Policy connecting subsequent nodes."""

    def _create_graph(self, *_: Any, **__: Any) -> Set[Tuple[K, K]]:
        return set(zip(self._cat[:-1], self._cat[1:]))


class TriangularPolicy(OrderedPolicy[K]):
    """Connect all nodes with all subsequent nodes."""

    def __init__(self, adata: Union[AnnData, pd.Series, pd.Categorical], upper: bool = True, **kwargs: Any):
        super().__init__(adata, **kwargs)
        self._compare = operator.lt if upper else operator.gt

    def _create_graph(self, **__: Any) -> Set[Tuple[K, K]]:
        return {(a, b) for a, b in itertools.product(self._cat, self._cat) if self._compare(a, b)}


class ExplicitPolicy(SimplePlanPolicy[K]):
    """A policy specified by the user."""

    def _create_graph(self, subset: Sequence[Tuple[K, K]], **_: Any) -> Set[Tuple[K, K]]:  # type: ignore[override]
        if subset is None:
            raise ValueError("No steps specifying the explicit policy.")
        return set(subset)  # pass-through, all checks are done later


class DummyPolicy(FormatterMixin, SubsetPolicy[str]):
    """Policy to handle sentinel values."""

    _SENTINEL = object()

    def __init__(
        self,
        adata: Union[AnnData, pd.Series, pd.Categorical],
        src_name: Literal["src"] = "src",
        tgt_name: Literal["tgt"] = "tgt",
        **kwargs: Any,
    ):
        super().__init__(pd.Series([self._SENTINEL] * len(adata)), verify_integrity=False, **kwargs)
        self._cat = (src_name, tgt_name)
        self._src_name = src_name
        self._tgt_name = tgt_name

    def _create_graph(self, **__: Any) -> Set[Tuple[object, object]]:  # type: ignore[override]
        return {(self._SENTINEL, self._SENTINEL)}

    def _plan(self, **_: Any) -> List[Tuple[str, str]]:
        return [(self._src_name, self._tgt_name)]

    def _format(self, _: Any, *, is_source: bool) -> str:
        return self._src_name if is_source else self._tgt_name

    def _filter_plan(
        self, plan: Sequence[Tuple[K, K]], filter: Sequence[Tuple[K, K]]  # noqa: A002
    ) -> Sequence[Tuple[K, K]]:
        return plan


# TODO(michalk8): in the future, use Registry
def create_policy(
    kind: Policy_t,
    adata: Union[AnnData, pd.Series, pd.Categorical],
    **kwargs: Any,
) -> SubsetPolicy[K]:
    """Create a policy.

    Parameters
    ----------
    %(policy_kind)s
    %(adata)s
    kwargs
        Keyword arguments for a :class:`~moscot.utils.subset_policy.SubsetPolicy`.

    Returns
    -------
    The policy.

    Notes
    -----
    TODO: link policy example.
    """
    if kind == _constants.SEQUENTIAL:
        return SequentialPolicy(adata, **kwargs)
    if kind == _constants.STAR:
        return StarPolicy(adata, **kwargs)
    if kind == _constants.EXTERNAL_STAR:
        return ExternalStarPolicy(adata, **kwargs)
    if kind == _constants.TRIU:
        return TriangularPolicy(adata, **kwargs, upper=True)
    if kind == _constants.TRIL:
        return TriangularPolicy(adata, **kwargs, upper=False)
    if kind == _constants.EXPLICIT:
        return ExplicitPolicy(adata, **kwargs)

    raise NotImplementedError(kind)
