from typing import Union, Sequence, Any, Optional
from numpy.typing import ArrayLike
from anndata import AnnData
from moscot.tmp.solvers._data import Tag


def _validate_losses(loss: Union[str, Sequence[ArrayLike]],
                     subsets: Sequence[Any]):
    if isinstance(loss, str):
        # TODO(@MUCDK) check for valid strings of losses. this does depend on backend
        return [loss] * len(subsets)
    elif isinstance(loss, Sequence):
        if len(subsets) != len(loss):
            raise ValueError("The number of cost matrices provided must be equal to the number of OT problems.")


def _validate_loss(loss: Union[str, ArrayLike],
                   adata_source: Optional[AnnData],
                   adata_target: Optional[AnnData]):
    if isinstance(loss, str):
        # TODO(@MUCDK) check for valid strings of losses. this does depend on backend
        return None
    elif isinstance(loss, ArrayLike):
        if adata_source.n_obs != len(loss.shape[0]):
            raise ValueError(f"Dimension mismatch. The cost matrix provided has dimension {loss.shape} "
                             f"but the number of data points for the source distribution is {adata_source.n_obs}.")
        if adata_target.n_obs != len(loss.shape[1]):
            raise ValueError(f"Dimension mismatch. The cost matrix provided has dimension {loss.shape} "
                             f"but the number of data points for the source distribution is {adata_target.n_obs}.")