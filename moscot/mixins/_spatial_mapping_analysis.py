from moscot.mixins._spatial_analysis import SpatialAnalysisMixin
from moscot.problems.space import SpatialMappingProblem
from typing import Any, Dict, Tuple, Union, Optional

class SpatialMappingAnalysisMixin(SpatialAnalysisMixin):

    def __init__(self, problem: SpatialMappingProblem):
        self._problem = problem

    def correlate(self, keys_subset: Optional[Union[str, Dict[Any, Tuple[str, str]]]] = None,
                  var_subset: Optional[Union[Tuple[Any, Any], Dict[Any, Tuple[Any, Any]]]] = None,):
        """
        compute correlation of spatial mappings sols wrt given genes
        Parameters
        ----------
        keys_subset: key(s) for .var which indicate marker genes (expects identical keys for `spatial` and 'scRNA' adata).
         either a single key or a key for each problem.
        var_subset: subset(s) of marker genes to use, either a single list or dictionary of lists.
        either a single list or a lists for each problem.
        Returns
        -------

        """
        # if var_subset is not None:
        #     if len()
        # for i, prob in enumerate(self._problem._problems):
        #
        #
        #     masks = var_subset if var_subset is not None else (prob.adata.var[keys_subset],
        #                                                              prob._adata_y.var[keys_subset])



        return