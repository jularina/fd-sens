from src.basis_functions.basis_functions import *


BASIS_FUNCTIONS_REGISTRY = {
    "RBFBasisFunction": RBFBasisFunction,
    "MaternBasisFunction": MaternBasisFunction,
    "MaternBasisFunctionMultidim": MaternBasisFunctionMultidim,
    "FixedCentersRBFBasisFunctionMultidim": FixedCentersRBFBasisFunctionMultidim,
}
