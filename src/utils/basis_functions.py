from src.basis_functions.basis_functions import *


BASIS_FUNCTIONS_REGISTRY = {
    "PolynomialBasisFunction": PolynomialBasisFunction,
    "RBFBasisFunction": RBFBasisFunction,
    "SigmoidBasisFunction": SigmoidBasisFunction,
    "MaternBasisFunction": MaternBasisFunction,
    "RBFBasisFunctionMultidim": RBFBasisFunctionMultidim,
    "SigmoidBasisFunctionMultidim": SigmoidBasisFunctionMultidim,
    "PolynomialBasisFunctionMultidim": PolynomialBasisFunctionMultidim,
    "MaternBasisFunctionMultidim": MaternBasisFunctionMultidim,
    "FixedCentersRBFBasisFunction": FixedCentersRBFBasisFunction,
}
