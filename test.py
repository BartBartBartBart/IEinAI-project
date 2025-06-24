from __future__ import annotations


import pysr
from pysr import PySRRegressor
import numpy as np

from typing import cast

import logging

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from sklearn.utils.validation import _check_feature_names_in  # type: ignore

from pysr.denoising import denoise, multi_denoise
import copy


try:
    from typing import List
except ImportError:
    from typing_extensions import List
from typing import Any, Literal, Tuple, Union, cast, TypeVar
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

T = TypeVar("T", bound=Any)

ArrayLike = Union[ndarray, List[T]]

pysr_logger = logging.getLogger(__name__)


def run_feature_selection_new(
    X: ndarray,
    y: ndarray,
    select_k_features: int,
    random_state: np.random.RandomState | None = None,
) -> NDArray[np.bool_]:
    """
    Find most important features.

    Uses a gradient boosting tree regressor as a proxy for finding
    the k most important features in X, returning indices for those
    features as output.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel
    print("Biem bam boem, feature selection is running...")

    clf = RandomForestRegressor(
        n_estimators=100, max_depth=3, random_state=random_state
    )
    clf.fit(X, y)
    selector = SelectFromModel(
        clf, threshold=-np.inf, max_features=select_k_features, prefit=True
    )
    return cast(NDArray[np.bool_], selector.get_support(indices=False))


class my_class(PySRRegressor):
    def __init__(self, *args, **kwargs):
        super(my_class, self).__init__(*args, **kwargs)
        
    def _pre_transform_training_data(
        self,
        X: ndarray,
        y: ndarray,
        Xresampled: ndarray | None,
        variable_names: ArrayLike[str],
        complexity_of_variables: int | float | list[int | float] | None,
        X_units: ArrayLike[str] | None,
        y_units: ArrayLike[str] | str | None,
        random_state: np.random.RandomState,
    ):
        if self.select_k_features:
            print("BIEM BAM BOEM")
            selection_mask = run_feature_selection_new(
                X, y, self.select_k_features, random_state=random_state
            )
            print("BIEM BAM BOEM 2")
            X = X[:, selection_mask]

            if Xresampled is not None:
                Xresampled = Xresampled[:, selection_mask]

            # Reduce variable_names to selection
            variable_names = cast(
                ArrayLike[str],
                [
                    variable_names[i]
                    for i in range(len(variable_names))
                    if selection_mask[i]
                ],
            )

            if isinstance(complexity_of_variables, list):
                complexity_of_variables = [
                    complexity_of_variables[i]
                    for i in range(len(complexity_of_variables))
                    if selection_mask[i]
                ]
                self.complexity_of_variables_ = copy.deepcopy(complexity_of_variables)

            if X_units is not None:
                X_units = cast(
                    ArrayLike[str],
                    [X_units[i] for i in range(len(X_units)) if selection_mask[i]],
                )
                self.X_units_ = copy.deepcopy(X_units)

            # Re-perform data validation and feature name updating
            X, y = self._validate_data_X_y(X, y)
            # Update feature names with selected variable names
            self.selection_mask_ = selection_mask
            self.feature_names_in_ = _check_feature_names_in(self, variable_names)
            self.display_feature_names_in_ = self.feature_names_in_
            pysr_logger.info(f"Using features {self.feature_names_in_}")

        # Denoising transformation
        if self.denoise:
            if self.nout_ > 1:
                X, y = multi_denoise(
                    X, y, Xresampled=Xresampled, random_state=random_state
                )
            else:
                X, y = denoise(X, y, Xresampled=Xresampled, random_state=random_state)

        return X, y, variable_names, complexity_of_variables, X_units, y_units
    
    # def feature_selection(self):

X_train = 2 * np.random.randn(100, 5)
y_train = 2 * np.cos(X_train[:, 3]) + X_train[:, 0] ** 2 - 2
X_test = 2 * np.random.randn(100, 5)
y_test = 2 * np.cos(X_test[:, 3]) + X_test[:, 0] ** 2 - 2
x1_plot = np.linspace(-2, 2, 100)
x2_plot = np.linspace(-2, 2, 100)
x1_plot, x2_plot = np.meshgrid(x1_plot, x2_plot)

pysr = my_class(
    niterations=1,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    verbosity=1
)
pysr.fit(X_train, y_train)
y_pysr = pysr.predict(np.c_[X_test[:, 0].ravel(), X_test[:, 1].ravel()]).reshape(X_test[:, 0].shape)
y_pysr_plot = pysr.predict(np.c_[x1_plot.ravel(), x2_plot.ravel()]).reshape(x1_plot.shape)
score_pysr = pysr.score(X_test, y_test)
expr_pysr = pysr.get_best().equation
complexity_pysr = pysr.get_best().complexity
print(f"PySR score: {score_pysr}")
print(f"PySR equation: {expr_pysr}")
print(f"PySR complexity: {complexity_pysr}")