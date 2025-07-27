from typing import Optional, Sequence, Union

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats

from utils import save_viz

class LikelihoodPlot:
    """
    Visualizes the statistics concept of "Likelihood".
    """
    def __init__(
        self,
        dist: dict,
    ):
        self.dist = dist

    def plot(
        self,
        x_values: Union[Sequence[Union[float, int]], NDArray],
        y_values: Union[Sequence[Union[float, int]], NDArray],
        x_demo: Union[float, int, None],
        y_demo: Union[float, int, None],
        figsize: Optional[tuple] = None,
        save_file: bool = True,
        file_name: str = "likelihood.png",
        save_path: Optional[str] = None,
        show_plot: bool = False,
    ) -> Figure:
        """
        Illustrates how is likelihood different from probability.

        Parameters
        ----------
        x_values : numpy.typing.NDArray, or a list of floats or ints
            Values in X axis.
        y_values : numpy.typing.NDArray, or a list of floats or ints
            Values in Y axis.
        x_demo : float, int, or None
            The value on X axis, which we want to highlight for illustration.
            If no value is provided, nothing will be highlighted.
        y_demo : float, int, or None
            Same as `x_demo` but in the direction of Y axis.
        figsize : tuple
        file_name : str
            Name of the image file to be saved.
        save_path : str
            Default value is none; but if value is provided then will save
            image to this path.
        show : bool
            Shows plot when set to True.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # Process data
        if x_demo is not None and y_demo is not None:
            raise ValueError('`x_demo` and `y_demo` cannot be both assigned values.')
        if x_demo is None and y_demo is None:
            x_values_common = x_values
            x_values_highlight = None

            y_values_common = y_values
            y_values_highlight = None

        if x_demo is None and y_demo is not None:
            y_values_common, y_values_highlight = self._split_data(y_values, y_demo)
            x_values_common = x_values
        if x_demo is not None and y_demo is None:
            x_values_common, x_values_highlight = self._split_data(x_values, x_demo)
            y_values_common = y_values

        # Generate data
        x, y, z = self._gen_3d_pmf_data(self.dist, x_values_common, y_values_common)
        bottom = np.zeros_like(z)
        width = depth = 1

        # Plot: common areas
        # ------------------
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

        ax.bar3d(x, y, bottom, width, depth, z, shade=True, alpha=0.3,
                 color="yellow")
        ax.set_title("Likelihood and Probability")
        ax.set_xlabel('X')
        ax.set_ylabel('Lambda')
        ax.set_box_aspect([1, 1, 1]) # ch!!!

        # Plot: highlighted areas
        # -----------------------
        if x_demo is None and y_demo is not None:
            x, y, z = self._gen_3d_pmf_data(self.dist, x_values_common,
                                            y_values_highlight)
            z_label = 'Probability'
            ax.set_zlabel(z_label)

        if x_demo is not None and y_demo is None:
            x, y, z = self._gen_3d_pmf_data(self.dist, x_values_highlight,
                                            y_values_common)
            z_label = 'Likelihood'
            ax.set_zlabel(z_label)

        bottom = np.zeros_like(z)
        width = depth = 1
        ax.bar3d(x, y, bottom, width, depth, z, shade=True, color="red",
                 edgecolor='black')

        if x_demo is None and y_demo is not None:
            # Calculate area under the curve
            x_diff = np.diff(np.sort(x), prepend=0)
            auc = np.sum(np.array(z - bottom) * x_diff)
            ax.text2D(0.05, 0.95, f"AUC of {z_label}: {auc:.2f}",
                      transform=ax.transAxes)

        if x_demo is not None and y_demo is None:
            # Calculate area under the curve
            y_diff = np.diff(np.sort(y), prepend=0)
            auc = np.sum(np.array(z - bottom) * y_diff)
            ax.text2D(0.05, 0.95, f"AUC of {z_label}: {auc:.2f}",
                      transform=ax.transAxes)

        # Save figure
        if save_file:
            save_viz(fig, file_name=file_name, save_path=save_path)

        if show_plot:
            plt.show()

        return fig

    def _split_data(
        self,
        axis_values: Union[Sequence[Union[float, int]], NDArray],
        axis_demo: Union[float, int, None]
    ) -> tuple:
        """
        Split values along an axis into "common" and "highlight".

        Parameters
        ----------
        axis_values : numpy.typing.NDArray, or a list of floats or ints
        axis_demo : float, int, or None

        Returns
        -------
        A tuple containing:
        1. A list of values on axis that will be part of "common display".
        2. A list of a single value on axis that will be highlighted for
        illustration.
        """
        axis_values_common = [v for v in axis_values if v != axis_demo]
        axis_values_highlight = [v for v in axis_values if v == axis_demo]

        return axis_values_common, axis_values_highlight

    def _gen_3d_pmf_data(
        self,
        dist: dict,
        x_values: Union[Sequence[Union[float, int]], NDArray],
        y_values: Union[Sequence[Union[float, int]], NDArray],
    ) -> tuple:
        """
        Parameters
        ----------
        dist : dict
            Containing 2 keys, "name" and "parameters".
        x_values : numpy.typing.NDArray, or a list of floats or ints.
            Values in X axis.
        y_values : numpy.typing.NDArray, or a list of floats or ints.
            Values in Y axis.

        Returns
        -------
        A tuple containing coordinates data in each axis for 3D plotting.
        """
        if dist['name'] in ['Poisson', 'poisson']:
            _x = x_values
            _l = y_values

            _xx, _ll = np.meshgrid(_x, _l)

            x_cord = _xx.ravel()
            l_cord = _ll.ravel()
            z_cord = stats.poisson.pmf(x_cord, l_cord)
        else:
            raise ValueError("Unsupported distribution.")

        return x_cord, l_cord, z_cord
