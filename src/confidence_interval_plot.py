"""
CONCEPT: Confidence Interval
"""

from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from central_limit_theorem_plot import CentralLimitTheoremPlot
from utils import simulate_sample_draws, draw_one_sample

class ConfidenceIntervalPlot(CentralLimitTheoremPlot):
    """
    A class to visualize confidence intervals and demonstrate their coverage
    probability.

    Inherits from `CentralLimitTheoremPlot` to leverage population distribution
    sampling and statistical calculations.
    """
    def plot(
        self,
        size_pop: int = 1000,
        bins_splg_dist: Union[int, Sequence[float], str, None] = None,
        save_path: Optional[str] = None,
    ):
        """
        Creates visualizations showing confidence intervals and their coverage
        probability.

        Parameters
        ----------
        size_pop : int, default=1000
            Size of the sample to approximate the population distribution.
        bins_splg_dist : np.array, default=np.linspace(0.2, 0.8, num=50)
            Bin edges for the sampling distribution histogram.
        save_path : str, optional
            Path to save the plot. If None, uses the default save_path.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        mu, var = self.get_pop_mean_and_var()
        samples = simulate_sample_draws(
            rng=self.rng,
            dist=self.pop_dist,
            draw_number=self.draw_number,
            sample_size=self.sample_size,
            )
        lower_bounds, sample_means, upper_bounds = self.get_conf_intvls(
            var=var,
            samples=samples,
        )
        pct_in = self.calc_pct_within_interval(mu, lower_bounds, upper_bounds)

        # Set up plots
        fig, (ax_pop, ax_intvls) = plt.subplots(
            nrows=2, ncols=1, figsize=(5, 15), sharex='col',
            gridspec_kw=dict(height_ratios=[1, 2], hspace=0),
        )

        # Plot population distribution
        sample_for_pop = draw_one_sample(self.rng, self.pop_dist, size_pop)
        ax_pop.hist(sample_for_pop)
        ax_pop.set_title('Population distribution')

        # Plot population mean
        ax_pop.axvline(x=mu, linestyle='--', c='red', lw=1.5)
        ax_pop.text(mu + 0.1, 200, f'Population mean\nmu = {mu:.2f}')

        # Plot confidence intervals
        self._plot_intervals(
            ax_intvls, mu, lower_bounds, sample_means, upper_bounds
        )

        # Plot population mean
        ax_intvls.axvline(x=mu, linestyle='--', c='red', lw=1.5)

        # Add texts
        ax_intvls.text(0, 0.5, f'Covering mu:\n{pct_in:.2%}')

        # Plot sampling distribution
        ax_intvls.hist(sample_means, bins=bins_splg_dist)
        ax_intvls.axvline(x=mu, linestyle='--', c='red', lw=1.5)

        # Save figure
        self._save_viz(fig, save_path)

        return fig


    def calc_pct_within_interval(self, mu, lower_bounds, upper_bounds):
        """
        Calculates the percentage of intervals containing the population mean.

        Parameters
        ----------
        mu: float
            Population mean.
        lower_bounds: numpy.ndarray
        upper_bounds: numpy.ndarray

        Returns
        -------
        pct_in: float
            Portion of intervals containing the population mean.
        """
        evals = (lower_bounds <= mu) & (mu <= upper_bounds)
        pct_in = evals.sum() / evals.shape[0]
        return pct_in

    def get_conf_intvls(self, var: float, samples: np.ndarray):
        """
        Calculates confidence intervals based on samples drawn.

        Parameters
        ----------
        var: float
            Population variance.
        samples: numpy.ndarray
            Output of `simulate_sample_draws` function.

        Returns
        -------
        A tuple of numpy.ndarray, each of shape (trial_num, ):
            1. lower_bounds
            2. sample_means
            3. upper_bounds
        """
        sample_means = np.mean(samples, axis=1)
        sample_size = samples.shape[1]
        std_err = self.calc_std_err_from_pop_var(var, sample_size)

        critical_value = norm.ppf(1 - self.sig_level / 2)

        lower_bounds = sample_means - critical_value * std_err
        upper_bounds = sample_means + critical_value * std_err

        return lower_bounds, sample_means, upper_bounds

    def _plot_intervals(
        self, ax, mu, lower_bounds, sample_means, upper_bounds
    ):
        """
        Plot confidence intervals on the given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to plot on.
        mu : float
            True population mean.
        lower_bounds : numpy.ndarray
            Lower bounds of confidence intervals.
        sample_means : numpy.ndarray
            Sample means for each interval.
        upper_bounds : numpy.ndarray
            Upper bounds of confidence intervals.

        Returns
        -------
        None
            Modifies the axis in place.
        """
        for i in range(lower_bounds.shape[0]):
            # Plot interval
            if mu >= lower_bounds[i] and mu <= upper_bounds[i]:
                color_int, color_mid = 'g', 'r'
            else:
                color_int, color_mid = 'y', 'r'

            ax.plot(
                np.array([lower_bounds[i], upper_bounds[i]]),
                np.repeat(i+1, 2),
                f'|-{color_int}'
            )
            # Plot sample mean
            ax.plot(sample_means[i], i+1, f'o-{color_mid}', markersize=1)
