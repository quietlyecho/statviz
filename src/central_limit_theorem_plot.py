from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils import simulate_sample_draws, draw_one_sample, save_viz

DEFAULT_SAMPLE_SIZE = 30


class CentralLimitTheoremPlot:
    """
    A class to demonstrate the central limit theorem.
    """

    def __init__(
        self,
        sig_level: float = 0.05,
        pop_dist: dict = {'name': 'beta', 'parameters': {'a': 0.5, 'b': 0.5}},
        sample_size: int = 30,
        draw_number: int = 500,
        random_seed: int = 42,
        **kwargs,
    ):
        """
        Parameters
        ----------
        sig_level: float
            Significance level defined in the study.
        pop_dist: dict
            Contains two keys, "name" and "parameters".
        sample_size: int
            Sample size
        draw_number: int
            Number of times of drawing samples.
        random_seed: int
        """
        self.sig_level = sig_level
        self.pop_dist = pop_dist
        self.sample_size = sample_size
        self.draw_number = draw_number
        self.random_seed = random_seed
        self.kwargs = kwargs

        self.rng = np.random.default_rng(random_seed)

        if pop_dist['name'] != 'beta':
            raise ValueError('Currently only supports beta distribution for'
                             'population')

    def get_pop_mean_and_var(self, pop_dist: Optional[dict] = None):
        """
        Get mean and variance of population distribution.
        """
        if pop_dist is None:
            pop_dist = self.pop_dist

        params = pop_dist['parameters']

        if pop_dist['name'] == 'beta':
            a = params['a']
            b = params['b']
            mu = (a + b) / 2
            var = (a * b) / ((a + b) ** 2 * (a + b + 1))
            return mu, var
        else:
            raise ValueError('Unsupported populationd distribution for now')

    def calc_std_err_from_pop_var(self, var, sample_size):
        """
        Calculate standard error from population variance.

        Parameters
        ----------
        var: float
            Variance of population distribution.
        sample_size: int
            Sample size of a randomly drawn sample.

        Returns
        -------
        std_err: float
            Standard error of estimator.
        """
        std_err = np.sqrt(var / sample_size)
        return std_err

    def plot(
        self,
        size_pop: int = 1000,
        bins_splg_dist: Union[int, Sequence[float], str, None] = None,
        file_name: str = "clt.png",
        save_path: Optional[str] = None,
    ):
        """
        Plot the population distribution and sampling distribution

        Parameters
        ----------
        size_pop : int
            Size of the sample to plot an approximate population distribution.
        bins_splg_dist : numpy.ndarray
            A numpy.ndarray that stores the intervals for plotting histogram.
        file_name : str
            Image file name.
        save_path : str
            If provided, image file will be saved to this path. Default path
            is "$HOME".

        Returns
        -------
        fig: matplotlib.figure.Figure
        """
        mu, var = self.get_pop_mean_and_var()
        samples = simulate_sample_draws(
            rng=self.rng,
            dist=self.pop_dist,
            draw_number=self.draw_number,
            sample_size=self.sample_size,
            )
        sample_means = np.mean(samples, axis=1)

        fig, ((ax_pop, ax_blank), (ax_splg, ax_norm_splg)) = plt.subplots(
            nrows=2, ncols=2, figsize=(10, 10), sharex='col',
            gridspec_kw=dict(height_ratios=[1, 1]),
        )

        # Ax plot population distribution
        sample_for_pop = draw_one_sample(self.rng, self.pop_dist, size_pop)
        ax_pop.hist(sample_for_pop, bins=20)
        ax_pop.set_title('Population distribution')
        ax_pop.axvline(x=mu, linestyle='--', c='red', lw=1.5)
        ax_pop.text(mu + 0.1, 200, f'Mean = {mu:.2f}')

        # Ax, top right set invisible
        ax_blank.set_visible(False)

        # Ax plot sampling distribution
        ax_splg.hist(sample_means, bins=bins_splg_dist)
        ax_splg.axvline(x=mu, linestyle='--', c='red', lw=1.5)
        ax_splg.set_title('Sampling distribution')
        ax_splg.axvline(x=mu, linestyle='--', c='red', lw=1.5)
        ax_splg.text(mu + 0.1, 200, f'Mean = {mu:.2f}')

        # Ax plot normalized sampling distribution
        std_err = self.calc_std_err_from_pop_var(var, self.sample_size)
        normed = (sample_means - mu) / std_err
        x_stdnorm = np.linspace(-2, 2, num=100)

        ax_norm_splg.hist(normed, bins=x_stdnorm, density=True)
        ax_norm_splg.axvline(x=0, linestyle='--', c='red', lw=1.5)
        ax_norm_splg.set_title('Normalized sampling distribution')
        y_stdnorm = norm.pdf(x_stdnorm)  # Theoretical curve of standard normal
        ax_norm_splg.plot(x_stdnorm, y_stdnorm, '-', lw=2)

        save_viz(fig, file_name=file_name, save_path=save_path)

        return fig
