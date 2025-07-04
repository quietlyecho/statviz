from typing import Optional, Sequence, Union
import inspect

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from central_limit_theorem_plot import CentralLimitTheoremPlot
from utils import simulate_sample_draws, draw_one_sample

class HypothesisTestingPlot(CentralLimitTheoremPlot):

    def __init__(
        self,
        pop_dist_1: Optional[dict] = None,
        pop_dist_2: Optional[dict] = None,
        **kwargs
    ):
        parent_params = inspect.signature(CentralLimitTheoremPlot.__init__).parameters
        parent_kwargs = {k: v for k, v in kwargs.items() if k in parent_params}
        super().__init__(**parent_kwargs)

        if pop_dist_1 is None:
            pop_dist_1 = self._set_default_dist()
        if pop_dist_2 is None:
            pop_dist_2 = self._set_default_dist()

        self.pop_dist_1 = pop_dist_1
        self.pop_dist_2 = pop_dist_2

    def _set_default_dist(self):
        return {'name': 'beta', 'parameters': {'a': 0.5, 'b': 0.5}}


    def simulate_sample_draws_of_sample_mean_diff(self) -> np.ndarray:
        """
        Generate values for the statistic `sample_mean_1 - sample_mean_2`, where
        `sample_mean_1` and `sample_mean_2` are sample means based on population
        1 and 2 respectively.
        """
        samples_1 = simulate_sample_draws(self.rng, self.pop_dist_1, self.draw_number, self.sample_size)
        samples_2 = simulate_sample_draws(self.rng, self.pop_dist_2, self.draw_number, self.sample_size)
        samples_diff = samples_1 - samples_2

        return samples_diff

    def plot(
        self,
        size_pop: int = 1000,
        bins_splg_dist: Union[int, Sequence[float], str, None] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 15),
    ) -> Figure:
        """
        Returns
        -------
        fig: matplotlib.figure.Figure
        """
        mu_1, var_1 = self.get_pop_mean_and_var(self.pop_dist_1)
        mu_2, var_2 = self.get_pop_mean_and_var(self.pop_dist_2)

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

        # Create the 5 subplots
        ax_pop_dist_1 = fig.add_subplot(gs[0, 0])  # Top left
        ax_pop_dist_2 = fig.add_subplot(gs[0, 1])  # Top right
        ax_splg_dist_1 = fig.add_subplot(gs[1, 0])  # Middle left
        ax_splg_dist_2 = fig.add_subplot(gs[1, 1])  # Middle right
        ax_splg_dist_3 = fig.add_subplot(gs[2, :])  # Bottom spanning both columns

        # Plot first population distribution
        sample_for_pop_1 = draw_one_sample(self.rng, self.pop_dist_1, size_pop)
        ax_pop_dist_1.hist(sample_for_pop_1, bins=20)
        ax_pop_dist_1.set_title('Population distribution 1')
        ax_pop_dist_1.axvline(x=mu_1, linestyle='--', c='red', lw=1.5)
        ax_pop_dist_1.text(mu_1 + 0.1, 200, f'Mean = {mu_1:.2f}')

        # Plot 2nd population distribution
        sample_for_pop_2 = draw_one_sample(self.rng, self.pop_dist_2, size_pop)
        ax_pop_dist_2.hist(sample_for_pop_2, bins=20)
        ax_pop_dist_2.set_title('Population distribution 2')
        ax_pop_dist_2.axvline(x=mu_2, linestyle='--', c='red', lw=1.5)
        ax_pop_dist_2.text(mu_2 + 0.1, 200, f'Mean = {mu_2:.2f}')

        # Plot 1st population sampling dist of sample means
        samples_1 = simulate_sample_draws(
            rng=self.rng,
            dist=self.pop_dist_1,
            draw_number=self.draw_number,
            sample_size=self.sample_size,
            )
        sample_means_1 = np.mean(samples_1, axis=1)

        ax_splg_dist_1.hist(sample_means_1, bins=bins_splg_dist)
        ax_splg_dist_1.axvline(x=mu_1, linestyle='--', c='red', lw=1.5)
        ax_splg_dist_1.set_title('Sampling distribution')
        ax_splg_dist_1.text(mu_1 + 0.1, 200, f'Mean = {mu_1:.2f}')


        # Plot 2nd population sampling dist of sample means
        samples_2 = simulate_sample_draws(
            rng=self.rng,
            dist=self.pop_dist_2,
            draw_number=self.draw_number,
            sample_size=self.sample_size,
            )
        sample_means_2 = np.mean(samples_2, axis=1)

        ax_splg_dist_2.hist(sample_means_2, bins=bins_splg_dist)
        ax_splg_dist_2.axvline(x=mu_2, linestyle='--', c='red', lw=1.5)
        ax_splg_dist_2.set_title('Sampling distribution')
        ax_splg_dist_2.text(mu_2 + 0.1, 200, f'Mean = {mu_2:.2f}')

        # Plot last ax
        samples_diff = self.simulate_sample_draws_of_sample_mean_diff()
        sample_means_diff = np.mean(samples_diff, axis=1)

        ax_splg_dist_3.hist(sample_means_diff, bins=bins_splg_dist)
        ax_splg_dist_3.axvline(x=mu_1 - mu_2, linestyle='--', c='red', lw=1.5)
        ax_splg_dist_3.set_title('Sampling distribution')
        ax_splg_dist_3.text(mu_1 - mu_2 + 0.1, 200, f'Mean = {mu_1 - mu_2:.2f}')

        # Save figure
        self._save_viz(fig, save_path)

        return fig
