import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

DEFAULT_SAMPLE_SIZE = 30


class CentralLimitTheoremPlot:
    """
    A class to demonstrate the central limit theorem.
    """

    def __init__(
        self,
        sig_level: float = 0.05,
        pop_dist: str = 'beta',
        sample_size: int = 30,
        draw_number: int = 500,
        random_seed: int = 42,
        save_path: str = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        sig_level: float
            Significance level defined in the study.
        pop_dist: str
            Distribution name of the population distribution.
        sample_size: int
            Sample size
        draw_number: int
            Number of times of drawing samples.
        random_seed: int
        save_path: str
            Path to save plots.
        """
        self.sig_level = sig_level
        self.pop_dist = pop_dist
        self.sample_size = sample_size
        self.draw_number = draw_number
        self.random_seed = random_seed
        self.kwargs = kwargs
        self.save_path = save_path

        self.rng = np.random.default_rng(random_seed)

    def get_pop_mean_and_var(self):
        """
        Get mean and variance of population distribution.
        """
        if self.pop_dist == 'beta':
            a = self.kwargs['a']
            b = self.kwargs['b']
            mu = (a + b) / 2
            var = (a * b) / ((a + b) ** 2 * (a + b + 1))
            return mu, var
        else:
            raise ValueError('Currently only supports beta population'
                             'distribution')

    def simulate_sample_draws(self, draw_number, sample_size):
        """
        Draw samples independently for `draw_number` times.

        Parameters
        ----------
        draw_number: int
            Number of samples to draw.
        sample_size: int
            Sample size.

        Returns
        -------
        samples: numpy.array
            A numpy array of drawn samples, of shape
            (draw_number, sample_size)
        """
        samples = np.array(
            [self.draw_one_sample(sample_size) for t in range(draw_number)]
        )
        return samples

    def draw_one_sample(self, size: int = DEFAULT_SAMPLE_SIZE):
        """
        Draw a single sample from the population distribution, with sample
        size of `size`.

        Parameters
        ----------
        size: int
            Sample size, default value is 30.

        Returns
        -------
        sample: numpy.array
            A numpy array of shape `size`
        """
        if size is not None and size > 0:
            size = size
        elif size == 0:
            raise ValueError('Invalid size, must be positive integers.')
        else:
            size = self.sample_size

        if self.pop_dist == 'beta':
            # Get key word arguments
            a = self.kwargs['a']
            b = self.kwargs['b']

            sample = self.rng.beta(a=a, b=b, size=size)

        else:
            raise ValueError('Currently only supports beta population'
                             'distribution')
        return sample

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
        bins_splg_dist: np.array = np.linspace(0.2, 0.8, num=50),
        save_path: str = None,
    ):
        """
        Plot the population distribution and sampling distribution

        Parameters
        ----------
        size_pop: int
            Size of the sample to plot an approximate population distribution.
        bins_splg_dist: numpy.array
            A numpy array that stores the intervals for plotting histogram.
        save_path: str
            The path to save the plotted image. Default is to not save.

        Returns
        -------
        fig: Figure
        """
        mu, var = self.get_pop_mean_and_var()
        samples = self.simulate_sample_draws(
            draw_number=self.draw_number,
            sample_size=self.sample_size,
            )
        sample_means = np.mean(samples, axis=1)

        fig, ((ax_pop, ax_blank), (ax_splg, ax_norm_splg)) = plt.subplots(
            nrows=2, ncols=2, figsize=(10, 10), sharex='col',
            gridspec_kw=dict(height_ratios=[1, 1]),
        )

        # Ax plot population distribution
        sample_for_pop = self.draw_one_sample(size=size_pop)
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

        if save_path is not None:
            # Save figure
            fig.savefig(save_path)
            print(f'Image saved at {save_path}')
        else:
            fig.savefig(self.save_path)
            print(f'Image saved at {self.save_path}')

        return fig
