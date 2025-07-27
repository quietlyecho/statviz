import os
from typing import Optional

from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

def simulate_sample_draws(
    rng: np.random.Generator,
    dist: dict,
    draw_number: int,
    sample_size: int
) -> NDArray:
    """
    Draw samples independently for `draw_number` times.

    Parameters
    ----------
    rng : numpy.random.Generator
    dist : dict
        Contains 2 keys, "name" and "parameters".
    draw_number: int
        Number of samples to draw.
    sample_size: int
        Sample size.

    Returns
    -------
    samples: numpy.ndarray
        A numpy.ndarray of drawn samples, of shape
        (draw_number, sample_size)
    """
    samples = np.array(
        [draw_one_sample(rng, dist, sample_size)
         for t in range(draw_number)]
    )
    return samples


def draw_one_sample(
    rng: np.random.Generator, dist: dict, size: int,
) -> NDArray:
    """
    Draw a single sample from the population distribution, with sample
    size of `size`.

    Parameters
    ----------
    rng : numpy.random.Generator
    dist : dict
        Contains 2 keys, "name" and "parameters".
    size: int
        Sample size, default value is 30.

    Returns
    -------
    sample: numpy.ndarray
        A numpy.ndarray of shape `size`
    """
    if size <= 0:
        raise ValueError('Invalid size, must be positive integers.')

    params = dist['parameters']

    if dist['name'] == 'beta':
        a = params['a']
        b = params['b']
        sample = rng.beta(a=a, b=b, size=size)
    else:
        raise ValueError('Currently only supports beta population'
                         'distribution')

    return sample


def save_viz(
    fig: Figure,
    file_name: str = "default.png",
    save_path: Optional[str] = None
):
    """
    Saves visualization into an image file. If not explicit `save_path` is
    provided, file will be saved in the "$HOME" path, with `file_name`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    file_name : str
    save_path : str
        If provided, save to this path; default is None.
    """
    if save_path is not None:
        fig.savefig(save_path)
        print(f'Image saved at {save_path}')
    else:
        home_path = os.environ.get("HOME")
        if home_path is None or file_name is None:
            raise ValueError("home_path or file_name cannot be None")
        full_path = os.path.join(home_path, file_name)
        fig.savefig(full_path)
        print(f'Image saved at {full_path}')
