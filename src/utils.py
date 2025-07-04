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
