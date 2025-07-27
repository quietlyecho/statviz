#!/usr/bin/env python3

import argparse

import numpy as np

from central_limit_theorem_plot import CentralLimitTheoremPlot
from confidence_interval_plot import ConfidenceIntervalPlot
from hypothesis_testing_plot import HypothesisTestingPlot
from likelihood_plot import LikelihoodPlot

parser = argparse.ArgumentParser()

parser.add_argument('concept',
                    help='Name of the statistics concept to visualize.')

args = parser.parse_args()



def main(concept: str):
    """
    Entry point.

    Parameters
    ----------
    concept : str
        Determine what stat concept to visualize.
    """
    print(f"Plotting {concept}...")

    if concept in ('central_limit_theorem', 'clt'):
        stat_viz_obj = CentralLimitTheoremPlot(
            sig_level=0.05,
            pop_dist={'name': 'beta', 'parameters': {'a': 0.5, 'b': 0.5}},
            sample_size=100,
            draw_number=3000,
            random_seed=42,
        )

        stat_viz_obj.plot(size_pop=5000)

    elif concept in ('confidence_interval', 'ci'):
        stat_viz_obj = ConfidenceIntervalPlot(
            sig_level=0.05,
            pop_dist={'name': 'beta', 'parameters': {'a': 0.5, 'b': 0.5}},
            sample_size=100,
            draw_number=3000,
            random_seed=42,
        )

        stat_viz_obj.plot(size_pop=5000)

    elif concept in ('hypothesis_testing', 'ht'):
        stat_viz_obj = HypothesisTestingPlot(
            sig_level=0.05,
            sample_size=100,
            draw_number=3000,
            random_seed=42,
        )

        stat_viz_obj.plot()

    elif concept in ('likelihood', 'l'):
        dist = {
            "name": "poisson",
            "parameters": {}
        }
        stat_viz_obj = LikelihoodPlot(dist=dist)

        stat_viz_obj.plot(
            x_values=np.arange(0, 25),
            y_values=np.arange(1, 6),
            x_demo=5,
            y_demo=None,
            save_file=False,
            show_plot=True
        )

    print("Plotting done")

if __name__ == '__main__':
    main(args.concept)
