#!/usr/bin/env python3

import argparse

from central_limit_theorem_plot import CentralLimitTheoremPlot
from confidence_interval_plot import ConfidenceIntervalPlot
from hypothesis_testing_plot import HypothesisTestingPlot

parser = argparse.ArgumentParser()

parser.add_argument('name', help='Name of the statistics concept to plot.')

args = parser.parse_args()



def main(name: str):
    """
    Entry point.

    Parameters
    ----------
    name : str
        Determine what stat concept to visualize.
    """
    print(f"Plotting {name}...")

    if name in ('central_limit_theorem', 'clt'):
        stat_viz_obj = CentralLimitTheoremPlot(
            sig_level=0.05,
            pop_dist={'name': 'beta', 'parameters': {'a': 0.5, 'b': 0.5}},
            sample_size=100,
            draw_number=3000,
            random_seed=42,
        )

        stat_viz_obj.plot(
            size_pop=5000,
            save_path='default.png'
        )

    elif name in ('confidence_interval', 'ci'):
        stat_viz_obj = ConfidenceIntervalPlot(
            sig_level=0.05,
            pop_dist={'name': 'beta', 'parameters': {'a': 0.5, 'b': 0.5}},
            sample_size=100,
            draw_number=3000,
            random_seed=42,
        )

        stat_viz_obj.plot(
            size_pop=5000,
            save_path='default.png'
        )

    elif name in ('hypothesis_testing', 'ht'):
        stat_viz_obj = HypothesisTestingPlot(
            sig_level=0.05,
            sample_size=100,
            draw_number=3000,
            random_seed=42,
        )

        stat_viz_obj.plot(save_path='default.png')

    print("Plotting done")

if __name__ == '__main__':
    main(args.name)
