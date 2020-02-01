"""
Infinite logistics, the best kind!
"""

import numpy as np
from numba import njit
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show


def logistic_map(rate, z):
    return rate * z * (1 - z)


@njit
def limit_cycle(max_cycles, settling_iterations, z_init, rate_resolution):
    rates = np.linspace(0, 4, rate_resolution)
    zs = np.empty((max_cycles, rate_resolution))
    z = np.ones(rate_resolution) * z_init
    for iteration in range(settling_iterations + max_cycles):
        z = rates * z * (1 - z)
        if iteration >= settling_iterations:
            zs[iteration - settling_iterations] = z
    return rates, zs


def calculate_map():
    max_cycles = 250
    settling_iterations = 20000
    z_init = 0.5
    rate_resolution = 15000

    rates, zs = limit_cycle(max_cycles, settling_iterations, z_init, rate_resolution)
    # vline_stack shows cumulative values, so undo that operation
    zs[1:] = np.diff(zs, axis=0)

    data = {'x': rates}
    keys = list(data.keys())
    for cycle_index in range(max_cycles):
        key = 'c' + str(cycle_index)
        data[key] = zs[cycle_index]
        keys.append(key)
    return data, keys


def plot_map(data, keys):
    source = ColumnDataSource(data=data)
    plot = figure(plot_width=1400, plot_height=750)
    visual_options = dict(line_width=0.25, alpha=0.2)
    lines = plot.vline_stack(keys[1:], x=keys[0], source=source, **visual_options)
    return plot


if __name__ == '__main__':
    output_file("logistic_map.html")
    plot = plot_map(*calculate_map())
    show(plot)
