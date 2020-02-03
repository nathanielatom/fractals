"""
Infinite logistics, the best kind!

TODO: more functions! windows! humps!
TODO: calculate ratio of bifurcations
TODO: calculate FFT of each limit cycle, display against rate
TODO: occurrence count / distribution of each LC cardinality (function of natural numbers); 6 occurs at least twice
TODO: is secondary occurrence (say of LC3 in LC6) contain an exact reflection?

"""

import numpy as np
from numba import njit
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show


def logistic_map(rate, z):
    return rate * z * (1 - z)


@njit
def limit_cycle(max_cycles, settling_iterations, rates, zs, z):
    for iteration in range(settling_iterations + max_cycles):
        # z = rates * z * (1 - np.tanh(z))
        # z = rates * np.sin(z * np.pi) # unifurcations? maybe hint of complex numbers? # what happens at rate=1?
        z = rates * z * (1 - z)
        if iteration >= settling_iterations:
            zs[iteration - settling_iterations] = z
    return rates, zs


def calculate_map():
    max_cycles = 150
    settling_iterations = 50000
    z_init = 0.5
    rate_resolution = 20000
    rate_bound = 20

    base = 2
    rate_bound_exp = np.log(rate_bound + 1) / np.log(base)
    rates = base ** rate_bound_exp - np.logspace(rate_bound_exp, 0, rate_resolution, base=base)
    zs = np.empty((max_cycles, rate_resolution))
    z = np.ones(rate_resolution) * z_init

    rates, zs = limit_cycle(max_cycles, settling_iterations, rates, zs, z)
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
    # zero_line = plot.line([0, 1], [0, 0], **visual_options)
    lines = plot.vline_stack(keys[1:], x=keys[0], source=source, **visual_options)
    return plot


if __name__ == '__main__':
    output_file("logistic_map.html")
    plot = plot_map(*calculate_map())
    show(plot)
