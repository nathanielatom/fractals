"""
Fractal F'ing Imaging
---------------------

... *the F stands for "Fractal F'ing Imag"* ...

Have fun with fractals! So far just Multibrot and Julia sets.

Run it with:

```
bokeh serve --show mandelbrot.py
```

# TODO: cleanup and document
# TODO: github repo
# TODO: add slider throttle to boost performance https://github.com/bokeh/bokeh/issues/4540
# TODO: try converge_threshold = 2; lookup and read proof
# TODO: visualize orbits (individual scatter points? multi-coloured?) on Julia set, or both??
# TODO: Try mandelbrots original iterative formula: z_new = h * z_prev * (1 - z_prev)
#       Why is it different? How do extra polynomial terms affect behaviour?
# TODO: iterate the inverse transformation z_prev = ±(z_new - c) ** 0.5, to get the Julia set boundary
# TODO: Investigate "Peitgen et al. (1992 p. 866) state in essence the following: if two buds
#       b1 and b3 have periodicities p and q, then the periodicity of the largest bud b2 that is smaller
#       than both b1 and b3 and which lies on the cardioid between them is equal to p + q."
# TODO: Investigate fixed points in C and rotational (non-attracting) orbits
# TODO: investigate alternative bokeh tickers to maybe allow for deeper max zoom
# TODO: figure out custom float128-like type (not supported by cuda)
# TODO: check-box for max iterations "Tied to Zoom", should it be linearly proportional?
# TODO: once zoomed in past a certain level, hide Julia set, and make Mandelbrot full-screen
# TODO: add indicator for active Julia crosshair
# TODO: profile/optimize cuda/bokeh - streams; hold/unhold?
# TODO: add indicator for Fractal Dimension(s) - start with Hausdorff, Frostman capacitary dimension
# TODO: add sliders = /UI for other fractal parameters (sin func? - see old notebook)
# TODO: add Mandelbar, Cubic, Lambda, Phoenix, Tetrate, Newton, Nova, Barnsley, Magnet
# TODO: add toggle for int8 vs int16 (likely require 2 allocated arrays)
# TODO: add logistic map, and estimator of Feigenbaum's constant, perhaps also along other directions than just the real line
# TODO: slider for z_0
# TODO: full 4D set described by bounded z_new = z_prev ** 2 + C; using
#       two plots for real, imag; or colour for mag, phase, showing full C plane or full z_0 plane
# TODO: convergent fractals and/or root-finding fractals
# TODO: animal-like fractal finder (image search, pre-trained conv net, or maybe just kNN)
# TODO: consider home server and domain / or AWS with ads

"""

import argparse

import numpy as np
from numba import njit
from numba import cuda
import math

from bokeh.plotting import ColumnDataSource, figure, curdoc
from bokeh.models import Slider, CrosshairTool, CustomJS
from bokeh.events import MouseMove, Tap
from bokeh.layouts import column, row, layout, gridplot
from bokeh.palettes import viridis


parser = argparse.ArgumentParser(
        description="Interactive Mandelbrot fractal app, zoom in and explore to your heart's desire!")
parser.add_argument(
        '--skip_julia',
        help="Don't show or calculate the corresponding Julia set for the crosshair.",
        action='store_true')
args = parser.parse_args()


gpu = cuda.is_available()
jitter = cuda.jit(device=True) if gpu else njit


@jitter
def abs2(z):
    return z.real * z.real + z.imag * z.imag


@jitter
def pow(z, exponent):
    r = abs2(z) ** (0.5 * exponent)
    theta = math.atan2(z.imag, z.real) * exponent
    return r * math.cos(theta) + 1j * r * math.sin(theta)

@jitter
def sin(z):
    return math.sin(z.real) * math.cosh(z.imag) + 1j * math.cos(z.real) * math.sinh(z.imag)

@jitter
def cos(z):
    return math.cos(z.real) * math.cosh(z.imag) - 1j * math.sin(z.real) * math.sinh(z.imag)

@jitter
def powcomp(a, exponent):
    # TODO: test
    return a ** exponent.real * math.cos(exponent.imag * math.log(a)) + 1j * math.sin(exponent.imag * math.log(a))

ei = np.e ** 1j

@jitter
def mandel(x, y, max_iters, converge_thresh, z_exponent, c_exponent):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.

    """
    # c = complex(x, y)
    # z = 0.0j
    z = complex(x, y)
    for i in range(max_iters):
        # try with e instead of sin or cos: math.exp(1j * math.pi * z)
        # z = ((7 * z + 2) - powcomp(ei, math.pi * z) * (5 * z + 2)) / 4 # collatz 1
        z = ((7 * z + 2) - np.exp(1j * math.pi * z) * (5 * z + 2)) / 4 # collatz 1
        # z = ((7 * z + 2) - cos(math.pi * z) * (5 * z + 2)) / 4 # collatz 1
        # z = (z / 2) * cos(math.pi / 2 * z) ** 2 + ((3 * z + 1) / 2) * sin(math.pi / 2 * z) ** 2 # complex collatz
        # z = z * sin(1 / z) + pow(c, c_exponent)

        # general mandelbrot formula
        # z = pow(z, z_exponent) + pow(c, c_exponent) # + math.pow(math.e, 1j * z) + math.pow(math.e, -1j * z)
        if abs2(z) >= converge_thresh:
            return i
    return max_iters


@jitter
def julia(c, x, y, max_iters, converge_thresh, z_exponent, c_exponent):
    """
    Given the real and imaginary parts of a complex number,
    and a given complex constant, determine if it is a candidate for
    membership in the Julia set given a fixed number of iterations.

    """
    z = complex(x, y)
    for i in range(max_iters):
        # Z <- Z ^ 2 + C
        z = pow(z, z_exponent) + pow(c, c_exponent) # + math.pow(math.e, 1j * z) + math.pow(math.e, -1j * z)
        if abs2(z) >= converge_thresh:
            return i
    return max_iters

@njit
def create_fractal(min_x, max_x, min_y, max_y, z_exponent, c_exponent, image, iters, converge_thresh):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(0, width):
        real = min_x + x * pixel_size_x
        for y in range(0, height):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, iters, converge_thresh, z_exponent, c_exponent)


@cuda.jit
def create_fractal_gpu(min_x, max_x, min_y, max_y, z_exponent, c_exponent, image, iters, converge_thresh):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, iters, converge_thresh, z_exponent, c_exponent)


@njit
def create_fractal_julia(c, min_x, max_x, min_y, max_y, z_exponent, c_exponent, image, iters, converge_thresh):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(0, width):
        real = min_x + x * pixel_size_x
        for y in range(0, height):
            imag = min_y + y * pixel_size_y
            image[y, x] = julia(c, real, imag, iters, converge_thresh, z_exponent, c_exponent)


@cuda.jit
def create_fractal_julia_gpu(c, min_x, max_x, min_y, max_y, z_exponent, c_exponent, image, iters, converge_thresh):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = julia(c, real, imag, iters, converge_thresh, z_exponent, c_exponent)


# if __name__ == '__main__':
# Static parameters
title = 'A Collatz Set'
if not args.skip_julia:
    h, w = 1024, 1280
else:
    h, w = 800, 1000
image = np.zeros((h, w), dtype=np.uint16) # 8 bit for overflow colours
image_julia = np.zeros((h, w), dtype=np.uint16) # 8 bit for overflow colours
blockdim = (32, 8)
griddim = (32, 16)
max_framerate = 10 # Hz
converge_threshold = 50 # 4 # 2

# Initial parameters
mandel_x_range = (-2.125, 1)
mandel_y_range = (-1.25, 1.25)
julia_x_range = (-2, 2)
julia_y_range = (-1.6, 1.6)
z_exponent = 2
c_exponent = 1
c_julia = 0 + 0j
max_iterations = 50

old_mandel_hash = hash((mandel_x_range, mandel_y_range, z_exponent, c_exponent, max_iterations))
old_julia_hash = hash((julia_x_range, julia_y_range, z_exponent, c_exponent, max_iterations, c_julia))

if gpu:
    gpu_image = cuda.to_device(image)
    create_fractal_gpu[griddim, blockdim](*mandel_x_range, *mandel_y_range, z_exponent, c_exponent, gpu_image, max_iterations, converge_threshold)
    gpu_image.to_host()
else:
    create_fractal(*mandel_x_range, *mandel_y_range, z_exponent, c_exponent, image, max_iterations, converge_threshold)

if not args.skip_julia:
    if gpu:
        gpu_image_julia = cuda.to_device(image_julia)
        create_fractal_julia_gpu[griddim, blockdim](c_julia, *julia_x_range, *julia_y_range, z_exponent, c_exponent, gpu_image_julia, max_iterations, converge_threshold)
        gpu_image_julia.to_host()
    else:
        create_fractal_julia(c_julia, *julia_x_range, *julia_y_range, z_exponent, c_exponent, image_julia, max_iterations, converge_threshold)

source = ColumnDataSource(data=dict(image=[image],
    x=[mandel_x_range[0]], y=[mandel_y_range[0]],
    dw=[mandel_x_range[1] - mandel_x_range[0]], dh=[mandel_y_range[1] - mandel_y_range[0]]))
mandelplot = figure(title=title, width=w, height=h, x_range=mandel_x_range, y_range=mandel_y_range, active_scroll='wheel_zoom')
mandelplot.image('image', x='x', y='y', dw='dw', dh='dh', palette=viridis(256), source=source)

# Cursor for Julia Set
hs = ColumnDataSource(data=dict(x=[c_julia.real], y=[c_julia.imag]))
def update_mouse(event):
    if mandelplot.toolbar.active_inspect == 'auto':
        hs.data.update(x=[event.x], y=[event.y])
def update_tap(event):
    if mandelplot.toolbar.active_inspect == 'auto':
        mandelplot.toolbar.active_inspect = None
    else:
        mandelplot.toolbar.active_inspect = 'auto'
        hs.data.update(x=[event.x], y=[event.y])
mandelplot.add_tools(CrosshairTool())
mandelplot.on_event(MouseMove, update_mouse)
mandelplot.on_event(Tap, update_tap)

source_julia = ColumnDataSource(data=dict(image=[image_julia],
    x=[julia_x_range[0]], y=[julia_y_range[0]],
    dw=[julia_x_range[1] - julia_x_range[0]], dh=[julia_y_range[1] - julia_y_range[0]]))
julia_plot = figure(title=f'Julia Set; c = {c_julia}', width=w, height=h, x_range=julia_x_range, y_range=julia_y_range, active_scroll='wheel_zoom')
julia_plot.image('image', x='x', y='y', dw='dw', dh='dh', palette=viridis(256), source=source_julia)

slider_exp_z = Slider(title="Z Exponent", start=0, end=10, value=z_exponent, step=0.01)
slider_exp_c = Slider(title="C Exponent", start=-10, end=10, value=c_exponent, step=0.01)
slider_max_i = Slider(title="Max Iterations", start=1, end=256 * 4, value=max_iterations, step=1)
slider_conv_thresh = Slider(title="Convergence Threshold", start=0, end=1000, value=converge_threshold, step=0.5)

mandelplot.tags = [0, old_mandel_hash]
julia_plot.tags = [0, old_julia_hash]
def update():
    z_exponent, c_exponent, max_iterations, converge_threshold = slider_exp_z.value, slider_exp_c.value, slider_max_i.value, slider_conv_thresh.value

    mandel_x_range = (mandelplot.x_range.start, mandelplot.x_range.end)
    mandel_y_range = (mandelplot.y_range.start, mandelplot.y_range.end)

    julia_x_range = (julia_plot.x_range.start, julia_plot.x_range.end)
    julia_y_range = (julia_plot.y_range.start, julia_plot.y_range.end)
    c_julia = complex(hs.data['x'][0], hs.data['y'][0])

    new_mandel_hash = hash((mandel_x_range, mandel_y_range, z_exponent, c_exponent, max_iterations))
    new_julia_hash = hash((julia_x_range, julia_y_range, z_exponent, c_exponent, max_iterations, c_julia))

    mandelplot.tags[0] += 1
    julia_plot.tags[0] += 1
    old_mandel_hash = mandelplot.tags[-1]
    old_julia_hash = julia_plot.tags[-1]

    if new_mandel_hash != old_mandel_hash:
        if gpu:
            create_fractal_gpu[griddim, blockdim](*mandel_x_range, *mandel_y_range, z_exponent, c_exponent, gpu_image, max_iterations, converge_threshold)
            gpu_image.to_host()
        else:
            create_fractal(*mandel_x_range, *mandel_y_range, z_exponent, c_exponent, image, max_iterations, converge_threshold)
        source.data.update(image=[image], x=[mandel_x_range[0]], y=[mandel_y_range[0]],
            dw=[mandel_x_range[1] - mandel_x_range[0]], dh=[mandel_y_range[1] - mandel_y_range[0]])
        mandelplot.tags[-1] = new_mandel_hash
        print(f'mandel event count: {mandelplot.tags[0]}')
    if new_julia_hash != old_julia_hash:
        if not args.skip_julia:
            julia_plot.title.text = f'Julia Set; c = {c_julia}'
            if gpu:
                create_fractal_julia_gpu[griddim, blockdim](c_julia, *julia_x_range, *julia_y_range, z_exponent, c_exponent, gpu_image_julia, max_iterations, converge_threshold)
                gpu_image_julia.to_host()
            else:
                create_fractal_julia(c_julia, *julia_x_range, *julia_y_range, z_exponent, c_exponent, image_julia, max_iterations, converge_threshold)
            source_julia.data.update(image=[image_julia], x=[julia_x_range[0]], y=[julia_y_range[0]],
                dw=[julia_x_range[1] - julia_x_range[0]], dh=[julia_y_range[1] - julia_y_range[0]])
            julia_plot.tags[-1] = new_julia_hash
            print(f'julia event count: {julia_plot.tags[0]}')

sliders = [column(slider_exp_z, slider_exp_c, slider_max_i, slider_conv_thresh)]
grid = gridplot([[mandelplot] + ([julia_plot] if not args.skip_julia else sliders), sliders if not args.skip_julia else []], sizing_mode='scale_width')

curdoc().add_periodic_callback(update, max_framerate ** -1 * 1000)
curdoc().add_root(grid)

