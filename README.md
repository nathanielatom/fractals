# Fractals
A playground for interactively exploring fractals.

## Fractal F'ing Imaging

where the F stands for "Fractal F'ing Imag".

### Mandelbrot and Julia Sets

An interactive app using bokeh server, which recalculates on zoom. Uses numba cuda to accelerate calculation.

A [bunch](https://photos.app.goo.gl/mqa36t17Scu4g3428) of fun fractalscapes from exploring the Mandelbrot and Julia sets with generalized exponents. Some of the images with inverted colours are the happy result of an integer overflow bug. Although it's been fixed, they can be reproduced by using `dtype=np.uint8` for the image array. Looking for animals while exploring a fractal is even more fun than staring at the shapes of clouds!

### Logistic Map

Just a rough plot for now, to explore the limit cycle behaviour when functions other than quadratics are used.

## Usage

To run the server for remote connections:

```
bokeh serve mandelbrot.py --address "0.0.0.0" --allow-websocket-origin=<ip>:5006
```

