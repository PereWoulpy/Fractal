# Fractal

## Description

This little project is an exercise to make me use GPU computation.
This program can generate a fractal and allows to navigate inside

## Build
In order to use CUDA for the GPU, you need to use gcc version 6 and not higher.
The script `init.sh` can be personalized for your system

```sh
./init.sh
cmake .
make
```

## Instruction
Run the program with `./fractal`. An X11 window should pop-up.

With the key `c`, by pointing with your cursor, you can change the center of 
vision to this point.

Keys `i` and `o` are respectively for zooming in and out.

Key `s` will save the picture in a PPM format image file. Beware when running 
multiple times the program, the name of the files are reused and it will rewrite 
over existing files.

## External library
Graphic interface is made thanks a little X11 framework : 
[gfx](http://www.nd.edu/~dthain/courses/cse20211/fall2011/gfx).
It is not fancy stuff but works perfectly fine.
