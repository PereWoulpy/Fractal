#include <iostream>
#include <cmath>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>

#include "gfx.h"
#include "cudaFractal.h"

#define HEIGHT 768
#define WIDTH  1280

void draw_fractal(const char *fractal) {
    for (int j = 0; j < HEIGHT; ++j) {
        for (int i = 0; i < WIDTH; ++i) {
            gfx_color(fractal[i + j * WIDTH],
                      fractal[i + j * WIDTH + 1],
                      fractal[i + j * WIDTH + 2]);
            gfx_point(i, j);
        }
    }
}

void save_fractal(const char *fractal) {
    static int number = 0;
    char outputImageFile[50];
    sprintf(outputImageFile, "mandelbrot_%d.ppm", number);
    number++;

    FILE *fp = fopen(outputImageFile, "wb"); /* b - binary mode */
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(fractal, sizeof(char), HEIGHT * WIDTH * 3, fp);
    fclose(fp);
}

int main(int argc, char *argv[]) {
    char c;
    char *fractal;

    // Open a new window for drawing.
    gfx_open(WIDTH, HEIGHT, "Example Graphics Program");
    init_gpu(WIDTH, HEIGHT);

    draw_fractal(fractal = create_fractal());

    while (true) {
        // Wait for the user to press a character.
        c = gfx_wait();

        // Quit if it is the letter q.
        if (c == 'q') break;

        if (c == 'c') {
            std::cout << gfx_xpos() << " " << gfx_ypos() << std::endl;
            set_center(gfx_xpos(), gfx_ypos());
            draw_fractal(fractal = create_fractal());
        }

        if (c == 'o') {
            set_zoom_scale(1.5);
            draw_fractal(fractal = create_fractal());
        }
        if (c == 'i') {
            set_zoom_scale(0.66);
            draw_fractal(fractal = create_fractal());
        }
        if (c == 's') {
            save_fractal(fractal);
            std::cout << "fractal saved" << std::endl;
        }
    }

    delete_gpu();

    return 0;
}
