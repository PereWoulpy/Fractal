#pragma once

#include <vector_types.h>

static char *hostImage;
static char *deviceImage;
static dim3 nb_block;
static dim3 nb_threads;

static int width;
static int height;

static double r_width;
static double r_height;

static double center_x = 0;
static double center_y = 0;

static int max_iter = 100;

char* create_fractal();

void init_gpu(int w, int h);

void delete_gpu();

void set_center(int pos_x, int pos_y);

void set_zoom_scale(double scale);