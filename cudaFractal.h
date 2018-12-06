#pragma once

#include <vector_types.h>

static char *hostImage;
static char *deviceImage;
static dim3 nb_block;
static dim3 nb_threads;

static int width;
static int height;

static float r_width;
static float r_height;

static float center_x = 0;
static float center_y = 0;

static int max_iter = 100;

char* create_fractal();

void init_gpu(int w, int h);

void delete_gpu();

void set_center(int pos_x, int pos_y);

void set_zoom_scale(float scale);