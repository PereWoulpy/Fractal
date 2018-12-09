#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "cudaFractal.h"
#include "utils.h"

#define NUM_THREADS_PER_BLOCK 256
#define NUM_COLOR 3

#define SATURATION 1.F
#define VALUE 0.8F

__global__ void
drawFractal(char *out, double center_x, double center_y, double w_real, double h_real, int w_image, int h_image,
            int max_iter) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < h_image * w_image) {
        int image_x = i % w_image;
        int image_y = i / w_image;
        double c_x = fma((double) image_x, w_real / w_image, center_x - w_real / 2.0);
        double c_y = fma((double) (h_image - image_y), h_real / h_image, center_y - h_real / 2.0);

        float iter = 0;
        double z_x = 0;
        double z_y = 0;
        while (iter < max_iter && (z_x * z_x + z_y * z_y) < 4) {
            iter++;
            double tmp = z_x;
            z_x = z_x * z_x - z_y * z_y + c_x;
            z_y = 2 * z_y * tmp + c_y;
        }

        int h = (int) (iter * 240 / max_iter);
        int h_ = (h / 60) % 6;
        float f = ((float) h / 60.F) - (float) h_;
        float l = VALUE * (1.F - SATURATION);
        float m = VALUE * (1.F - f * SATURATION);
        float n = VALUE * (1.F - (1.F - f) * SATURATION);

        int color_index = i * 3;

        switch (h_) {
            case 0:
                out[color_index] = (char) (VALUE * 255.F);
                out[color_index + 1] = (char) (n * 255.F);
                out[color_index + 2] = (char) (l * 255.F);
                break;
            case 1:
                out[color_index] = (char) (m * 255.F);
                out[color_index + 1] = (char) (VALUE * 255.F);
                out[color_index + 2] = (char) (l * 255.F);
                break;
            case 2:
                out[color_index] = (char) (l * 255.F);
                out[color_index + 1] = (char) (VALUE * 255.F);
                out[color_index + 2] = (char) (n * 255.F);
                break;
            case 3:
                out[color_index] = (char) (l * 255.F);
                out[color_index + 1] = (char) (m * 255.F);
                out[color_index + 2] = (char) (VALUE * 255.F);
                break;
            case 4:
                out[color_index] = (char) (n * 255.F);
                out[color_index + 1] = (char) (l * 255.F);
                out[color_index + 2] = (char) (VALUE * 255.F);
                break;
            case 5:
                out[color_index] = (char) (VALUE * 255.F);
                out[color_index + 1] = (char) (l * 255.F);
                out[color_index + 2] = (char) (m * 255.F);
                break;
        }
    }
}

char *create_fractal() {
    std::cout << "center : " << center_x << " " << center_y << std::endl;
    std::cout << "dimension : " << r_width << " " << r_height << std::endl;
    std::cout << "max iteration : " << max_iter << std::endl;

    //calling the kernel !
    drawFractal << < nb_block, nb_threads >> >
                               (deviceImage, center_x, center_y, r_width, r_height, width, height, max_iter);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    size_t size = height * width * sizeof(char) * NUM_COLOR;
    HANDLE_ERROR(cudaMemcpy(hostImage, deviceImage, size, cudaMemcpyDeviceToHost));

    return hostImage;
}

void init_gpu(int w, int h) {
    height = h;
    width = w;

    r_height = 2.5;
    r_width = r_height * (double) width / (double) height;

    std::cout << "dimension " << r_width << " " << r_height << std::endl;

    size_t size = height * width * sizeof(char) * NUM_COLOR;

    hostImage = (char *) malloc(size);
    HANDLE_ERROR(cudaMalloc(&deviceImage, size));

    nb_block = ceil((double) (height * width) / (double) NUM_THREADS_PER_BLOCK);
    nb_threads = NUM_THREADS_PER_BLOCK;
}

void delete_gpu() {
    cudaFree(deviceImage);
    free(hostImage);
}

void set_center(int pos_x, int pos_y) {
    center_x = ((double) pos_x / (double) width * r_width) + center_x - r_width / 2.0;
    center_y = ((double) (height - pos_y) / (double) height * r_height) + center_y - r_height / 2.0;
}

void set_zoom_scale(double scale) {
    r_width *= scale;
    r_height *= scale;

    max_iter += 20 * ((scale < 1) ? 1 : -1);
}
