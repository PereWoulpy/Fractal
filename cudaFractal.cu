#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#include "cudaFractal.h"
#include "utils.h"

#define NUM_THREADS_PER_BLOCK 256

__global__ void
drawFractal(char *out, float center_x, float center_y, float w_real, float h_real, int w_image, int h_image, int max_iter) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < h_image * w_image) {
        int image_x = i % w_image;
        int image_y = i / w_image;
        float c_x = fmaf(image_x, w_real / w_image, center_x - w_real / 2);
        float c_y = fmaf(h_image - image_y, h_real / h_image, center_y - h_real / 2);

        float iter = 0;
        float z_x = 0;
        float z_y = 0;
        while (iter < max_iter && (z_x * z_x + z_y * z_y) < 4) {
            iter++;
            float tmp = z_x;
            z_x = z_x * z_x - z_y * z_y + c_x;
            z_y = 2 * z_y * tmp + c_y;
        }

        out[i] = (char) (iter * 255 / max_iter);
    }
}

char *create_fractal() {
    std::cout << "center : " << center_x << " " << center_y << std::endl;
    std::cout << "dimension : " << r_width << " " << r_height << std::endl;
    std::cout << "max iteration : " << max_iter << std::endl;

    //calling the kernel !
    drawFractal << < nb_block, nb_threads >> > (deviceImage, center_x, center_y, r_width, r_height, width, height, max_iter);

    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    size_t size = height * width * sizeof(char);
    HANDLE_ERROR(cudaMemcpy(hostImage, deviceImage, size, cudaMemcpyDeviceToHost));

    return hostImage;
}

void init_gpu(int w, int h) {
    height = h;
    width = w;

    r_height = 2.5F;
    r_width = r_height * (float) width / (float) height;

    std::cout << "dimension " << r_width << " " << r_height << std::endl;

    size_t size = height * width * sizeof(char);

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
    center_x = ((float) pos_x / (float) width * r_width) + center_x - r_width / 2.F;
    center_y = ((float) (height - pos_y) / (float) height * r_height) + center_y - r_height / 2.F;
}

void set_zoom_scale(float scale) {
    r_width *= scale;
    r_height *= scale;

    max_iter += 20 * ((scale < 1) ? 1 : -1);
}
