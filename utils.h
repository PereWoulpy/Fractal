#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define HANDLE_ERROR(err) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}