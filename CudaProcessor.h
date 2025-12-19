// CudaProcessor.h

#pragma once
#include <vector>

#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>)
#    include <cuda_runtime.h>
#    define HAVE_CUDA 1
#  endif
#endif

#ifndef HAVE_CUDA
// If CUDA headers aren't available, provide minimal fallback so headers compile.
typedef int cudaError_t;
#endif

// Forward declare the kernel to avoid including unnecessary headers in other translation units
struct cudaGraphicsResource;

class CudaProcessor {
public:
    CudaProcessor();
    ~CudaProcessor();

    // Processes an image from raw pixel data
    void applyBlur(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void applyBlur_a(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void applyBlur_b(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);

    void applyMixedFilter(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);
    void applySepia(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);
    void applyInvert(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);

    // Calculates a histogram for one channel
    std::vector<int> calculateHistogram(const unsigned char* h_img, int width, int height, int channels, int channel_idx);

    // Return true if a real CUDA device is available and will be used
    bool isCudaAvailable();

private:
    // Helper for CUDA error checking
    void checkCudaError(cudaError_t err, const char* file, int line);

#if defined(HAVE_CUDA)
    // Persistent device buffers to avoid repeated cudaMalloc/cudaFree
    unsigned char* d_in_buffer = nullptr;
    unsigned char* d_out_buffer = nullptr;
    size_t d_buffer_size = 0; // in bytes
    cudaStream_t stream = 0;
#endif
};