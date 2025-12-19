// cpuProcessor.cpp

#include "cpuProcessor.h"
#include <algorithm>
#include <cstddef>

cpuProcessor::cpuProcessor() {
}

cpuProcessor::~cpuProcessor() {
}

// Simple box blur on CPU (naive implementation)
void cpuProcessor::cpuapplyBlur(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    if (!h_in || !h_out || width <= 0 || height <= 0 || channels < 3) return;
    int kernelSize = 2 * radius + 1;
    int kernelArea = kernelSize * kernelSize;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int rAcc = 0, gAcc = 0, bAcc = 0, aAcc = 0, count = 0;
            for (int ky = -radius; ky <= radius; ++ky) {
                int ny = y + ky;
                if (ny < 0 || ny >= height) continue;
                for (int kx = -radius; kx <= radius; ++kx) {
                    int nx = x + kx;
                    if (nx < 0 || nx >= width) continue;
                    int idx = (ny * width + nx) * channels;
                    bAcc += h_in[idx + 0];
                    gAcc += h_in[idx + 1];
                    rAcc += h_in[idx + 2];
                    if (channels == 4) aAcc += h_in[idx + 3];
                    ++count;
                }
            }
            int outIdx = (y * width + x) * channels;
            h_out[outIdx + 0] = static_cast<unsigned char>(bAcc / count);
            h_out[outIdx + 1] = static_cast<unsigned char>(gAcc / count);
            h_out[outIdx + 2] = static_cast<unsigned char>(rAcc / count);
            if (channels == 4) h_out[outIdx + 3] = static_cast<unsigned char>(aAcc / count);
        }
    }
}

// Variants simply dispatch to the base blur for now
void cpuProcessor::cpuapplyBlur_a(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    cpuapplyBlur(h_in, h_out, width, height, channels, radius);
}

void cpuProcessor::cpuapplyBlur_b(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius) {
    cpuapplyBlur(h_in, h_out, width, height, channels, radius);
}

void cpuProcessor::cpuapplyBW(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels) {
    if (!h_in || !h_out || width <= 0 || height <= 0 || channels < 3) return;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            unsigned char b = h_in[idx + 0];
            unsigned char g = h_in[idx + 1];
            unsigned char r = h_in[idx + 2];
            int gray = static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b);
            unsigned char gVal = static_cast<unsigned char>(std::min(255, gray));
            h_out[idx + 0] = gVal;
            h_out[idx + 1] = gVal;
            h_out[idx + 2] = gVal;
            if (channels == 4) {
                h_out[idx + 3] = h_in[idx + 3];
            }
        }
    }
}