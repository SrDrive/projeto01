//cpuProcessor.h

#pragma once


class cpuProcessor {
public:
    cpuProcessor();

    ~cpuProcessor();

    // Processes an image from raw pixel data
    void cpuapplyBlur(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void cpuapplyBlur_a(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void cpuapplyBlur_b(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels, int radius);
    void cpuapplyBW(const unsigned char* h_in, unsigned char* h_out, int width, int height, int channels);

private:
    // Helper for CUDA error checking
  
};