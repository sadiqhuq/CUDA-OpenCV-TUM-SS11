/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: convolution
* file:    convolution_cpu.h
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef __CONVOLUTION_CPU_H
#define __CONVOLUTION_CPU_H


const char* cpu_getStudentName();
int         cpu_getStudentID();
bool        cpu_checkStudentNameAndID();


float *makeGaussianKernel(int radiusX, int radiusY, float sigmaX, float sigmaY);

float *cpu_normalizeGaussianKernel_cv(const float *kernel, float *kernelImgdata, int kRadiusX, int kRadiusY, int step);

void cpu_convolutionGrayImage(const float *inputImage, const float *kernel, float *outputImage,
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY);

void cpu_convolutionRGB(const float *inputImage, const float *kernel, float *outputImage,
                        int iWidth, int iHeight, int kRadiusX, int kRadiusY);

void cpu_convolutionBenchmark(const float *inputImage, const float *kernel, float *outputImage,
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                              int numKernelTestCalls);

#endif // #ifndef __CONVOLUTION_CPU_H
