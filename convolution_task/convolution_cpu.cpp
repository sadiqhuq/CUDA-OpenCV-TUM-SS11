/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: convolution
* file:    convolution_cpu.cpp
*
* 
\********* PLEASE ENTER YOUR CORRECT STUDENT NAME AND ID BELOW **************/
const char* cpu_studentName = "John Doe";
const int   cpu_studentID   = 1234567;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* makeGaussianKernel
* cpu_convolutionGrayImage
*
\****************************************************************************/


#include "convolution_cpu.h"

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>


const char* cpu_getStudentName() { return cpu_studentName; };
int         cpu_getStudentID()   { return cpu_studentID; };
bool        cpu_checkStudentNameAndID() { return strcmp(cpu_studentName, "John Doe") != 0 && cpu_studentID != 1234567; };



float *makeGaussianKernel(int kRadiusX, int kRadiusY, float sigmaX, float sigmaY)
{
  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;
  float *kernel = new float[kWidth*kHeight];
  
  
  // ### build a normalized gaussian kernel ###


  return kernel;
}


// the following kernel normalization is only for displaying purposes with openCV
float *cpu_normalizeGaussianKernel_cv(const float *kernel, float *kernelImgdata, int kRadiusX, int kRadiusY, int step)
{
  int i,j;
  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  // the kernel is assumed to be decreasing when going outwards from the center and rotational symmetric
  // for normalization we substract the minimum and devide by the maximum
  const float minimum = kernel[0];                                      // top left pixel is assumed to contain the minimum kernel value
  const float maximum = kernel[2*kRadiusX*kRadiusY+kRadiusX+kRadiusY] - minimum;  // center pixel is assumed to contain the maximum value

  for (i=0;i<kHeight;i++) 
	  for (j=0;j<kWidth;j++) 
		  kernelImgdata[i*step+j] = (kernel[i*kWidth+j]-minimum) / maximum;

  return kernelImgdata;
}


// mode 0: standard cpu implementation of a convolution
void cpu_convolutionGrayImage(const float *inputImage, const float *kernel, float *outputImage, int iWidth, int iHeight, int kRadiusX, int kRadiusY)
{

  // ### implement a convolution ### 

}




void cpu_convolutionRGB(const float *inputImage, const float *kernel, float *outputImage, int iWidth, int iHeight, int kRadiusX, int kRadiusY)
{
  // for separated red, green and blue channels a convolution is straightforward by using the gray value convolution for each color channel
  const int imgSize = iWidth*iHeight;
  cpu_convolutionGrayImage(inputImage, kernel, outputImage, iWidth, iHeight, kRadiusX, kRadiusY);
  cpu_convolutionGrayImage(inputImage+imgSize, kernel, outputImage+imgSize, iWidth, iHeight, kRadiusX, kRadiusY);
  cpu_convolutionGrayImage(inputImage+(imgSize<<1), kernel, outputImage+(imgSize<<1), iWidth, iHeight, kRadiusX, kRadiusY);
}



void cpu_convolutionBenchmark(const float *inputImage, const float *kernel, float *outputImage,
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                              int numKernelTestCalls)
{
  clock_t startTime, endTime;
  float fps;

  startTime = clock();

  for(int c=0;c<numKernelTestCalls;c++)
    cpu_convolutionRGB(inputImage, kernel, outputImage, iWidth, iHeight, kRadiusX, kRadiusY);

  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  fprintf(stderr, "%f fps - cpu version\n",fps);
}
