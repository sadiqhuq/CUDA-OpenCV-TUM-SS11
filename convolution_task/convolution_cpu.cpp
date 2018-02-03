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
const char* cpu_studentName = "Sadiq Huq";
const int   cpu_studentID   = 3273623;
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
//  float kcenterX = kWidth/2;
//  float kcenterY = kHeight/2;

float sum = 0;
  // ### build a normalized gaussian kernel ###
  for (int i=0;i<kHeight;i++)
  {
	  for (int j=0;j<kWidth;j++)
	  {
		  kernel[i*kWidth+j] = exp ( -0.5 * (
				  ( (i-kRadiusX)*(i-kRadiusX) ) / (sigmaX*sigmaX)
				  + ( (j-kRadiusY)*(j-kRadiusY) ) / (sigmaY*sigmaY) )  );
		  sum += kernel[i*kWidth+j] ;
	  }
  }

  for (int i=0;i<kHeight;i++)
	  for (int j=0;j<kWidth;j++)
		  kernel[i*kWidth+j] = kernel[i*kWidth+j] / sum;

  return kernel;
}


// the following kernel normalization is only for displaying purposes with openCV
float *cpu_normalizeGaussianKernel_cv(const float *kernel, float *kernelImgdata, int kRadiusX, int kRadiusY, int step)
{
  int i,j;
  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  // the kernel is assumed to be decreasing when going outwards from the center and rotational symmetric
  // for normalization we subtract the minimum and divide by the maximum
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
	const int kWidth  = (kRadiusX<<1) + 1;
	const int kHeight = (kRadiusY<<1) + 1;
	// ### implement a convolution ###

int xk, yk,xi,yi,temp_x,temp_y;
float sum;

	for (yi=0;yi<iHeight;yi++)
	{
		for (xi=0;xi<iWidth;xi++)
		{
			sum=0;
			for (yk=0;yk<kHeight;yk++)
			{
				for (xk=0;xk<kWidth;xk++)
				{
					temp_x = xi + xk - kRadiusX;
					temp_y = yi + yk - kRadiusY;

					if(temp_x < 0)
						temp_x=0;
					if(temp_x >= iWidth)
						temp_x=iWidth-1;
					if(temp_y < 0)
						temp_y=0;
					if(temp_y >= iHeight)
						temp_y=iHeight-1;

					sum = sum+kernel[yk*kWidth+xk]*inputImage[temp_x+temp_y*iWidth];
				}
			}
			outputImage[yi*iWidth+xi] = sum;
		}
	}

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
