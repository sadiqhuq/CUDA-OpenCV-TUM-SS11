/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: gradient
* file:    main.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "gradient.cuh"

#include "highgui.h"
#include "cv.h"

using namespace std;




int main(int argc,char **argv) {
  if (argc <= 1) {
    cout << "Too few arguments!\n" << endl;
    cout << "Usage: " << argv[0] << "[ImageName] [mode=3]" << endl;
    cout << endl << "Available modes:" << endl;
    cout << "0 - X Derivative" << endl;
    cout << "1 - Y Derivative" << endl;
    cout << "2 - Gradient Magnitude" << endl;
    cout << "3 - ALL" << endl;
    
    return 1;
  }

  int i,j,s;
  const int mode = (argc > 2) ? atoi(argv[2]) : 3;
  
  if (!checkStudentNameAndID()) std::cout << "WARNING: Please enter your correct student name and ID in file gradient.cu!"  << std::endl;

  // Read Input Picture
  IplImage* img = cvLoadImage(argv[1],-1);
  if (!img) { std::cout << "Error: Could not open file" << std::endl; return 1; }

  const int imgHeight = img->height;
  const int imgWidth = img->width;
  const int imgSpectrum = img->nChannels;
  const int imageSize = imgHeight*imgWidth*imgSpectrum*sizeof(float);
  const int step = img->widthStep/sizeof(uchar);
  uchar *cvImageData = (uchar *)img->imageData;
  cvNamedWindow("Original Image", CV_WINDOW_AUTOSIZE);
  cvMoveWindow("Original Image", 100, 100);
  cvShowImage("Original Image", img);

  IplImage* derXImg = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_32F,imgSpectrum);
  IplImage* derYImg = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_32F,imgSpectrum);
  IplImage* magnImg = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_32F,imgSpectrum);
  const int outputStep = derXImg->widthStep/sizeof(float);
  float *cvDerXImgData = (float *)derXImg->imageData;
  float *cvDerYImgData = (float *)derYImg->imageData;
  float *cvMagnImgData = (float *)magnImg->imageData;

  float *imgData = new float[imageSize];
  float *derXImgData = new float[imageSize];
  float *derYImgData = new float[imageSize];
  float *magnImgData = new float[imageSize];

  for (i=0;i<imgHeight;i++)
    for (j=0;j<imgWidth;j++)
      for (s=0;s<imgSpectrum;s++)
        imgData[(i*imgWidth+j)*imgSpectrum+s] = (float)cvImageData[i*step+j*imgSpectrum+s];

  if(mode == 0) {

		gpu_derivative_sm_d(imgData, derXImgData, imgWidth, imgHeight, imgSpectrum, 0);

    for (i=0;i<imgHeight;i++)
      for (j=0;j<imgWidth;j++)
        for (s=0;s<imgSpectrum;s++)
          cvDerXImgData[i*outputStep+j*imgSpectrum+s] = derXImgData[(i*imgWidth+j)*imgSpectrum+s]/255;

    cvNamedWindow("Derivative X", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Derivative X", 500, 100);
    cvShowImage("Derivative X", derXImg);
  }
  else if (mode == 1) {
		
    gpu_derivative_sm_d(imgData, derYImgData, imgWidth, imgHeight, imgSpectrum, 1);

    for (i=0;i<imgHeight;i++)
      for (j=0;j<imgWidth;j++)
        for (s=0;s<imgSpectrum;s++)
          cvDerYImgData[i*outputStep+j*imgSpectrum+s] = derYImgData[(i*imgWidth+j)*imgSpectrum+s]/255;

    cvNamedWindow("Derivative Y", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Derivative Y", 500, 100);
    cvShowImage("Derivative Y", derYImg);
  }
  else if(mode == 2) {

		gpu_derivative_sm_d(imgData, magnImgData, imgWidth, imgHeight, imgSpectrum, 2);

    for (i=0;i<imgHeight;i++)
      for (j=0;j<imgWidth;j++)
        for (s=0;s<imgSpectrum;s++)
          cvMagnImgData[i*outputStep+j*imgSpectrum+s] = magnImgData[(i*imgWidth+j)*imgSpectrum+s]/255;

    cvNamedWindow("Magnitude", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Magnitude", 500, 100);
    cvShowImage("Magnitude", magnImg);
  }
  else if(mode == 3) {

    gpu_derivative_sm_d(imgData, derXImgData, imgWidth, imgHeight, imgSpectrum, 0);
    gpu_derivative_sm_d(imgData, derYImgData, imgWidth, imgHeight, imgSpectrum, 1);
    gpu_derivative_sm_d(imgData, magnImgData, imgWidth, imgHeight, imgSpectrum, 2);

    for (i=0;i<imgHeight;i++) {
      for (j=0;j<imgWidth;j++) {
        for (s=0;s<imgSpectrum;s++) {
          cvDerXImgData[i*outputStep+j*imgSpectrum+s] = derXImgData[(i*imgWidth+j)*imgSpectrum+s]/255;
          cvDerYImgData[i*outputStep+j*imgSpectrum+s] = derYImgData[(i*imgWidth+j)*imgSpectrum+s]/255;
          cvMagnImgData[i*outputStep+j*imgSpectrum+s] = magnImgData[(i*imgWidth+j)*imgSpectrum+s]/255;
        }
      }
    }

    cvNamedWindow("Derivative X", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Derivative X", 500, 100);
    cvShowImage("Derivative X", derXImg);

    cvNamedWindow("Derivative Y", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Derivative Y", 500, 100);
    cvShowImage("Derivative Y", derYImg);

    cvNamedWindow("Magnitude", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Magnitude", 500, 100);
    cvShowImage("Magnitude", magnImg);
  }

  std::cout << std::endl << "Press any key on the image to exit..." << std::endl;
  cvWaitKey(0);


  delete[] imgData;
  delete[] derXImgData;
  delete[] derYImgData;
  delete[] magnImgData;

  cvReleaseImage(&img);
  cvReleaseImage(&derXImg);
  cvReleaseImage(&derYImg);
  cvReleaseImage(&magnImg);

  return 0;
} 
