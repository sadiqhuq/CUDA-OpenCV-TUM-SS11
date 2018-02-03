/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: diffusion
* file:    main.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "diffusion.cuh"
#include "highgui.h"
#include "cv.h"


using namespace std;

int main(int argc,char **argv) {
  //----------------------------------------------------------------------------
  // Initialization
  //----------------------------------------------------------------------------
  if (argc <= 1) {
    cout << "Too few arguments!\n" << endl;
    cout << "Usage: " << argv[0] << " [ImageName] [mode=0] [timeStepWidth=0.05] [Iterations=100] [regWeight=10] [l_iter=3] [overrelaxation=1]" << endl;
    cout << "   or: " << argv[0] << " [camera] [mode=3] [W=320] [H=240] [timeStepWidth=0.05] [Iterations=50] [regWeight=10] [l_iter=3]" << endl;
    cout << endl << "Available modes:" << endl;
    cout << "0 - linear isotropic diffusion - gradient descent" << endl;
    cout << "1 - non-linear isotropic diffusion - explicit" << endl;
    cout << "2 - non-linear isotropic diffusion - jacobi" << endl;
    cout << "3 - non-linear isotropic diffusion - SOR" << endl;
    return 1;
  }
  // check for student name and ID
  if (!checkStudentNameAndID()) std::cout << "WARNING: Please enter your correct student name and ID in file diffusion.cu!"  << std::endl;

  int mode, numIterations, lagged_iterations, cameraWidth, cameraHeight;
  float timeStepWidth, regWeight, overrelaxation;
  if (strcmp(argv[1], "camera") != 0) {
    mode = (argc > 2) ? atoi(argv[2]) : 0;
    timeStepWidth = (argc > 3) ? (float)atof(argv[3]) : 0.05f;
    numIterations = (argc > 4) ? atoi(argv[4]) : 100;
    regWeight = (argc > 5) ? (float)atof(argv[5]) : 10.0f;
    lagged_iterations = (argc > 6) ? atoi(argv[6]) : 3;
    overrelaxation = (argc > 7) ? (float)atof(argv[7]) : 1.0f;
  }
  else {
    mode = (argc > 2) ? atoi(argv[2]) : 3;
    cameraWidth = (argc > 3) ? atoi(argv[3]) : 320;
    cameraHeight = (argc > 4) ? atoi(argv[4]) : 240;
    timeStepWidth = (argc > 5) ? (float)atof(argv[5]) : 0.05f;
    numIterations = (argc > 6) ? atoi(argv[6]) : 50;
    regWeight = (argc > 7) ? (float)atof(argv[7]) : 10.0f;
    lagged_iterations = (argc > 8) ? atoi(argv[8]) : 3;
    overrelaxation = (argc > 9) ? (float)atof(argv[9]) : 1.0f; // notused
  }

  //----------------------------------------------------------------------------
  // Image File Mode
  //----------------------------------------------------------------------------
  if (strcmp(argv[1], "camera") != 0) {  
    int i,j,s;

    // Read Input Picture
    IplImage* img = cvLoadImage(argv[1],-1);
    if (!img) { std::cout << "Error: Could not open file" << std::endl; return 1; }

    const int imgHeight = img->height;
    const int imgWidth = img->width;
    const int imgSpectrum = img->nChannels;
    const int imageSize = imgHeight*imgWidth*sizeof(float)*imgSpectrum;
    const int step = img->widthStep/sizeof(uchar);
    uchar *imageData = (uchar *)img->imageData;

    // Initializa Output Picture
    IplImage* outputImg = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_32F,imgSpectrum);
    const int outputStep = outputImg->widthStep/sizeof(float);
    float *outputData = (float *)outputImg->imageData;

    float *interleavedImg = new float[imageSize];
    float *interleavedOutput = new float[imageSize];

    // Copy <- Input Image (RGB)
    for (i=0;i<imgHeight;i++) 
      for (j=0;j<imgWidth;j++) 
        for (s=0;s<imgSpectrum;s++)
          interleavedImg[i*imgWidth*imgSpectrum+j*imgSpectrum+s] = (float)imageData[i*step+j*imgSpectrum+s];

    gpu_diffusion(interleavedImg, interleavedOutput, imgWidth, imgHeight, imgSpectrum, 
      timeStepWidth, numIterations, regWeight, lagged_iterations, overrelaxation, mode);

    // Copy -> Output Image (RGB)
    for (i=0;i<imgHeight;i++) 
      for (j=0;j<imgWidth;j++) 
        for (s=0;s<imgSpectrum;s++)
          outputData[i*outputStep+j*imgSpectrum+s] = interleavedOutput[i*imgWidth*imgSpectrum+j*imgSpectrum+s]/255;

    delete[] interleavedImg;
    delete[] interleavedOutput;

    cvNamedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Original Image", 100, 100);
    cvShowImage("Original Image", img);
    cvNamedWindow("Output Image", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Output Image", 500, 100);
    cvShowImage("Output Image", outputImg);
    std::cout << std::endl << "Press any key on the image to exit..." << std::endl;
    cvWaitKey(0);

    cvReleaseImage(&img);
    cvReleaseImage(&outputImg);
  }  //endif image file
  //----------------------------------------------------------------------------
  // Camera Mode
  //----------------------------------------------------------------------------
  else {
    CvCapture* capture;
    IplImage *img;
    capture = cvCaptureFromCAM(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, cameraWidth );
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, cameraHeight );

    img = cvQueryFrame(capture);
    const int imgHeight     = img->height;
    const int imgWidth      = img->width;
    const int step          = img->widthStep/sizeof(uchar);
    const int imgSpectrum   = img->nChannels;
    uchar *imageData = (uchar *)img->imageData;

    const int imageSize = imgHeight*imgWidth*sizeof(float)*imgSpectrum;
    float *interleavedImg = new float[imageSize];
    float *interleavedOutput = new float[imageSize];

    IplImage* outputImg = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_32F,imgSpectrum);
    const int outputStep = outputImg->widthStep/sizeof(float);
    float *outputData = (float *)outputImg->imageData;

    cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Input", 50, 100);
    cvNamedWindow("Output", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Output", 450, 100);

    std::cout << std::endl << "Press ESC on the image to exit..." << std::endl;

    int i,j,s;
    while (cvWaitKey(30) != 27)
    {
      img = cvQueryFrame(capture);

      for (i=0;i<imgHeight;i++) 
        for (j=0;j<imgWidth;j++) 
          for (s=0;s<imgSpectrum;s++)
            interleavedImg[i*imgWidth*3+j*imgSpectrum+s] = (float)imageData[i*step+j*imgSpectrum+s];

      gpu_diffusion(interleavedImg, interleavedOutput, imgWidth, imgHeight, imgSpectrum, 
        timeStepWidth, numIterations, regWeight, lagged_iterations, overrelaxation, mode);

      for (i=0;i<imgHeight;i++) 
        for (j=0;j<imgWidth;j++) 
          for (s=0;s<imgSpectrum;s++)
            outputData[i*outputStep+j*imgSpectrum+s] = interleavedOutput[i*imgWidth*3+j*imgSpectrum+s]/255;

      cvShowImage("Input", img);
      cvShowImage("Output", outputImg);
    }

    delete[] interleavedImg;
    delete[] interleavedOutput;
    cvReleaseCapture(&capture);
    //cvReleaseImage(&img);
    cvReleaseImage(&outputImg);
    cvDestroyAllWindows();
  } // endif "camera"

  return 0;
} 
