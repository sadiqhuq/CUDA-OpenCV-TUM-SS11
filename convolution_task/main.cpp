/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: convolution
* file:    main.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "convolution_cpu.h"
#include "convolution_gpu.cuh"
#include "highgui.h"
#include "cv.h"

using namespace std;

int main(int argc,char **argv) {
  //----------------------------------------------------------------------------
  // Initialization
  //----------------------------------------------------------------------------
  if (argc <= 1) {
    cout << "Too few arguments!\n" << endl;
    cout << "Usage: " << argv[0] << " [ImageName] [Mode=0] [Radius=5] [Sigma=10] [Interleaved=0] [NumBenchmarkCycles=0]" << endl;
    cout << "   or: " << argv[0] << " [camera] [Mode=5] [Radius=5] [Sigma=10] [Width=640] [Height=480]" <<endl;
    cout << endl << "Available modes:" << endl;
    cout << "0 - cpu only" << endl;
    cout << "1 - global memory only" << endl;
    cout << "2 - global memory for image & constant memory for kernel access" << endl;
    cout << "3 - shared memory for image & global memory for kernel access" << endl;
    cout << "4 - shared memory for image & constant memory for kernel access" << endl;
    cout << "5 - dyn. shared memory for image & const memory for kernel access" << endl;
    cout << "6 - texture memory for image & const memory for kernel access" << endl;
    cout << "interleaved=1 - for modes 5 and 6 only" << endl;
    return 1;
  }
  // check for student name and ID
  if (!cpu_checkStudentNameAndID()) std::cout << "WARNING: Please enter your correct student name and ID in file convolution_cpu.cpp!" << std::endl;
  if (!gpu_checkStudentNameAndID()) std::cout << "WARNING: Please enter your correct student name and ID in file convolution_gpu.cu!"  << std::endl;
  if (strcmp(cpu_getStudentName(), gpu_getStudentName()) != 0) std::cout << "WARNING: Student name mismatch in files convolution_cpu.cpp and convolution_gpu.cu" << std::endl;
  if (cpu_getStudentID() != gpu_getStudentID()) std::cout << "WARNING: Student ID mismatch in files convolution_cpu.cpp and convolution_gpu.cu" << std::endl;

  // read program arguments
  const int kRadiusX = (argc > 3) ? atoi(argv[3]) : 5;
  const int kRadiusY = kRadiusX;
  const float sigma = (argc > 4) ? (float)atof(argv[4]) : 10.0f;
  int mode, numBenchmarkCycles, cameraWidth, cameraHeight;
  bool interleavedMode;  
  if (strcmp(argv[1],"camera") != 0) {
    mode = (argc > 2) ? atoi(argv[2]) : 0;
    interleavedMode = ((argc > 5) ? atoi(argv[5]) != 0 : false);
    numBenchmarkCycles = (argc > 6) ? atoi(argv[6]) : 0;
  }
  else{
    mode = (argc > 2) ? atoi(argv[2]) : 5;
    if (mode != 5 && mode != 6) {
      std::cout << "camera - for modes 5 or 6 only" << std::endl;
      return 1;
    }
    interleavedMode = true;
    cameraWidth = (argc > 5) ? atoi(argv[5]) : 640;
    cameraHeight = (argc > 6) ? atoi(argv[6]) : 480;
  }

  // Compute Gaussian Kernel in CPU method  
  float *kernel = makeGaussianKernel(kRadiusX, kRadiusY, sigma, sigma);

  //----------------------------------------------------------------------------
  // Image File Mode
  //----------------------------------------------------------------------------
  if (strcmp(argv[1],"camera") != 0) {

    int i,j;

    // Read Input Picture
    IplImage* img = cvLoadImage(argv[1],-1);
    if (!img) { std::cout << "Error: Could not open file" << std::endl; return 1; }

    const int imgHeight = img->height;
    const int imgWidth = img->width;
    const int imgSpectrum = img->nChannels;
    const int imageSize = imgHeight*imgWidth*sizeof(float)*imgSpectrum;
    const int step = img->widthStep/sizeof(uchar);
    uchar *imageData = (uchar *)img->imageData;
    cvNamedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Original Image", 100, 100);
    cvShowImage("Original Image", img);

    // Initializa Output Picture
    IplImage* outputImg = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_32F,imgSpectrum);
    const int outputStep = outputImg->widthStep/sizeof(float);
    float *outputData = (float *)outputImg->imageData;

    //---------- gray values images ----------
    if (imgSpectrum == 1) {
      float *imgdata = new float[imageSize];
      float *outputImgData = new float[imageSize];

      // Copy <- Input Image   < 0.001 sec to do this loop ,i think.it is not necessary to write in GPU because GPU may be even slower
      for (i=0; i<imgHeight; i++) for (j=0;j<imgWidth;j++) imgdata[i*imgWidth+j] = (float)imageData[i*step+j];  

      if (mode == 0)
        cpu_convolutionGrayImage(imgdata, kernel, outputImgData, imgWidth, imgHeight, kRadiusX, kRadiusY);
      else
        gpu_convolutionGrayImage(imgdata, kernel, outputImgData, imgWidth, imgHeight, kRadiusX, kRadiusY, mode);

      // Copy -> Output Image
      for (i=0; i<imgHeight; i++) for (j=0;j<imgWidth;j++) outputData[i*outputStep+j] = outputImgData[i*imgWidth+j]/255;

      delete[] imgdata;
      delete[] outputImgData;
    } // endif gray image
    //---------- RGB color images ----------
    else if (imgSpectrum == 3) {

      if (interleavedMode) {
        int s;
        float *interleavedImg = new float[imageSize];
        float *interleavedOutput = new float[imageSize];

        // Copy <- Input Image (RGB) < 0.016 sec "ArchUtah.ppm"
        for (i=0;i<imgHeight;i++) 
          for (j=0;j<imgWidth;j++) 
            for (s=0;s<imgSpectrum;s++)
              interleavedImg[i*imgWidth*3+j*imgSpectrum+s] = (float)imageData[i*step+j*imgSpectrum+s];

        gpu_convolutionInterleavedRGB(interleavedImg, kernel, interleavedOutput, imgWidth, imgHeight, kRadiusX, kRadiusY, mode);

        // Copy -> Output Image (RGB) < 0.016 sec
        for (i=0;i<imgHeight;i++) 
          for (j=0;j<imgWidth;j++) 
            for (s=0;s<imgSpectrum;s++)
              outputData[i*outputStep+j*imgSpectrum+s] = interleavedOutput[i*imgWidth*3+j*imgSpectrum+s]/255;

        delete[] interleavedImg;
        delete[] interleavedOutput;
      } // endif interleaved mode
      else {
        int s;
        float *imgdata = new float[imageSize];
        float *outputImgData = new float[imageSize];

        for (i=0;i<imgHeight;i++) 
          for (j=0;j<imgWidth;j++) 
            for (s=0;s<imgSpectrum;s++)
              imgdata[s*imgHeight*imgWidth+i*imgWidth+j] = (float)imageData[i*step+j*imgSpectrum+s];

        if (mode == 0)
          cpu_convolutionRGB(imgdata, kernel, outputImgData, imgWidth, imgHeight, kRadiusX, kRadiusY);
        else
          gpu_convolutionRGB(imgdata, kernel, outputImgData, imgWidth, imgHeight, kRadiusX, kRadiusY, mode);

        for (i=0;i<imgHeight;i++) 
          for (j=0;j<imgWidth;j++) 
            for (s=0;s<imgSpectrum;s++)
              outputData[i*outputStep+j*imgSpectrum+s] = outputImgData[s*imgHeight*imgWidth+i*imgWidth+j]/255;

        delete[] imgdata;
        delete[] outputImgData;

      } // endif non-interleaved mode

    } // endif RGB image (spectrum == 3)
    else {
      std::cout << "Error: Unsupported image type!" << std::endl;
      return 1;
    }

    cvNamedWindow("Output Image", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Output Image", 500, 100);
    cvShowImage("Output Image", outputImg);

    // Normalize Gaussian Mask and show
    IplImage* kernelImg = cvCreateImage(cvSize(kRadiusX*2+1,kRadiusY*2+1),IPL_DEPTH_32F,1);
    const int kernelImgstep = kernelImg->widthStep/sizeof(float);
    float *kernelImgdata = (float *)kernelImg->imageData;
    kernelImgdata = cpu_normalizeGaussianKernel_cv(kernel, kernelImgdata, kRadiusX, kRadiusY, kernelImgstep);
    cvNamedWindow("Kernel Image", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Kernel Image", 150, 150);
    cvShowImage("Kernel Image", kernelImg);
    cvWaitKey(1);

    //---------- BENCHMARK ----------
    if (numBenchmarkCycles > 0 && imgSpectrum == 3){
      std::cout << std::endl << "Running Benchmark:" << std::endl;

      // the image and kernel here are no longer correct
      float *imgdata = new float[imageSize/3];
      float *outputImgData = new float[imageSize/3];
      for (i=0;i<imgHeight;i++) for (j=0;j<imgWidth;j++) imgdata[i*imgWidth+j] = (float)imageData[i*step+j];
      cpu_convolutionBenchmark(imgdata, kernel, outputImgData, imgWidth, imgHeight, kRadiusX, kRadiusY, numBenchmarkCycles);
      gpu_convolutionKernelBenchmarkGrayImage(imgdata, kernel, outputImgData, imgWidth, imgHeight, kRadiusX, kRadiusY, numBenchmarkCycles);
      delete[] imgdata;
      delete[] outputImgData;

      int s;
      float *interleavedImg = new float[imageSize];
      float *interleavedOutput = new float[imageSize];
      for (i=0;i<imgHeight;i++) for (j=0;j<imgWidth;j++) for (s=0;s<imgSpectrum;s++) interleavedImg[i*imgWidth*3+j*imgSpectrum+s] = (float)imageData[i*step+j*imgSpectrum+s];
      gpu_convolutionKernelBenchmarkInterleavedRGB(interleavedImg, kernel, interleavedOutput, imgWidth, imgHeight, kRadiusX, kRadiusY, numBenchmarkCycles);
      delete[] interleavedImg;
      delete[] interleavedOutput;
    }// end benchmark
    std::cout << std::endl << "Press any key on the image to exit..." << std::endl;
    cvWaitKey(0);
    cvReleaseImage(&img);
    cvReleaseImage(&kernelImg);
    cvReleaseImage(&outputImg);
  } // endif image file
  //----------------------------------------------------------------------------
  // Camera Mode
  //----------------------------------------------------------------------------
  else {
    CvCapture* capture;
    IplImage *img;
    capture = cvCaptureFromCAM(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, cameraWidth );
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, cameraHeight );
    cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Input", 50, 100);
    cvNamedWindow("Output", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Output", 750, 100);

    img = cvQueryFrame(capture);

    const int imgHeight = img->height;
    const int imgWidth = img->width;
    const int step = img->widthStep/sizeof(uchar);
    const int imgSpectrum = img->nChannels;
    uchar *imageData = (uchar *)img->imageData;

    const int imageSize = imgHeight*imgWidth*sizeof(float)*imgSpectrum;
    float *interleavedImg = new float[imageSize];
    float *interleavedOutput = new float[imageSize];

    IplImage* outputImg = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_32F,imgSpectrum);
    const int outputStep = outputImg->widthStep/sizeof(float);
    float *outputData = (float *)outputImg->imageData;

    int i,j,s;
    std::cout << std::endl << "Press Esc on the image to exit..." << std::endl;

    while (cvWaitKey(30) != 27)
    {
      // ~0 sec
      img = cvQueryFrame(capture);

      // < 0.016 sec
      for (i=0;i<imgHeight;i++)
        for (j=0;j<imgWidth;j++)
          for (s=0;s<imgSpectrum;s++)
            interleavedImg[i*imgWidth*3+j*imgSpectrum+s] = (float)imageData[i*step+j*imgSpectrum+s];

      // ~0.032 sec
      gpu_convolutionInterleavedRGB(interleavedImg, kernel, interleavedOutput, imgWidth, imgHeight, kRadiusX, kRadiusY, mode);

      // ~0.016 sec
      for (i=0;i<imgHeight;i++)
        for (j=0;j<imgWidth;j++)
          for (s=0;s<imgSpectrum;s++)
            outputData[i*outputStep+j*imgSpectrum+s] = interleavedOutput[i*imgWidth*3+j*imgSpectrum+s]/255;

      cvShowImage("Input", img);
      cvShowImage("Output", outputImg);
    }

    cvReleaseCapture(&capture);
    //cvReleaseImage(&img);
    cvReleaseImage(&outputImg);
    cvDestroyAllWindows();
  } //endif "camera"

  return 0;
}
