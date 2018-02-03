/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: sorflow
* file:    main.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <iostream>

#include "sorflow.h"
#include "flowio.h"
#include "highgui.h"
#include "cv.h"

int main(int argc, char *argv[])
{
  if (argc < 2) {
    fprintf(stderr,"Usage:""\n%s imageData1 imageData2", argv[0]);
    fprintf(stderr,"   or:""\n%s camera", argv[0]);
    fprintf(stderr, "\n\ngeneral usage further parameters: -option number\n");
    fprintf(stderr, "-l  -lambda \t\t defines lambda [default=5.0]\n");
    fprintf(stderr, "-oi -outeriterations \t defines outeriterations [default=40]\n");
    fprintf(stderr, "-ii -inneriterations \t defines inneriterations [default=2]\n");
    fprintf(stderr, "-sl -startlevel \t defines startlevel [default=3]\n");
    fprintf(stderr, "-el -endlevel \t\t defines endlevel [default=0]\n");
    fprintf(stderr, "-m  -method \t\t defines method [default=2]\n");
    fprintf(stderr, "-w  -width \t\t defines camera capture size [default=320]\n");
    fprintf(stderr, "\navailable methods:\n");
    fprintf(stderr, "0 - linear (phi function is identity = quatratic penalization)\n");
    fprintf(stderr, "1 - non-linear (phi function is square root = robust penalization)\n");
    fprintf(stderr, "2 - non-linear with warping (phi function is square root = robust penalization)\n");

    fprintf(stderr,"\n\n");
    return 0;
  }

  int start_level = 3;
  int end_level = 0;
  int outer_iterations = 40;
  int inner_iterations = 2;
  float lambda = 5.0f;
  int cameraWidth = 320, cameraHeight = 240;
  int method = 2;

  std::string partype = "";
  std::string parval = "";

  int parnum;
  if (strcmp(argv[1], "camera") != 0) 
    parnum = 3;
  else
    parnum = 2;
  while(parnum < argc) {
    partype = argv[parnum++];
    if(partype == "-l" || partype == "-lambda") {
      lambda = (float)atof(argv[parnum++]);
      fprintf(stderr,"\nSmoothness Weight: %f", lambda);
    }
    else if(partype == "-oi" || partype == "-outeriterations") {
      outer_iterations = atoi(argv[parnum++]);
      fprintf(stderr,"\nOuter Iterations: %i", outer_iterations);
    }
    else if(partype == "-ii" || partype == "-inneriterations") {
      inner_iterations = atoi(argv[parnum++]);
      fprintf(stderr,"\nInner Iterations: %i", inner_iterations);
    }
    else if(partype == "-sl" || partype == "-startlevel") {
      start_level = atoi(argv[parnum++]);
      fprintf(stderr,"\nStart Level: %i", start_level);
    }
    else if(partype == "-el" || partype == "-endlevel") {
      end_level = atoi(argv[parnum++]);
      fprintf(stderr,"\nEnd Level: %i", end_level);
    }
    else if(partype == "-m" || partype == "-method") {
      method = atoi(argv[parnum++]);
      fprintf(stderr,"\nComputation Method: %i", method);
    }
    else if ((partype == "-w" || partype == "-width") && strcmp(argv[1], "camera") == 0) {
      cameraWidth = atoi(argv[parnum++]);
      cameraHeight = (int)(cameraWidth*3/4);
      fprintf(stderr,"\nCamera Width: %i\nCamera Height: %i", cameraWidth, cameraHeight);
    }
  }

  //----------------------------------------------------------------------------
  // Image File Mode
  //----------------------------------------------------------------------------
  if (strcmp(argv[1], "camera") != 0) {

    int i,j,s;

    IplImage* img1 = cvLoadImage(argv[1],0); // 0 = force the image to be grayscale
    IplImage* img2 = cvLoadImage(argv[2],0);
    if (!img1 || !img2) {
      std::cout << "Error: Could not open file" << std::endl;
      return 1;
    }
    const int imgHeight = img1->height;
    const int imgWidth = img1->width;
    if (imgHeight != img2->height || imgWidth != img2->width) {
      std::cout << "Error: Image formats are not equal!" << std::endl;
      return 1;
    }

    const int step1 = img1->widthStep/sizeof(uchar);
    const int step2 = img1->widthStep/sizeof(uchar);
    uchar *cvImageData1 = (uchar *)img1->imageData;
    uchar *cvImageData2 = (uchar *)img2->imageData;

    float *imageData1 = new float[imgWidth*imgHeight];
    float *imageData2 = new float[imgWidth*imgHeight];
    float *colorImage = new float[imgWidth*imgHeight*3];

    for (i=0;i<imgHeight;i++) 
      for (j=0;j<imgWidth;j++) {
        imageData1[i*imgWidth+j] = (float)cvImageData1[i*step1+j];
        imageData2[i*imgWidth+j] = (float)cvImageData2[i*step2+j];
      }

      SORFlow *sorflow = new SORFlow(imgWidth, imgHeight);

      fprintf(stderr,"\nSetting up Flow");
      sorflow->set_input(imageData1);
      sorflow->set_input(imageData2);
      sorflow->set_outer_iterations(outer_iterations);
      sorflow->set_inner_iterations(inner_iterations);
      sorflow->set_lambda(lambda);
      sorflow->set_levels(start_level,end_level);
      sorflow->set_method(method);

      fprintf(stderr,"\nComputing Flow");
      sorflow->compute_flow();
      fprintf(stderr,"\nGetting Output");
      sorflow->get_output_RGB(colorImage);
      fprintf(stderr,"\nSaving Flow");
      float *u = new float[imgWidth*imgHeight*2];
      sorflow->get_output(u);
      save_flow_file("output.flo",u, imgWidth, imgHeight);
      delete [] u;

      fprintf(stderr,"\nDisplaying Output");

      IplImage* outputImg = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_8U,3);
      const int outputStep = outputImg->widthStep/sizeof(uchar);
      uchar *outputData = (uchar *)outputImg->imageData;

      for (i=0;i<imgHeight;i++) 
        for (j=0;j<imgWidth;j++) 
          for (s=0;s<3;s++)
            outputData[i*outputStep+j*3+s] = (uchar)(colorImage[(i*imgWidth+j)*3+s]);


      IplImage* outputImgRGB = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_8U,3);
      cvCvtColor(outputImg, outputImgRGB, CV_BGR2RGB);

      cvSaveImage("flow_image.ppm", outputImgRGB);

      cvNamedWindow("Input Image 1", CV_WINDOW_AUTOSIZE);
      cvMoveWindow("Input Image 1", 100, 100);
      cvShowImage("Input Image 1", img1);
      cvNamedWindow("Input Image 2", CV_WINDOW_AUTOSIZE);
      cvMoveWindow("Input Image 2", 450, 100);
      cvShowImage("Input Image 2", img2);
      cvNamedWindow("Output Image", CV_WINDOW_AUTOSIZE);
      cvMoveWindow("Output Image", 100, 400);
      cvShowImage("Output Image", outputImgRGB);

      printf("\nPress Esc on the image to exit...\n");
      cvWaitKey(0);

      delete sorflow;
      delete [] imageData2;
      delete [] imageData1;
      cvReleaseImage(&img1);
      cvReleaseImage(&img2);
      cvReleaseImage(&outputImg);
      cvReleaseImage(&outputImgRGB);
  } // endif image file
  //----------------------------------------------------------------------------
  // Camera Mode
  //----------------------------------------------------------------------------
  else{
    int i,j,s;
    CvCapture* capture;
    IplImage *img;
    capture = cvCaptureFromCAM(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, cameraWidth );
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, cameraHeight );

    img = cvQueryFrame(capture);

    if (!img) {
      std::cerr << "No Camera detected!" << std::endl;

      cvReleaseCapture(&capture);

      return 1;
    }

    const int imgHeight   = img->height;
    const int imgWidth    = img->width;
    const int step        = img->widthStep/sizeof(uchar);
    const int imgSpectrum = img->nChannels;
    uchar *cvImageData    = (uchar *)img->imageData;

    int imageSize = imgHeight*imgWidth*sizeof(float)*imgSpectrum;
    float *img1 = new float[imgHeight*imgWidth*sizeof(float)];
    float *colorImage = new float[imageSize];
    for (i=0;i<imgHeight;i++) 
      for (j=0;j<imgWidth;j++) {
        for (s=0;s<imgSpectrum;s++)
          img1[i*imgWidth+j] += (float)cvImageData[i*step+j*imgSpectrum+s];
        img1[i*imgWidth+j] /= imgSpectrum;
      }

      IplImage* outputImg = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_8U,3);
      const int outputStep = outputImg->widthStep/sizeof(uchar);
      uchar *outputData = (uchar *)outputImg->imageData;

      IplImage* outputImgRGB = cvCreateImage(cvSize(imgWidth,imgHeight),IPL_DEPTH_8U,3);

      cvNamedWindow("Input Image", CV_WINDOW_AUTOSIZE);
      cvMoveWindow("Input Image", 50, 100);
      cvNamedWindow("Output Image", CV_WINDOW_AUTOSIZE);
      cvMoveWindow("Output Image", 450, 100);

      SORFlow *sorflow = new SORFlow(cameraWidth, cameraHeight);

      fprintf(stderr,"\nSetting up Flow");
      sorflow->set_input(img1);
      sorflow->set_outer_iterations(outer_iterations);
      sorflow->set_inner_iterations(inner_iterations);
      sorflow->set_lambda(lambda);
      sorflow->set_levels(start_level,end_level);
      sorflow->set_method(method);

      printf("\nPress Esc on the image to exit...\n");


      while (cvWaitKey(30) != 27){
        img = cvQueryFrame(capture);
        for (i=0;i<imgHeight;i++) 
          for (j=0;j<imgWidth;j++) {
            for (s=0;s<imgSpectrum;s++)
              img1[i*imgWidth+j] += (float)cvImageData[i*step+j*imgSpectrum+s];
            img1[i*imgWidth+j] /= imgSpectrum;
          }

          sorflow->set_input(img1);
          sorflow->compute_flow();
          sorflow->get_output_RGB(colorImage);

          for (i=0;i<imgHeight;i++) 
            for (j=0;j<imgWidth;j++) 
              for (s=0;s<3;s++)
                outputData[i*outputStep+j*3+s] = (uchar)(colorImage[(i*imgWidth+j)*3+s]);

          cvCvtColor(outputImg, outputImgRGB, CV_BGR2RGB);

          cvShowImage("Input Image", img);
          cvShowImage("Output Image", outputImgRGB);
      }
      delete sorflow;
      delete[] img1;
      delete[] colorImage;
      cvReleaseCapture(&capture);
      //cvReleaseImage(&img);
      cvReleaseImage(&outputImg);
      cvReleaseImage(&outputImgRGB);
      cvDestroyAllWindows();

  } // endif camera

  fprintf(stderr,"\n\n");
  return 0;
}
