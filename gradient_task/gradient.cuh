/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: gradient
* file:    gradient.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef GRADIENT_CUH
#define GRADIENT_CUH




const char* getStudentName();
int         getStudentID();
bool        checkStudentNameAndID();



void gpu_derivative_sm_d(const float *inputImage, float *outputImage,
                          int iWidth, int iHeight, int iSpectrum, int mode=0);



#endif // #ifndef GRADIENT_CUH
