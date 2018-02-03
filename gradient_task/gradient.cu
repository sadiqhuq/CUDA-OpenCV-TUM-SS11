/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: gradient
* file:    gradient.cu
*
* 
\********* PLEASE ENTER YOUR CORRECT STUDENT NAME AND ID BELOW **************/
const char* studentName = "Sadiq Huq";
const int   studentID   = 3273623;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* derivativeY_sm_d(const float *inputImage, ... )
* derivativeY_sm_d(const float3 *inputImage, ... )
* gradient_magnitude_d(const float *inputImage, ... )
* gradient_magnitude_d(const float3 *inputImage, ... )
*
\****************************************************************************/


#include <cutil.h>
#include <cutil_inline.h>
#include "gradient.cuh"



#define BW 16
#define BH 16



const char* getStudentName() { return studentName; };
int         getStudentID()   { return studentID; };
bool        checkStudentNameAndID() { return strcmp(studentName, "John Doe") != 0 && studentID != 1234567; }; 




__global__ void derivativeX_sm_d(const float *inputImage, float *outputImage,
                                 int iWidth, int iHeight, size_t iPitchBytes)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float u[BW+2][BH];


  if (x < iWidth && y < iHeight) {
    u[threadIdx.x+1][threadIdx.y] = *((float*)((char*)inputImage + y*iPitchBytes)+x);

    if (x == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
    else if (x == (iWidth-1)) u[threadIdx.x+2][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
    else {
      if (threadIdx.x == 0) u[threadIdx.x][threadIdx.y] = *((float*)((char*)inputImage + y*iPitchBytes)+x-1);
      else if (threadIdx.x == blockDim.x-1) u[threadIdx.x+2][threadIdx.y] = *((float*)((char*)inputImage + y*iPitchBytes)+x+1);
    }
  }

  __syncthreads();

  if (x < iWidth && y < iHeight)
    *((float*)(((char*)outputImage) + y*iPitchBytes)+ x) = 0.5f*(u[threadIdx.x+2][threadIdx.y]-u[threadIdx.x][threadIdx.y])+128;
}




__global__ void derivativeX_sm_d(const float3 *inputImage, float3 *outputImage,
                                 int iWidth, int iHeight, size_t iPitchBytes)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  float3 imgValue ;
  __shared__ float3 u[BW+2][BH];

  if (x < iWidth && y < iHeight) {
    u[threadIdx.x+1][threadIdx.y] = *((float3*)((char*)inputImage + y*iPitchBytes)+x);

    if (x == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
    else if (x == (iWidth-1)) u[threadIdx.x+2][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
    else {
      if (threadIdx.x == 0) u[threadIdx.x][threadIdx.y] = *((float3*)((char*)inputImage + y*iPitchBytes)+x-1);
      else if (threadIdx.x == blockDim.x-1) u[threadIdx.x+2][threadIdx.y] = *((float3*)((char*)inputImage + y*iPitchBytes)+x+1);
    }
  }

  __syncthreads();

  // +128 to stay within range 255
  if (x < iWidth && y < iHeight) {
    imgValue.x = 0.5f*(u[threadIdx.x+2][threadIdx.y].x - u[threadIdx.x][threadIdx.y].x)+128;
    imgValue.y = 0.5f*(u[threadIdx.x+2][threadIdx.y].y - u[threadIdx.x][threadIdx.y].y)+128;
    imgValue.z = 0.5f*(u[threadIdx.x+2][threadIdx.y].z - u[threadIdx.x][threadIdx.y].z)+128;
    
    *((float3*)(((char*)outputImage) + y*iPitchBytes)+ x) = imgValue;
  }
  
}



__global__ void derivativeY_sm_d(const float *inputImage, float *outputImage,
                                 int iWidth, int iHeight, size_t iPitchBytes)
{

  // ### implement me ### 
	 const int x = blockIdx.x * blockDim.x + threadIdx.x;
	  const int y = blockIdx.y * blockDim.y + threadIdx.y;

	  __shared__ float u[BW][BH+2];


	  if (x < iWidth && y < iHeight) {
	    u[threadIdx.x][threadIdx.y+1] = *((float*)((char*)inputImage + y*iPitchBytes)+x);

	    if (y == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x][threadIdx.y+1];
	    else if (y == (iHeight-1)) u[threadIdx.x][threadIdx.y+2] = u[threadIdx.x][threadIdx.y+1];
	    else {
	      if (threadIdx.y == 0) u[threadIdx.x][threadIdx.y] = *((float*)((char*)inputImage + (y-1)*iPitchBytes)+x);
	      else if (threadIdx.y == blockDim.y-1) u[threadIdx.x][threadIdx.y+2] = *((float*)((char*)inputImage + (y+1)*iPitchBytes)+x);
	    }
	  }

	  __syncthreads();

	  if (x < iWidth && y < iHeight)
	    *((float*)(((char*)outputImage) + y*iPitchBytes)+ x) = 0.5f*(u[threadIdx.x][threadIdx.y+2]-u[threadIdx.x][threadIdx.y])+128;

}



__global__ void derivativeY_sm_d(const float3 *inputImage, float3 *outputImage,
                                 int iWidth, int iHeight, size_t iPitchBytes)
{

 //  ### implement me ### 
	  const int x = blockIdx.x * blockDim.x + threadIdx.x;
	  const int y = blockIdx.y * blockDim.y + threadIdx.y;
	  float3 imgValue ;
	  __shared__ float3 u[BW][BH+2];

	  if (x < iWidth && y < iHeight) {
	    u[threadIdx.x][threadIdx.y+1] = *((float3*)((char*)inputImage + y*iPitchBytes)+x);

	    if (y == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x][threadIdx.y+1];
	    else if (y == (iWidth-1)) u[threadIdx.x][threadIdx.y+2] = u[threadIdx.x][threadIdx.y+1];
	    else {
	      if (threadIdx.y == 0) u[threadIdx.x][threadIdx.y] = *((float3*)((char*)inputImage + (y-1)*iPitchBytes)+x);
	      else if (threadIdx.y == blockDim.y-1) u[threadIdx.x][threadIdx.y+2] = *((float3*)((char*)inputImage + (y+1)*iPitchBytes)+x);
	    }
	  }

	  __syncthreads();

	  // +128 to stay within range 255
	  if (x < iWidth && y < iHeight) {
	    imgValue.x = 0.5f*(u[threadIdx.x][threadIdx.y+2].x - u[threadIdx.x][threadIdx.y].x)+128;
	    imgValue.y = 0.5f*(u[threadIdx.x][threadIdx.y+2].y - u[threadIdx.x][threadIdx.y].y)+128;
	    imgValue.z = 0.5f*(u[threadIdx.x][threadIdx.y+2].z - u[threadIdx.x][threadIdx.y].z)+128;
//	    float3 value = make_float3(0.0f, 0.0f, 0.0f);
//	    *((float3*)(((char*)outputImage) + y*iPitchBytes)+ x) = value;
	    *((float3*)(((char*)outputImage) + y*iPitchBytes)+ x) = imgValue;
	  }
	  
}





__global__ void gradient_magnitude_d(const float *inputImage, float *outputImage,
                                     int iWidth, int iHeight, size_t iPitchBytes)
{

  // ### implement me ### 
	 const int x = blockIdx.x * blockDim.x + threadIdx.x;
	  const int y = blockIdx.y * blockDim.y + threadIdx.y;
	  
	  __shared__ float u[BW + 2][BH+2]; 

	  if (x < iWidth && y < iHeight) 
	  {
	    u[threadIdx.x + 1][threadIdx.y] = *((float*)((char*)inputImage + y*iPitchBytes) + x);
	    u[threadIdx.x][threadIdx.y+1] = *((float*)((char*)inputImage + y*iPitchBytes) + x);

	    // BC for X
	    if (x == 0) // clamp left border 
	    	u[threadIdx.x][threadIdx.y+1] = u[threadIdx.x+1][threadIdx.y];
	    else if (x == (iWidth-1)) // clamp right
	    	u[threadIdx.x+2][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
	    else // interier pixels
	    {
	      if (threadIdx.x == 0) 
	      	u[threadIdx.x][threadIdx.y] = *((float*)((char*)inputImage + y*iPitchBytes)+x-1);
	      else if (threadIdx.x == blockDim.x-1) 
	      	u[threadIdx.x+2][threadIdx.y] = *((float*)((char*)inputImage + y*iPitchBytes)+x+1);
	    }
	    
	    // BC for Y
	    if (y == 0) // clamp left border 
	    	u[threadIdx.x + 1][threadIdx.y] = u[threadIdx.x][threadIdx.y+1];
	    else if (y == (iHeight-1)) // clamp right
	    	u[threadIdx.x + 1][threadIdx.y+2] = u[threadIdx.x + 1][threadIdx.y+1];
	    else // interior 
	    {
	      if (threadIdx.y == 0) 
	      	u[threadIdx.x ][threadIdx.y] = *((float*)((char*)inputImage + (y-1)*iPitchBytes)+x);
	      else if (threadIdx.y == blockDim.y-1) 
	      	u[threadIdx.x ][threadIdx.y+2] = *((float*)((char*)inputImage + (y+1)*iPitchBytes)+x);
	    }        
	  }  
	 
	  __syncthreads();  

	  if (x < iWidth && y < iHeight)
	  {
		  
		  float dx =  0.5f*(u[threadIdx.x+2][threadIdx.y]-u[threadIdx.x][threadIdx.y])+128;
		  float dy = 0.5f*(u[threadIdx.x][threadIdx.y+2]-u[threadIdx.x][threadIdx.y])+128;
	   
	    *((float*)(((char*)outputImage) + y * iPitchBytes) + x) =  	sqrt(dx * dx + dy * dy);
	  }
}





__global__ void gradient_magnitude_d(const float3 *inputImage, float3 *outputImage,
                                     int iWidth, int iHeight, size_t iPitchBytes)
{

  // ### implement me ### 

	  const int x = blockIdx.x * blockDim.x + threadIdx.x;
	  const int y = blockIdx.y * blockDim.y + threadIdx.y;
	  float3 imgValue ;
	  __shared__ float3 u[BW+2][BH+2];
	
	
	  if (x < iWidth && y < iHeight) {
		  
	    u[threadIdx.x+1][threadIdx.y] = *((float3*)((char*)inputImage + y*iPitchBytes)+x);
	    u[threadIdx.x][threadIdx.y+1] = *((float3*)((char*)inputImage + y*iPitchBytes)+x);

	    // BC for X
	    if (x == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
	    else if (x == (iWidth-1)) u[threadIdx.x+2][threadIdx.y] = u[threadIdx.x+1][threadIdx.y];
	    else {
	      if (threadIdx.x == 0) u[threadIdx.x][threadIdx.y] = *((float3*)((char*)inputImage + y*iPitchBytes)+x-1);
	      else if (threadIdx.x == blockDim.x-1) u[threadIdx.x+2][threadIdx.y] = *((float3*)((char*)inputImage + y*iPitchBytes)+x+1);
	    }
	    
	   // BC for Y
		  if (y == 0) u[threadIdx.x][threadIdx.y] = u[threadIdx.x][threadIdx.y+1];
		 	    else if (y == (iHeight-1)) u[threadIdx.x][threadIdx.y+2] = u[threadIdx.x][threadIdx.y+1];
		 	    else {
		 	      if (threadIdx.y == 0) u[threadIdx.x][threadIdx.y] = *((float3*)((char*)inputImage + (y-1)*iPitchBytes)+x);
		 	      else if (threadIdx.y == blockDim.y-1) u[threadIdx.x][threadIdx.y+2] = *((float3*)((char*)inputImage + (y+1)*iPitchBytes)+x);
		 	    }
	  }

	  __syncthreads();

	
	  if (x < iWidth && y < iHeight) 
	  {
		  float3 dx3 = make_float3(
				  0.5f*(u[threadIdx.x+2][threadIdx.y].x - u[threadIdx.x][threadIdx.y].x)+128,
				  0.5f*(u[threadIdx.x+2][threadIdx.y].y - u[threadIdx.x][threadIdx.y].y)+128,
				  0.5f*(u[threadIdx.x+2][threadIdx.y].z - u[threadIdx.x][threadIdx.y].z)+128);

		  float3 dy3 = make_float3(
				  0.5f*(u[threadIdx.x][threadIdx.y+2].x - u[threadIdx.x][threadIdx.y].x)+128,
				  0.5f*(u[threadIdx.x][threadIdx.y+2].y - u[threadIdx.x][threadIdx.y].y)+128,
				  0.5f*(u[threadIdx.x][threadIdx.y+2].z - u[threadIdx.x][threadIdx.y].z)+128 );

		  float3 value = make_float3( 
				  sqrt(dx3.x * dx3.x + dy3.x * dy3.x ),
				  sqrt(dx3.y * dx3.z + dy3.y * dy3.y ),
				  sqrt(dx3.z * dx3.z + dy3.z * dy3.z ));
		  
		  *((float3*)(((char*)outputImage) + y*iPitchBytes)+ x) = value;
  
		  //	 	    *((float3*)(((char*)outputImage) + y*iPitchBytes)+ x) = sqrt(dx3*dx3 + dy3*dy3);
	  }
	
	
	
	
	
}



void gpu_derivative_sm_d(const float *inputImage, float *outputImage,
                         int iWidth, int iHeight, int iSpectrum, int mode)
{
  size_t iPitchBytes;
  float *inputImage_d = 0, *outputImage_d = 0;

  dim3 blockSize(BW, BH);  
  dim3 gridSize( (int)ceil(iWidth/(float)BW), (int)ceil(iHeight/(float)BH) );
  //dim3 smSize(BW+2,BH);

  if(iSpectrum == 1) {
    cutilSafeCall( cudaMallocPitch( (void**)&(inputImage_d), &iPitchBytes, iWidth*sizeof(float), iHeight ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(outputImage_d), &iPitchBytes, iWidth*sizeof(float), iHeight ) );

    cutilSafeCall( cudaMemcpy2D(inputImage_d, iPitchBytes, inputImage, iWidth*sizeof(float), iWidth*sizeof(float), iHeight, cudaMemcpyHostToDevice) );

    if (mode == 0)
      derivativeX_sm_d<<<gridSize, blockSize>>>(inputImage_d, outputImage_d, iWidth, iHeight, iPitchBytes);
    else if (mode == 1)
      derivativeY_sm_d<<<gridSize, blockSize>>>(inputImage_d, outputImage_d, iWidth, iHeight, iPitchBytes);
    else if (mode == 2)
      gradient_magnitude_d<<<gridSize, blockSize>>>(inputImage_d, outputImage_d, iWidth, iHeight, iPitchBytes);

    cutilSafeCall( cudaThreadSynchronize() );
    cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float), outputImage_d, iPitchBytes, iWidth*sizeof(float), iHeight, cudaMemcpyDeviceToHost) );
  }
  else if(iSpectrum == 3) {
    cutilSafeCall( cudaMallocPitch( (void**)&(inputImage_d), &iPitchBytes, iWidth*sizeof(float3), iHeight ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(outputImage_d), &iPitchBytes, iWidth*sizeof(float3), iHeight ) );

    cutilSafeCall( cudaMemcpy2D(inputImage_d, iPitchBytes, inputImage, iWidth*sizeof(float3), iWidth*sizeof(float3), iHeight, cudaMemcpyHostToDevice) );

    if (mode == 0)
      derivativeX_sm_d<<<gridSize, blockSize>>>((float3*)inputImage_d, (float3*)outputImage_d, iWidth, iHeight, iPitchBytes);
    else if (mode == 1)
      derivativeY_sm_d<<<gridSize, blockSize>>>((float3*)inputImage_d, (float3*)outputImage_d, iWidth, iHeight, iPitchBytes);
    else if (mode == 2)
      gradient_magnitude_d<<<gridSize, blockSize>>>((float3*)inputImage_d, (float3*)outputImage_d, iWidth, iHeight, iPitchBytes);

    cutilSafeCall( cudaThreadSynchronize() );
    cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float3), outputImage_d, iPitchBytes, iWidth*sizeof(float3), iHeight, cudaMemcpyDeviceToHost) );
  }

  cutilSafeCall( cudaFree(inputImage_d) );
  cutilSafeCall( cudaFree(outputImage_d) );
}
