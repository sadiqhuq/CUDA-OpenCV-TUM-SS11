/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: convolution
* file:    convolution_gpu.cu
*
* 
\********* PLEASE ENTER YOUR CORRECT STUDENT NAME AND ID BELOW **************/
const char* gpu_studentName = "Sadiq Huq";
const int   gpu_studentID   = 3273623;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* gpu_convolutionGrayImage_gm_d
* gpu_convolutionGrayImage_gm_cm_d
* gpu_convolutionGrayImage_sm_d
* gpu_convolutionGrayImage_sm_cm_d
* gpu_convolutionGrayImage_dsm_cm_d 
* gpu_convolutionInterleavedRGB_dsm_cm_d - Mode 5 interleaved 
* gpu_convolutionInterleavedRGB_tex_cm_d -  Model 6 interleaved
*
\****************************************************************************/


#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <iostream>


#include "convolution_gpu.cuh"


#define BW 16                 // block width
#define BH 16                 // block height
#define MAXKERNELSIZE 3000    // actual size, i.e. NOT the byte size





// constant memory block on device
__constant__ float constKernel[MAXKERNELSIZE];


// texture memory and descriptor
cudaChannelFormatDesc tex_Image_desc = cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> tex_Image;

cudaChannelFormatDesc tex_Image_descF4 = cudaCreateChannelDesc<float4>();
texture<float4, 2, cudaReadModeElementType> tex_ImageF4;


__host__ const char* gpu_getStudentName() { return gpu_studentName; };
__host__ int         gpu_getStudentID()   { return gpu_studentID; };
__host__ bool        gpu_checkStudentNameAndID() { return strcmp(gpu_studentName, "John Doe") != 0 && gpu_studentID != 1234567; };




//----------------------------------------------------------------------------
// Gray Image Functions
//----------------------------------------------------------------------------


// mode 1: using global memory only
__global__ void gpu_convolutionGrayImage_gm_d(const float *inputImage, const float *kernel, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch, size_t kPitch)
{

  // ### implement me ### 
	int temp_x,temp_y;
	float sum;
	const int kWidth  = (kRadiusX<<1) + 1;
	const int kHeight = (kRadiusY<<1) + 1;
	
	const uint thposX = threadIdx.x + blockIdx.x * blockDim.x;
	const uint thposY = threadIdx.y + blockIdx.y * blockDim.y;

//	
	if (thposX >=iWidth || thposY >=iHeight)
		return;
	sum=0;
	for (int y=0;y<kHeight;y++)
	{
		for (int x=0;x<kWidth;x++)
		{
			temp_x= thposX + x - kRadiusX;
			temp_y= thposY + y - kRadiusY;
			
			if(temp_x < 0)
				temp_x=0;
			if(temp_x >= iWidth)
				temp_x=iWidth-1;
			if(temp_y < 0)
				temp_y=0;
			if(temp_y >= iHeight)
				temp_y=iHeight-1;

			sum = sum+kernel[y*kPitch+x]*inputImage[temp_x+temp_y*iPitch];
		}
	}
	outputImage[thposY*iPitch+thposX] = sum;	

}

// mode 2: using global memory and constant memory
__global__ void gpu_convolutionGrayImage_gm_cm_d(const float *inputImage, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch)
{

	int temp_x,temp_y;
	float sum;

	const int kWidth  = (kRadiusX<<1) + 1;
	const int kHeight = (kRadiusY<<1) + 1;

	const uint thposX = threadIdx.x + blockIdx.x * blockDim.x;
	const uint thposY = threadIdx.y + blockIdx.y * blockDim.y;


	if (thposX >=iWidth || thposY >=iHeight)
		return;
	sum=0;
	for (int y=0;y<kHeight;y++)
	{
		for (int x=0;x<kWidth;x++)
		{
			temp_x = thposX + x - kRadiusX;
			temp_y = thposY + y - kRadiusY;
			
			if(temp_x < 0)
				temp_x=0;
			if(temp_x >= iWidth)
				temp_x=iWidth-1;
			if(temp_y < 0)
				temp_y=0;
			if(temp_y >= iHeight)
				temp_y=iHeight-1;

			sum = sum+constKernel[y*kWidth+x]*inputImage[temp_x+temp_y*iPitch];
		}
	}
	outputImage[thposY*iPitch+thposX] = sum;		
}


// mode 3: using shared memory for image and global memory for kernel access
__global__ void gpu_convolutionGrayImage_sm_d(const float *inputImage, const float *kernel, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch, size_t kPitch)
{
  // make use of the constant MAXKERNELSIZE in order to define the shared memory size
  
	__shared__ float sdata[MAXKERNELSIZE];

	  const uint thposX = threadIdx.x + blockIdx.x * blockDim.x;
	  const uint thposY = threadIdx.y + blockIdx.y * blockDim.y;
	  
	  outputImage[thposX + thposY * iPitch] = 0;

	  const uint width = (kRadiusX << 1) + BW; 	
	  const uint height = (kRadiusY << 1) + BH;

	  const uint blockPixels = BW * BH;   		
	  const uint totalPixels = width * height;	

	  const uint startX = blockIdx.x * blockDim.x - kRadiusX;
	  const uint startY = blockIdx.y * blockDim.y - kRadiusY;

	  const uint threadPos = threadIdx.x + threadIdx.y * blockDim.x;

	  uint offset = 0;

	  while(offset < totalPixels)
	  {		    
		  int temp_x,temp_y;
		  temp_x=startX + ((threadPos + offset) % width);
		  temp_y=startY + ((threadPos + offset) / width);
		  if(temp_x < 0)
			  temp_x=0;
		  if(temp_x >= iWidth)
			  temp_x=iWidth-1;
		  if(temp_y < 0)
			  temp_y=0;
		  if(temp_y >= iHeight)
			  temp_y=iHeight-1;

		  if(offset + threadPos < totalPixels)  
			  sdata[offset + threadPos] = inputImage[temp_x + temp_y * iPitch];
		  offset += blockPixels; // stride
	  }
	  
	  __syncthreads();
	  
//	  outputImage[thposX + thposY * iPitch]=sdata[offset + blockPos];
	  
	    const int kWidth  = (kRadiusX<<1) + 1;
	  	const int kHeight = (kRadiusY<<1) + 1;
	  	
	  if(thposX < iWidth && thposY < iHeight)
	  {	    
	    float sum = 0.0f;
	          
	    const int blockWidth = blockDim.x + (kRadiusX<<1);
	    const int blockHeight = blockDim.y + (kRadiusY<<1);

	    for (int y = 0; y < kHeight; ++y)
	    	for (int x = 0; x < kWidth; ++x) {
	    		const int sharedY = threadIdx.y + y;
	    		const int sharedX = threadIdx.x + x;
	    		sum += sdata[sharedY * blockWidth + sharedX] * kernel[y * kPitch + x];
	    	}
	    outputImage[thposX + thposY * iPitch] = sum; 
    
	  }
}


// mode 4: using shared memory for image and constant memory for kernel access
__global__ void gpu_convolutionGrayImage_sm_cm_d(const float *inputImage, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch)
{
  // make use of the constant MAXKERNELSIZE in order to define the shared memory size
	  
			__shared__ float sdata[MAXKERNELSIZE];

		  // Global position of thread in image coordinates 
		  const uint thposX = threadIdx.x + blockIdx.x * blockDim.x;
		  const uint thposY = threadIdx.y + blockIdx.y * blockDim.y;
		  
		  outputImage[thposX + thposY * iPitch] = 0;  // Debug black

		  // apron => extra boundary space. width = block + apron region
		  const uint width = (kRadiusX << 1) + BW; 	
		  const uint height = (kRadiusY << 1) + BH;

		  // 16x16 number of pixels in block   BW = blockDim.x 
		  const uint blockPixels = BW * BH;   		
		  // number of pixels in (256 + apron pixels)
		  const uint totalPixels = width * height;	

		  // (block+apron)-region in image pixel coordinates  
		  // first pixel of each block, same for all threads in a particular block
		  const uint startX = blockIdx.x * blockDim.x - kRadiusX;
		  const uint startY = blockIdx.y * blockDim.y - kRadiusY;
		      
		  // y*rowsize + x => linear pos of the pixel thread in block
		  const uint threadPosInBlock = threadIdx.x + threadIdx.y * blockDim.x;
		 
		  uint offset = 0;

		  while(offset < totalPixels)
		  {		    
			  int temp_x,temp_y;
				temp_x=startX + ((threadPosInBlock + offset) % width);
				temp_y=startY + ((threadPosInBlock + offset) / width);
				if(temp_x < 0)
					temp_x=0;
				if(temp_x >= iWidth)
					temp_x=iWidth-1;
				if(temp_y < 0)
					temp_y=0;
				if(temp_y >= iHeight)
					temp_y=iHeight-1;
					
			    if(offset + threadPosInBlock < totalPixels)  
					      sdata[offset + threadPosInBlock] = inputImage[temp_x + temp_y * iPitch];
		    offset += blockPixels; 
		    
		  }
		  
		  __syncthreads();
		  
		    // convolution -  image on shared memory 
		  
		    const int kWidth  = (kRadiusX<<1) + 1;
		  	const int kHeight = (kRadiusY<<1) + 1;
		  	
		  if(thposX < iWidth && thposY < iHeight)
		  {
		    float sum = 0.0f;
		    for (int y = 0; y < kHeight; ++y)
		    	for (int x = 0; x < kWidth; ++x) {
		    		const int sharedY = threadIdx.y + y;
		    		const int sharedX = threadIdx.x + x;
		    		sum += sdata[sharedY * width + sharedX] * constKernel[y * kWidth + x];
		    	}

		    outputImage[thposX + thposY * iPitch] = sum; 	
		  }

} 



// mode 5: using dynamically allocated shared memory for image and constant memory for kernel access
__global__ void gpu_convolutionGrayImage_dsm_cm_d(const float *inputImage, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch)
{

  // ### implement me ### 
	extern __shared__ float sdata[];

	  // Global position of thread in image coordinates 
	  const uint thposX = threadIdx.x + blockIdx.x * blockDim.x;
	  const uint thposY = threadIdx.y + blockIdx.y * blockDim.y;
	  
//	  outputImage[thposX + thposY * iPitch] = 0;  // Debug black

	  // apron => extra boundary space. width = block + apron region
	  const uint width = (kRadiusX << 1) + BW; 	
	  const uint height = (kRadiusY << 1) + BH;

	  // 16x16 number of pixels in block   BW = blockDim.x 
	  const uint blockPixels = BW * BH;   		
	  // number of pixels in (256 + apron pixels)
	  const uint totalPixels = width * height;	

	  // (block+apron)-region in image pixel coordinates  
	  // first pixel of each block, same for all threads in a particular block
	  const uint startX = blockIdx.x * blockDim.x - kRadiusX;
	  const uint startY = blockIdx.y * blockDim.y - kRadiusY;
	      
	  // y*rowsize + x => linear pos of the pixel thread in block
	  const uint threadPosInBlock = threadIdx.x + threadIdx.y * blockDim.x;
	 
	  uint offset = 0;

	  while(offset < totalPixels)
	  {		    
		  int temp_x,temp_y;
			temp_x=startX + ((threadPosInBlock + offset) % width);
			temp_y=startY + ((threadPosInBlock + offset) / width);
			if(temp_x < 0)
				temp_x=0;
			if(temp_x >= iWidth)
				temp_x=iWidth-1;
			if(temp_y < 0)
				temp_y=0;
			if(temp_y >= iHeight)
				temp_y=iHeight-1;
				
		    if(offset + threadPosInBlock < totalPixels)  
				      sdata[offset + threadPosInBlock] = inputImage[temp_x + temp_y * iPitch];
	    offset += blockPixels; 
	    
	  }
	  
	  __syncthreads();
	  
	    // convolution -  image on shared memory 
	  
	    const int kWidth  = (kRadiusX<<1) + 1;
	  	const int kHeight = (kRadiusY<<1) + 1;
	  	
	  if(thposX < iWidth && thposY < iHeight)
	  {
	    float sum = 0.0f;
	    for (int y = 0; y < kHeight; ++y)
	    	for (int x = 0; x < kWidth; ++x) {
	    		const int sharedY = threadIdx.y + y;
	    		const int sharedX = threadIdx.x + x;
	    		sum += sdata[sharedY * width + sharedX] * constKernel[y * kWidth + x];
	    	}

	    outputImage[thposX + thposY * iPitch] = sum; 	
	  }

} 


// mode 6: using texture memory for image and constant memory for kernel access
__global__ void gpu_convolutionGrayImage_tex_cm_d(const float *inputImage, float *outputImage,
    int iWidth, int iHeight, int kRadiusX, int kRadiusY,
    size_t iPitch)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= iWidth || y >= iHeight) return;
  
  const float xx = (float)x;
  const float yy = (float)y;
  const int kWidth = (kRadiusX<<1) + 1;

  int xk, yk;
  float value = 0.0f;

  for (xk=-kRadiusX; xk <= kRadiusX; xk++) {
    for (yk=-kRadiusY; yk <= kRadiusY; yk++) {
      value += tex2D(tex_Image,xx+xk,yy+yk) * constKernel[(yk+kRadiusY)*kWidth + xk+kRadiusX];
    }
  }

  outputImage[y*iPitch + x] = value;
}





void gpu_convolutionGrayImage(const float *inputImage, const float *kernel, float *outputImage, 
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode)
{
  size_t iPitchBytes, kPitchBytes;
  size_t iPitch, kPitch;
  float *d_inputImage;
  float *d_kernel;
  float *d_outputImage;

  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  assert(kWidth*kHeight <= MAXKERNELSIZE);
  
  // allocate device memory
  cutilSafeCall( cudaMallocPitch( (void**)&d_inputImage, &iPitchBytes, iWidth*sizeof(float), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&d_outputImage, &iPitchBytes, iWidth*sizeof(float), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&d_kernel, &kPitchBytes, kWidth*sizeof(float), kHeight ) );  
  iPitch = iPitchBytes/sizeof(float);
  kPitch = kPitchBytes/sizeof(float);
  //std::cout << "iPitchBytes=" << iPitchBytes << " iPitch=" << iPitch << " kPitchBytes=" << kPitchBytes << " kPitch=" << kPitch << std::endl;
  
  cutilSafeCall( cudaMemcpy2D(d_inputImage, iPitchBytes, inputImage, iWidth*sizeof(float), iWidth*sizeof(float), iHeight, cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy2D(d_kernel, kPitchBytes, kernel, kWidth*sizeof(float), kWidth*sizeof(float), kHeight, cudaMemcpyHostToDevice) );


  gpu_bindConstantMemory(kernel, kWidth*kHeight);
  gpu_bindTextureMemory(d_inputImage, iWidth, iHeight, iPitchBytes);

  dim3 blockSize(BW,BH);  
  dim3 gridSize( ((iWidth%BW) ? (iWidth/BW+1) : (iWidth/BW)), ((iHeight%BH) ? (iHeight/BH+1) : (iHeight/BH)) );

  
  // invoke the kernel of your choice here
  const int smSize =  (blockSize.x+(kRadiusX<<1)) * (blockSize.y+(kRadiusY<<1)) * sizeof(float);  

  switch(mode) {
    case 1:
      gpu_convolutionGrayImage_gm_d<<<gridSize,blockSize>>>(d_inputImage, d_kernel, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch, kPitch);
      break;
    case 2:
      gpu_convolutionGrayImage_gm_cm_d<<<gridSize,blockSize,smSize>>>(d_inputImage, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
      break;
    case 3:
      gpu_convolutionGrayImage_sm_d<<<gridSize,blockSize>>>(d_inputImage, d_kernel, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch, kPitch);
      break;
    case 4:
      gpu_convolutionGrayImage_sm_cm_d<<<gridSize,blockSize,smSize>>>(d_inputImage, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
      break;
    case 5:
      gpu_convolutionGrayImage_dsm_cm_d<<<gridSize,blockSize,smSize>>>(d_inputImage, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
      break;
    case 6:
      gpu_convolutionGrayImage_tex_cm_d<<<gridSize,blockSize,smSize>>>(d_inputImage, d_outputImage,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
      break;
    default:
      std::cout << "gpu_convolutionGrayImage() Warning: mode " << mode << " is not supported." << std::endl;
  }

  cutilSafeCall( cudaThreadSynchronize() );
  cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float), d_outputImage, iPitchBytes, iWidth*sizeof(float), iHeight, cudaMemcpyDeviceToHost) );
  

  // free memory
  gpu_unbindTextureMemory();
  cutilSafeCall( cudaFree(d_inputImage) );
  cutilSafeCall( cudaFree(d_outputImage) );
  cutilSafeCall( cudaFree(d_kernel) );
}




//----------------------------------------------------------------------------
// RGB Image Functions (for separated color channels)
//----------------------------------------------------------------------------



void gpu_convolutionRGB(const float *inputImage, const float *kernel, float *outputImage, 
                        int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode)
{
  const int imgSize = iWidth*iHeight;
  gpu_convolutionGrayImage(inputImage, kernel, outputImage, iWidth, iHeight, kRadiusX, kRadiusY, mode);
  gpu_convolutionGrayImage(inputImage+imgSize, kernel, outputImage+imgSize, iWidth, iHeight, kRadiusX, kRadiusY, mode);
  gpu_convolutionGrayImage(inputImage+(imgSize<<1), kernel, outputImage+(imgSize<<1), iWidth, iHeight, kRadiusX, kRadiusY, mode);
}




//----------------------------------------------------------------------------
// RGB Image Functions (for interleaved color channels)
//----------------------------------------------------------------------------



// mode 5 (interleaved): using shared memory for image and constant memory for kernel access
__global__ void gpu_convolutionInterleavedRGB_dsm_cm_d(const float3 *inputImage, float3 *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitchBytes)
{
	extern __shared__ float3 sdata3[];

	// Global position of thread in image coordinates 
	const uint thposX = threadIdx.x + blockIdx.x * blockDim.x;
	const uint thposY = threadIdx.y + blockIdx.y * blockDim.y;

	//	  outputImage[thposX + thposY * iPitch] = 0;  // Debug black

	// apron => extra boundary space. width = block + apron region
	const uint width = (kRadiusX << 1) + BW; 	
	const uint height = (kRadiusY << 1) + BH;

	const uint blockPixels = BW * BH;   		// 16x16 number of pixels in block   BW = blockDim.x 
	const uint totalPixels = width * height;	// number of pixels in (256 + apron pixels)

	// (block+apron)-region in image pixel coordinates  
	// first pixel of each block, same for all threads in a particular block
	const uint startX = blockIdx.x * blockDim.x - kRadiusX;
	const uint startY = blockIdx.y * blockDim.y - kRadiusY;

	// x + y*rowsize  => linear pos of the pixel thread in block
	const uint threadPosInBlock = threadIdx.x + threadIdx.y * blockDim.x;

	uint offset = 0;

	while(offset < totalPixels)
	{		    
		int temp_x,temp_y;
		temp_x=startX + ((threadPosInBlock + offset) % width);
		temp_y=startY + ((threadPosInBlock + offset) / width);
		if(temp_x < 0)
			temp_x=0;
		if(temp_x >= iWidth)
			temp_x=iWidth-1;
		if(temp_y < 0)
			temp_y=0;
		if(temp_y >= iHeight)
			temp_y=iHeight-1;

		if(offset + threadPosInBlock < totalPixels)
			sdata3[offset + threadPosInBlock] = *((float3*)(((char*)inputImage) + temp_y*iPitchBytes) + temp_x);
		//	 sdata3[offset + threadPosInBlock] = inputImage[temp_x + temp_y * iPitchBytes];
		offset += blockPixels; 

	}

	__syncthreads();
		
	// convolution -  image on shared memory 

	const int kWidth  = (kRadiusX<<1) + 1;
	const int kHeight = (kRadiusY<<1) + 1;

	if(thposX < iWidth && thposY < iHeight)
	{
		float3 sum = make_float3(0.0f, 0.0f, 0.0f);

		for (int y = 0; y < kHeight; ++y)
		{
			for (int x = 0; x < kWidth; ++x) {
				const int sharedY = threadIdx.y + y;
				const int sharedX = threadIdx.x + x;

				sum.x += constKernel[y * kWidth + x] * sdata3[sharedX + sharedY * width].x;
				sum.y += constKernel[y * kWidth + x] * sdata3[sharedX + sharedY * width].y;
				sum.z += constKernel[y * kWidth + x] * sdata3[sharedX + sharedY * width].z;
//				sum += sdata[sharedY * width + sharedX] * constKernel[y * kWidth + x];
			}
		}
		*((float3*)(((char*)outputImage) + thposY*iPitchBytes)+ thposX) = sum;	
		
	}
	


} 




__global__ void gpu_ImageFloat3ToFloat4_d(const float3 *inputImage, float4 *outputImage, int iWidth, int iHeight, size_t iPitchBytes, size_t oPitchBytes)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= iWidth || y >= iHeight) return;

  float3 rgb = *((float3*)((char*)inputImage + y*iPitchBytes)+x);
  *((float4*)(((char*)outputImage) + y*oPitchBytes)+ x) = make_float4(rgb.x, rgb.y, rgb.z, 0.0f);
}





// mode 6 (interleaved): using texture memory for image and constant memory for kernel access
__global__ void gpu_convolutionInterleavedRGB_tex_cm_d(float3 *outputImage,
    int iWidth, int iHeight, int kRadiusX, int kRadiusY, size_t oPitchBytes)
{
	// ### implement me ### 
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= iWidth || y >= iHeight) return;

	const float xx = (float)x;
	const float yy = (float)y;
	const int kWidth = (kRadiusX<<1) + 1;

	int xk, yk;
	float3 value = make_float3(0.0f, 0.0f, 0.0f);

	float4 imageVal;
	for (xk=-kRadiusX; xk <= kRadiusX; xk++) 	  {
		for (yk=-kRadiusY; yk <= kRadiusY; yk++)   {
			imageVal = tex2D(tex_ImageF4,xx+xk,yy+yk);
			value.x += constKernel[(yk+kRadiusY)*kWidth + xk+kRadiusX] * imageVal.x;
			value.y += constKernel[(yk+kRadiusY)*kWidth + xk+kRadiusX] * imageVal.y;
			value.z += constKernel[(yk+kRadiusY)*kWidth + xk+kRadiusX] * imageVal.z;
		}
	}

	*((float3*)(((char*)outputImage) + y*oPitchBytes) + x) = value;
//	*((float3*)(((char*)outputImage) + y*oPitchBytes) + x) = make_float3(0,0,0);
}


void gpu_convolutionInterleavedRGB(const float *inputImage, const float *kernel, float *outputImage,
                                   int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode)
{
  size_t iPitchBytesF3, iPitchBytesF4;
  float3 *d_inputImageF3, *d_outputImageF3;
  float4 *d_inputImageF4;
  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  //  allocate memory and copy data
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImageF3), &iPitchBytesF3, iWidth*sizeof(float3), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_outputImageF3), &iPitchBytesF3, iWidth*sizeof(float3), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImageF4), &iPitchBytesF4, iWidth*sizeof(float4), iHeight ) );

  cutilSafeCall( cudaMemcpy2D(d_inputImageF3, iPitchBytesF3, inputImage, iWidth*sizeof(float3), iWidth*sizeof(float3), iHeight, cudaMemcpyHostToDevice) );


  dim3 blockSize(BW,BH);  
  dim3 gridSize( ((iWidth%BW) ? (iWidth/BW+1) : (iWidth/BW)), ((iHeight%BH) ? (iHeight/BH+1) : (iHeight/BH)) );
  int smSizeF3 =  (blockSize.x+(kRadiusX<<1)) * (blockSize.y+(kRadiusY<<1)) * sizeof(float3);
  
  // convert image from float3* to float4*
  gpu_ImageFloat3ToFloat4_d<<<gridSize, blockSize>>>(d_inputImageF3, d_inputImageF4, iWidth, iHeight, iPitchBytesF3, iPitchBytesF4);
  
  gpu_bindConstantMemory(kernel, kWidth*kHeight);
  gpu_bindTextureMemoryF4(d_inputImageF4, iWidth, iHeight, iPitchBytesF4);


  switch(mode) {
    case 1:
    case 2:
    case 3:
    case 4:
      std::cout << "gpu_convolutionInterleavedRGB() Warning: mode " << mode << " is not supported." << std::endl;
      break;
    case 5:
      gpu_convolutionInterleavedRGB_dsm_cm_d<<<gridSize,blockSize,smSizeF3>>>(d_inputImageF3, d_outputImageF3,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitchBytesF3);
      break;
    case 6:
      gpu_convolutionInterleavedRGB_tex_cm_d<<<gridSize,blockSize>>>(d_outputImageF3,
        iWidth, iHeight, kRadiusX, kRadiusY, iPitchBytesF3);
      break;
    default:
      std::cout << "gpu_convolutionInterleavedRGB() Warning: mode " << mode << " is not supported." << std::endl;
  }

  cutilSafeCall( cudaThreadSynchronize() );
  cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float3), d_outputImageF3, iPitchBytesF3, iWidth*sizeof(float3), iHeight, cudaMemcpyDeviceToHost) );


  // free memory
  gpu_unbindTextureMemoryF4();
  cutilSafeCall( cudaFree(d_inputImageF4) );
  cutilSafeCall( cudaFree(d_inputImageF3) );
  cutilSafeCall( cudaFree(d_outputImageF3) );
}












//----------------------------------------------------------------------------
// Benchmark Functions
//----------------------------------------------------------------------------




void gpu_convolutionKernelBenchmarkGrayImage(const float *inputImage, const float *kernel, float *outputImage,
                                             int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                             int numKernelTestCalls)
{
  size_t iPitchBytes, kPitchBytes;
  size_t iPitch, kPitch;
  clock_t startTime, endTime;
  float *d_inputImage, *d_kernel, *d_outputImage;
  float fps;

  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  assert(kWidth*kHeight <= MAXKERNELSIZE);

  dim3 blockSize(BW,BH);  
  dim3 gridSize( ((iWidth%BW) ? (iWidth/BW+1) : (iWidth/BW)), ((iHeight%BH) ? (iHeight/BH+1) : (iHeight/BH)) );
  int smSize =  (blockSize.x+(kRadiusX<<1)) * (blockSize.y+(kRadiusY<<1)) * sizeof(float);

  //  allocate memory and copy data
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImage), &iPitchBytes, iWidth*sizeof(float), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_outputImage), &iPitchBytes, iWidth*sizeof(float), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_kernel), &kPitchBytes, kWidth*sizeof(float), kHeight ) );   
  iPitch = iPitchBytes/sizeof(float);
  kPitch = kPitchBytes/sizeof(float);
  
  cutilSafeCall( cudaMemcpy2D(d_inputImage, iPitchBytes, inputImage, iWidth*sizeof(float), iWidth*sizeof(float), iHeight, cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy2D(d_kernel, kPitchBytes, kernel, kWidth*sizeof(float), kWidth*sizeof(float), kHeight, cudaMemcpyHostToDevice) );

  gpu_bindConstantMemory(kernel, kWidth*kHeight);
  gpu_bindTextureMemory(d_inputImage, iWidth, iHeight, iPitchBytes);

  // --- global memory only ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_gm_d<<<gridSize,blockSize>>>(d_inputImage, d_kernel, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch, kPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  fprintf(stderr, "%f fps - global memory only\n",fps);
  
  
  // --- global memory for image and constant memory for kernel access ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_gm_cm_d<<<gridSize,blockSize>>>(d_inputImage, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  fprintf(stderr, "%f fps - global memory for image & constant memory for kernel access\n",fps);


  // --- shared memory for image and global memory for kernel access ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_sm_d<<<gridSize,blockSize>>>(d_inputImage, d_kernel, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch, kPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  fprintf(stderr, "%f fps - shared memory for image & global memory for kernel access\n",fps);

  
  // --- shared memory for image and constant memory for kernel access ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_sm_cm_d<<<gridSize,blockSize>>>(d_inputImage, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  fprintf(stderr, "%f fps - shared memory for image & constant memory for kernel access\n",fps);


   // --- shared memory for image and constant memory for kernel access ---  
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_dsm_cm_d<<<gridSize,blockSize,smSize>>>(d_inputImage, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  fprintf(stderr, "%f fps - dyn. shared memory for image & const memory for kernel access\n",fps);



  // --- texture memory for image and constant memory for kernel access ---
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionGrayImage_tex_cm_d<<<gridSize,blockSize>>>(d_inputImage, d_outputImage, iWidth, iHeight, kRadiusX, kRadiusY, iPitch);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC;
  fprintf(stderr, "%f fps - texture memory for image & const memory for kernel access\n",fps);


  cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float), d_outputImage, iPitchBytes, iWidth*sizeof(float), iHeight, cudaMemcpyDeviceToHost) );

  // free memory
  gpu_unbindTextureMemory();
  cutilSafeCall( cudaFree(d_inputImage) );
  cutilSafeCall( cudaFree(d_outputImage) );
  cutilSafeCall( cudaFree(d_kernel) );
}



void gpu_convolutionKernelBenchmarkInterleavedRGB(const float *inputImage, const float *kernel, float *outputImage,
                                                  int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                                  int numKernelTestCalls)
{
  size_t iPitchBytesF3, iPitchBytesF4;
  clock_t startTime, endTime;
  float3 *d_inputImageF3, *d_outputImageF3;
  float4 *d_inputImageF4;
  float fps;

  const int kWidth  = (kRadiusX<<1) + 1;
  const int kHeight = (kRadiusY<<1) + 1;

  assert(kWidth*kHeight <= MAXKERNELSIZE);

  dim3 blockSize(BW,BH);  
  dim3 gridSize( ((iWidth%BW) ? (iWidth/BW+1) : (iWidth/BW)), ((iHeight%BH) ? (iHeight/BH+1) : (iHeight/BH)) );
  int smSizeF3 =  (blockSize.x+(kRadiusX<<1)) * (blockSize.y+(kRadiusY<<1)) * sizeof(float3);

  //  allocate memory and copy data
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImageF3), &iPitchBytesF3, iWidth*sizeof(float3), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_inputImageF4), &iPitchBytesF4, iWidth*sizeof(float4), iHeight ) );
  cutilSafeCall( cudaMallocPitch( (void**)&(d_outputImageF3), &iPitchBytesF3, iWidth*sizeof(float3), iHeight ) );   
  cutilSafeCall( cudaMemcpy2D(d_inputImageF3, iPitchBytesF3, inputImage, iWidth*sizeof(float3), iWidth*sizeof(float3), iHeight, cudaMemcpyHostToDevice) );

  gpu_bindConstantMemory(kernel, kWidth*kHeight);
  
  
  // --- shared memory for interleaved image and constant memory for kernel access ---  
  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionInterleavedRGB_dsm_cm_d<<<gridSize,blockSize,smSizeF3>>>(d_inputImageF3, d_outputImageF3, iWidth, iHeight, kRadiusX, kRadiusY, iPitchBytesF3);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC * 3;
  fprintf(stderr, "%f fps - dyn. shared mem for interleaved img & const mem for kernel\n",fps);


  // --- texture memory for interleaved image and constant memory for kernel access ---
  gpu_ImageFloat3ToFloat4_d<<<gridSize, blockSize>>>(d_inputImageF3, d_inputImageF4, iWidth, iHeight, iPitchBytesF3, iPitchBytesF4);
  gpu_bindTextureMemoryF4(d_inputImageF4, iWidth, iHeight, iPitchBytesF4);

  startTime = clock();
  for(int c=0;c<numKernelTestCalls;c++) {
    gpu_convolutionInterleavedRGB_tex_cm_d<<<gridSize,blockSize>>>(d_outputImageF3, iWidth, iHeight, kRadiusX, kRadiusY, iPitchBytesF3);
    cutilSafeCall( cudaThreadSynchronize() );
  }
  endTime = clock();
  fps = (float)numKernelTestCalls / float(endTime - startTime) * CLOCKS_PER_SEC * 3;
  fprintf(stderr, "%f fps - texture mem for interleaved img & const mem for kernel access\n",fps);


  cutilSafeCall( cudaMemcpy2D(outputImage, iWidth*sizeof(float3), d_outputImageF3, iPitchBytesF3, iWidth*sizeof(float3), iHeight, cudaMemcpyDeviceToHost) );
  

  // free memory
  gpu_unbindTextureMemoryF4();
  cutilSafeCall( cudaFree(d_inputImageF3) );
  cutilSafeCall( cudaFree(d_outputImageF3) );
  cutilSafeCall( cudaFree(d_inputImageF4) );
}





void gpu_bindConstantMemory(const float *kernel, int size) 
{
  cutilSafeCall( cudaMemcpyToSymbol( (const char *)&constKernel, kernel, size*sizeof(float)) );
}



void gpu_bindTextureMemory(float *d_inputImage, int iWidth, int iHeight, size_t iPitchBytes)
{
  // >>>> prepare usage of texture memory
  tex_Image.addressMode[0] = cudaAddressModeClamp;
  tex_Image.addressMode[1] = cudaAddressModeClamp;
  tex_Image.filterMode = cudaFilterModeLinear;
  tex_Image.normalized = false;
  // <<<< prepare usage of texture memory

  cutilSafeCall( cudaBindTexture2D(0, &tex_Image, d_inputImage, &tex_Image_desc, iWidth, iHeight, iPitchBytes) );
}


void gpu_unbindTextureMemory()
{
  cutilSafeCall( cudaUnbindTexture(tex_Image) );
}



void gpu_bindTextureMemoryF4(float4 *d_inputImageF4, int iWidth, int iHeight, size_t iPitchBytesF4)
{
  // >>>> prepare usage of texture memory
  tex_ImageF4.addressMode[0] = cudaAddressModeClamp;
  tex_ImageF4.addressMode[1] = cudaAddressModeClamp;
  tex_ImageF4.filterMode = cudaFilterModeLinear;
  tex_ImageF4.normalized = false;
  // <<<< prepare usage of texture memory

  cutilSafeCall( cudaBindTexture2D(0, &tex_ImageF4, d_inputImageF4, &tex_Image_descF4, iWidth, iHeight, iPitchBytesF4) );
}


void gpu_unbindTextureMemoryF4()
{
  cutilSafeCall( cudaUnbindTexture(tex_ImageF4) );
}
