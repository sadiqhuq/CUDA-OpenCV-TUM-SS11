/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: diffusion
* file:    diffusion.cu
*
* 
\********* PLEASE ENTER YOUR CORRECT STUDENT NAME AND ID BELOW **************/
const char* studentName = "Sadiq Huq";
const int   studentID   = 3273623;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* diffuse_linear_isotrop_shared(const float  *d_input, ... ) - Done
* diffuse_linear_isotrop_shared(const float3 *d_input, ... ) - Done
* diffuse_nonlinear_isotrop_shared(const float  *d_input, ... ) - Done
* diffuse_nonlinear_isotrop_shared(const float3 *d_input, ... ) - Done 
* compute_tv_diffusivity_shared - Done 
* compute_tv_diffusivity_joined_shared - Done 
* compute_tv_diffusivity_separate_shared - Done
* jacobi_shared(float  *d_output, ... ) - Done 
* jacobi_shared(float3 *d_output, ... ) - Done
* sor_shared(float  *d_output, ... ) - Done
* sor_shared(float3 *d_output, ... ) - Done
*
\****************************************************************************/


#define DIFF_BW 16
#define DIFF_BH 16

#define TV_EPSILON 0.1f



#include <cutil.h>
#include <cutil_inline.h>

#include "diffusion.cuh"



__host__ const char* getStudentName() { return studentName; };
__host__ int         getStudentID()   { return studentID; };
__host__ bool        checkStudentNameAndID() { return strcmp(studentName, "John Doe") != 0 && studentID != 1234567; };


//----------------------------------------------------------------------------
// Linear Diffusion
//----------------------------------------------------------------------------


__global__ void diffuse_linear_isotrop_shared(
  const float *d_input,
  float *d_output,
  float timeStep, 
  int nx, int ny,
  size_t pitch)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const int idx = y*pitch + x;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

	  u[tx][ty] = d_input[idx];

	  if (x == 0)  u[0][ty] = u[tx][ty];
	  else if (x == nx-1) u[tx+1][ty] = u[tx][ty];
	  else {
		  if (threadIdx.x == 0) u[0][ty] = d_input[idx-1];
		  else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = d_input[idx+1];
	  }

	  if (y == 0)  u[tx][0] = u[tx][ty];
	  else if (y == ny-1) u[tx][ty+1] = u[tx][ty];
	  else {
		  if (threadIdx.y == 0) u[tx][0] = d_input[idx-pitch];
		  else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = d_input[idx+pitch];
	  }
  }

  __syncthreads();

  // ### implement me ###
  
  if(x < nx && y < ny)
  {
	  float RHS = u[tx][ty] + ( timeStep * ( u[tx+1][ty] + u[tx-1][ty] + u[tx][ty+1] + u[tx][ty-1] - 4*u[tx][ty] ) )  ;
	  d_output[idx] = RHS;	  
  }

}


__global__ void diffuse_linear_isotrop_shared
(
 const float3 *d_input,
 float3 *d_output,
 float timeStep,
 int nx, int ny,
 size_t pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  float3 imgValue;

  // load data into shared memory
  if (x < nx && y < ny) {

    imgValue = *( (float3*)imgP );
    u[tx][ty] = imgValue;

    if (x == 0)  u[0][ty] = imgValue;
    else if (x == nx-1) u[tx+1][ty] = imgValue;
    else {
      if (threadIdx.x == 0) u[0][ty] = *( ((float3*)imgP)-1 );
      else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *( ((float3*)imgP)+1 );
    }

    if (y == 0)  u[tx][0] = imgValue;
    else if (y == ny-1) u[tx][ty+1] = imgValue;
    else {
      if (threadIdx.y == 0) u[tx][0] = *( (float3*)(imgP-pitchBytes) );
      else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
    }
  }

  __syncthreads();
  
  // ### implement me ###
  
  float3 RHS3 = make_float3 (
		  u[tx][ty].x + ( timeStep * ( u[tx+1][ty].x + u[tx-1][ty].x + u[tx][ty+1].x + u[tx][ty-1].x - 4*u[tx][ty].x ) ),
		  u[tx][ty].y + ( timeStep * ( u[tx+1][ty].y + u[tx-1][ty].y + u[tx][ty+1].y + u[tx][ty-1].y - 4*u[tx][ty].y ) ),
		  u[tx][ty].z + ( timeStep * ( u[tx+1][ty].z + u[tx-1][ty].z + u[tx][ty+1].z + u[tx][ty-1].z - 4*u[tx][ty].z ) ) );

  *((float3*)(((char*)d_output) + y*pitchBytes)+ x) = RHS3 ;
}




//----------------------------------------------------------------------------
// Non-linear Diffusion - explicit scheme
//----------------------------------------------------------------------------




__global__ void diffuse_nonlinear_isotrop_shared
(
 const float *d_input,
 const float *d_diffusivity,
 float *d_output,
 float timeStep,
 int   nx,
 int   ny,
 size_t   pitch
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const int idx = y*pitch + x;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = d_input[idx];
    g[tx][ty] = d_diffusivity[idx];

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else {
      if (threadIdx.x == 0) {
        u[0][ty] = d_input[idx-1];
        g[0][ty] = d_diffusivity[idx-1];
      }
      else if (threadIdx.x == blockDim.x-1) {
        u[tx+1][ty] = d_input[idx+1];
        g[tx+1][ty] = d_diffusivity[idx+1];
      }
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else {
      if (threadIdx.y == 0) {
        u[tx][0] = d_input[idx-pitch];
        g[tx][0] = d_diffusivity[idx-pitch];
      }
      else if (threadIdx.y == blockDim.y-1) {
        u[tx][ty+1] = d_input[idx+pitch];
        g[tx][ty+1] = d_diffusivity[idx+pitch];
      }
    }
  }

  __syncthreads();
  
  // ### implement me ###
  
  if(x < nx && y < ny)
  {
	  float alpha = 0.5 * ( g[tx+1][ty] + g[tx][ty] );
	  float beta  = 0.5 * ( g[tx][ty] + g[tx-1][ty] );
	  float gamma = 0.5 * ( g[tx][ty+1] + g[tx][ty] );
	  float sigma = 0.5 * ( g[tx][ty] + g[tx][ty-1] );
	  
	  
	  float RHS = alpha * u[tx+1][ty] + beta * u[tx-1][ty] + gamma * u[tx][ty+1] + sigma * u[tx][ty-1] - (alpha+beta+gamma+sigma)*u[tx][ty] ;
	  d_output[idx] = u[tx][ty] + timeStep * RHS;	  
	  

  }
  
}




__global__ void diffuse_nonlinear_isotrop_shared
(
 const float3 *d_input,
 const float3 *d_diffusivity,
 float3 *d_output,
 float timeStep,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* diffP = (char*)d_diffusivity + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float3 g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = *( (float3*)imgP );
    g[tx][ty] = *( (float3*)diffP );

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else {
      if (threadIdx.x == 0) {
        u[0][ty] = *( ((float3*)imgP)-1 );
        g[0][ty] = *( ((float3*)diffP)-1 );
      }
      else if (threadIdx.x == blockDim.x-1) {
        u[tx+1][ty] = *( ((float3*)imgP)+1 );
        g[tx+1][ty] = *( ((float3*)diffP)+1 );
      }
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else {
      if (threadIdx.y == 0) {
        u[tx][0] = *( (float3*)(imgP-pitchBytes) );
        g[tx][0] = *( (float3*)(diffP-pitchBytes) );
      }
      else if (threadIdx.y == blockDim.y-1) {
        u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
        g[tx][ty+1] = *( (float3*)(diffP+pitchBytes) );
      }
    }
  }

  __syncthreads();

  
//   ### implement me ###
//
  if(x < nx && y < ny)
  {
	  float3 alpha = make_float3( 
			  0.5 * ( g[tx+1][ty].x + g[tx][ty].x),
			  0.5 * ( g[tx+1][ty].y + g[tx][ty].y),
			  0.5 * ( g[tx+1][ty].z + g[tx][ty].z) );

	  float3 beta  = make_float3( 
			  0.5 * ( g[tx][ty].x + g[tx-1][ty].x ),
			  0.5 * ( g[tx][ty].y + g[tx-1][ty].y ),
			  0.5 * ( g[tx][ty].z + g[tx-1][ty].z ));
	  
	  float3 gamma = make_float3(
			  0.5 * ( g[tx][ty+1].x + g[tx][ty].x ),
			  0.5 * ( g[tx][ty+1].y + g[tx][ty].y ),
			  0.5 * ( g[tx][ty+1].z + g[tx][ty].z ) );
	  
	  float3 sigma = make_float3(
			  0.5 * ( g[tx][ty].x + g[tx][ty-1].x ),
			  0.5 * ( g[tx][ty].y + g[tx][ty-1].y ),
			  0.5 * ( g[tx][ty].z + g[tx][ty-1].z ) );
	  
//	  float3 RHS3 = alpha * u[tx+1][ty] + beta * u[tx-1][ty] + gamma * u[tx][ty+1] + sigma * u[tx][ty-1] - (alpha+beta+gamma+sigma)*u[tx][ty] ;
//	  d_output[idx] = u[tx][ty] + timeStep * RHS;	  
	
	    float3 RHS3 = make_float3 (
	  		  alpha.x * u[tx+1][ty].x + beta.x * u[tx-1][ty].x + gamma.x * u[tx][ty+1].x + sigma.x * u[tx][ty-1].x - (alpha.x+beta.x+gamma.x+sigma.x)*u[tx][ty].x,
	  		  alpha.y * u[tx+1][ty].y + beta.y * u[tx-1][ty].y + gamma.y * u[tx][ty+1].y + sigma.y * u[tx][ty-1].y - (alpha.y+beta.y+gamma.y+sigma.y)*u[tx][ty].y,
	  		  alpha.z * u[tx+1][ty].z + beta.z * u[tx-1][ty].z + gamma.z * u[tx][ty+1].z + sigma.z * u[tx][ty-1].z - (alpha.z+beta.z+gamma.z+sigma.z)*u[tx][ty].z);
	  
	     *((float3*)(((char*)d_output) + y*pitchBytes)+ x) =  make_float3 ( 
	    		 u[tx][ty].x + timeStep * RHS3.x ,
	    		 u[tx][ty].y + timeStep * RHS3.y,
	    		 u[tx][ty].z + timeStep * RHS3.z );


  
  }
 

}



__global__ void compute_tv_diffusivity_shared
(
 const float *d_input,
 float *d_output,
 int   nx,
 int   ny,
 size_t   pitch
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const int idx = y*pitch + x;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = d_input[idx];

    if (x == 0)  u[0][ty] = u[tx][ty];
    else if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else {
      if (threadIdx.x == 0) u[0][ty] = d_input[idx-1];
      else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = d_input[idx+1];
    }

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else {
      if (threadIdx.y == 0) u[tx][0] = d_input[idx-pitch];
      else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = d_input[idx+pitch];
    }
  }

  __syncthreads();

  
  // make use of the constant TV_EPSILON

  // ### implement me ###
  
  
  if(x < nx && y < ny)
  {
    float dx = (u[tx + 1][ty] - u[tx - 1][ty]);
    float dy = (u[tx][ty + 1] - u[tx][ty - 1]);

    d_output[idx] = 1.0f / sqrtf(dx * dx + dy * dy + TV_EPSILON);
  }
  

}



__global__ void compute_tv_diffusivity_joined_shared
(
 const float3 *d_input,
 float3 *d_output,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = *( (float3*)imgP );

    if (x == 0)  u[0][ty] = u[tx][ty];
    else if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else {
      if (threadIdx.x == 0) u[0][ty] = *( ((float3*)imgP)-1 );
      else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *( ((float3*)imgP)+1 );
    }

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else {
      if (threadIdx.y == 0) u[tx][0] = *( (float3*)(imgP-pitchBytes) );
      else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
    }
  }

  __syncthreads();
  
  // make use of the constant TV_EPSILON

  // ### implement me ###
  
  if(x < nx && y < ny)
    {
	  float3 dx = make_float3 (
			  0.5*(u[tx + 1][ty].x - u[tx - 1][ty].x),
			  0.5*(u[tx + 1][ty].y - u[tx - 1][ty].y),
			  0.5*(u[tx + 1][ty].z - u[tx - 1][ty].z) );
	  
	  float3 dy = make_float3(
			  0.5*(u[tx][ty + 1].x - u[tx][ty - 1].x),
			  0.5*(u[tx][ty + 1].y - u[tx][ty - 1].y),
			  0.5*(u[tx][ty + 1].z - u[tx][ty - 1].z) );


      float3 RHS3 = make_float3 (
    		  1.0f / sqrtf(dx.x * dx.x + dy.x * dy.x + TV_EPSILON),
    		  1.0f / sqrtf(dx.y * dx.y + dy.y * dy.y + TV_EPSILON),
    		  1.0f / sqrtf(dx.z * dx.z + dy.z * dy.z + TV_EPSILON) );
      
      const float diffuse = ( (RHS3.x+RHS3.y+RHS3.z)/3 );
      *((float3*)(((char*)d_output) + y*pitchBytes)+ x) = make_float3(10,10,10) ; // Debug
      *((float3*)(((char*)d_output) + y*pitchBytes)+ x) = make_float3(diffuse,diffuse,diffuse) ;
      

    }
    
    
  
}



__global__ void compute_tv_diffusivity_separate_shared
(
 const float3 *d_input,
 float3 *d_output,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];

  // load data into shared memory
  if (x < nx && y < ny) {

    u[tx][ty] = *( (float3*)imgP );

    if (x == 0)  u[threadIdx.x][ty] = u[tx][ty];
    else if (x == nx-1) u[tx+1][ty] = u[tx][ty];
    else {
      if (threadIdx.x == 0) u[0][ty] = *( ((float3*)imgP)-1 );
      else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *( ((float3*)imgP)+1 );
    }

    if (y == 0)  u[tx][0] = u[tx][ty];
    else if (y == ny-1) u[tx][ty+1] = u[tx][ty];
    else {
      if (threadIdx.y == 0) u[tx][0] = *( (float3*)(imgP-pitchBytes) );
      else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
    }
  }

  __syncthreads();


  // make use of the constant TV_EPSILON
 

  if(x < nx && y < ny)
  {
	  float3 result;
	  float dx = 0.5f*(u[tx+1][ty].x-u[tx-1][ty].x);
	  float dy = 0.5f*(u[tx][ty+1].x-u[tx][ty-1].x);
	  result.x = 1.0f / (sqrt(dx*dx+dy*dy+TV_EPSILON));

	  dx = 0.5f*(u[tx+1][ty].y-u[tx-1][ty].y);
	  dy = 0.5f*(u[tx][ty+1].y-u[tx][ty-1].y);
	  result.y = 1.0f / (sqrt(dx*dx+dy*dy+TV_EPSILON));

	  dx = 0.5f*(u[tx+1][ty].z-u[tx-1][ty].z);
	  dy = 0.5f*(u[tx][ty+1].z-u[tx][ty-1].z);
	  result.z = 1.0f / (sqrt(dx*dx+dy*dy+TV_EPSILON));

//	  *((float3*)(((char*)d_output) + y*pitchBytes)+ x) = make_float3(10,10,10) ; // Debug
	  *((float3*)((char*)d_output + y*pitchBytes + x*sizeof(float3))) =result;
	  
  }
}




//----------------------------------------------------------------------------
// Non-linear Diffusion - Jacobi scheme
//----------------------------------------------------------------------------



__global__ void jacobi_shared
(
 float *d_output,
 const float *d_input,
 const float *d_original,
 const float *d_diffusivity,
 float weight,
 float overrelaxation,
 int   nx,
 int   ny,
 size_t   pitch
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int idx = y*pitch + x;

  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = d_input[idx];
    g[tx][ty] = d_diffusivity[idx];

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else {
      if (threadIdx.x == 0) {
        u[0][ty] = d_input[idx-1];
        g[0][ty] = d_diffusivity[idx-1];
      }
      else if (threadIdx.x == blockDim.x-1) {
        u[tx+1][ty] = d_input[idx+1];
        g[tx+1][ty] = d_diffusivity[idx+1];
      }
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else {
      if (threadIdx.y == 0) {
        u[tx][0] = d_input[idx-pitch];
        g[tx][0] = d_diffusivity[idx-pitch];
      }
      else if (threadIdx.y == blockDim.y-1) {
        u[tx][ty+1] = d_input[idx+pitch];
        g[tx][ty+1] = d_diffusivity[idx+pitch];
      }
    }
  }

  __syncthreads();
 


  // ### implement me ###

  
  if (x < nx && y < ny){
	  
	  
	  float alpha = 0.5 * ( g[tx+1][ty] + g[tx][ty]   );
	  float beta  = 0.5 * ( g[tx][ty]   + g[tx-1][ty] );
	  float gamma = 0.5 * ( g[tx][ty+1] + g[tx][ty]   );
	  float sigma = 0.5 * ( g[tx][ty]   + g[tx][ty-1] );

	  // B.C
	  if (x >= nx) alpha = 0;
	  if (x <= 0)  beta = 0;
	  if (y >= ny) gamma = 0;
	  if (y <= 0)  sigma = 0;

	  float RHS = d_original[idx]+ weight * (alpha * u[tx+1][ty] + beta * u[tx-1][ty] + gamma * u[tx][ty+1] + sigma * u[tx][ty-1] );

	  d_output[idx] = RHS /(1 + weight * (alpha + beta + gamma + sigma));
  }

}



__global__ void jacobi_shared
(
 float3 *d_output,
 const float3 *d_input,
 const float3 *d_original,
 const float3 *d_diffusivity,
 float weight,
 float overrelaxation,
 int   nx,
 int   ny,
 size_t   pitchBytes
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* diffP = (char*)d_diffusivity + y*pitchBytes + x*sizeof(float3);

  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float3 g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = *( (float3*)imgP );
    g[tx][ty] = *( (float3*)diffP );

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else {
      if (threadIdx.x == 0) {
        u[0][ty] = *( ((float3*)imgP)-1 );
        g[0][ty] = *( ((float3*)diffP)-1 );
      }
      else if (threadIdx.x == blockDim.x-1) {
        u[tx+1][ty] = *( ((float3*)imgP)+1 );
        g[tx+1][ty] = *( ((float3*)diffP)+1 );
      }
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else {
      if (threadIdx.y == 0) {
        u[tx][0] = *( (float3*)(imgP-pitchBytes) );
        g[tx][0] = *( (float3*)(diffP-pitchBytes) );
      }
      else if (threadIdx.y == blockDim.y-1) {
        u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
        g[tx][ty+1] = *( (float3*)(diffP+pitchBytes) );
      }
    }
  }

  __syncthreads();


  // ### implement me ###

  if (x < nx && y < ny){


	  float3 alpha = make_float3( 
			  0.5 * ( g[tx+1][ty].x + g[tx][ty].x),
			  0.5 * ( g[tx+1][ty].y + g[tx][ty].y),
			  0.5 * ( g[tx+1][ty].z + g[tx][ty].z) );

	  float3 beta  = make_float3( 
			  0.5 * ( g[tx][ty].x + g[tx-1][ty].x ),
			  0.5 * ( g[tx][ty].y + g[tx-1][ty].y ),
			  0.5 * ( g[tx][ty].z + g[tx-1][ty].z ));

	  float3 gamma = make_float3(
			  0.5 * ( g[tx][ty+1].x + g[tx][ty].x ),
			  0.5 * ( g[tx][ty+1].y + g[tx][ty].y ),
			  0.5 * ( g[tx][ty+1].z + g[tx][ty].z ) );

	  float3 sigma = make_float3(
			  0.5 * ( g[tx][ty].x + g[tx][ty-1].x ),
			  0.5 * ( g[tx][ty].y + g[tx][ty-1].y ),
			  0.5 * ( g[tx][ty].z + g[tx][ty-1].z ) );


	  if (x >= nx) alpha = make_float3(0,0,0);
	  if (x <= 0)  beta  = make_float3(0,0,0);
	  if (y >= ny) gamma = make_float3(0,0,0);
	  if (y <= 0)  sigma = make_float3(0,0,0);

	  float3 Img = *((float3*)(((char*)d_original) + y*pitchBytes)+ x) ;

	  float3 RHS3 = make_float3 (
			  alpha.x * u[tx+1][ty].x + beta.x * u[tx-1][ty].x + gamma.x * u[tx][ty+1].x + sigma.x * u[tx][ty-1].x ,
			  alpha.y * u[tx+1][ty].y + beta.y * u[tx-1][ty].y + gamma.y * u[tx][ty+1].y + sigma.y * u[tx][ty-1].y ,
			  alpha.z * u[tx+1][ty].z + beta.z * u[tx-1][ty].z + gamma.z * u[tx][ty+1].z + sigma.z * u[tx][ty-1].z );

	  float3 RHSdenom = make_float3(
			  1 + weight * (alpha.x + beta.x+gamma.x+sigma.x),
			  1 + weight * (alpha.y + beta.y+gamma.y+sigma.y),
			  1 + weight * (alpha.z + beta.z+gamma.z+sigma.z) );

	  *((float3*)(((char*)d_output) + y*pitchBytes)+ x) = make_float3(
			  (Img.x + weight * RHS3.x )/ RHSdenom.x,	  
			  (Img.y + weight * RHS3.y )/ RHSdenom.y,
			  (Img.z + weight * RHS3.z )/ RHSdenom.z) ; 

//	  	  	*((float3*)(((char*)d_output) + y*pitchBytes)+ x) = make_float3 (0,255,255); // Debug

  }
    
}



//----------------------------------------------------------------------------
// Non-linear Diffusion - Successive Over-Relaxation (SOR)
//----------------------------------------------------------------------------


__global__ void sor_shared
(
 float *d_output,
 const float *d_input,
 const float *d_original,
 const float *d_diffusivity,
 float weight,
 float overrelaxation,
 int   nx,
 int   ny,
 size_t   pitch,
 int   red
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int idx = y*pitch + x;
  
  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float g[DIFF_BW+2][DIFF_BH+2];


  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = d_input[idx];
    g[tx][ty] = d_diffusivity[idx];

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else {
      if (threadIdx.x == 0) {
        u[0][ty] = d_input[idx-1];
        g[0][ty] = d_diffusivity[idx-1];
      }
      else if (threadIdx.x == blockDim.x-1) {
        u[tx+1][ty] = d_input[idx+1];
        g[tx+1][ty] = d_diffusivity[idx+1];
      }
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else {
      if (threadIdx.y == 0) {
        u[tx][0] = d_input[idx-pitch];
        g[tx][0] = d_diffusivity[idx-pitch];
      }
      else if (threadIdx.y == blockDim.y-1) {
        u[tx][ty+1] = d_input[idx+pitch];
        g[tx][ty+1] = d_diffusivity[idx+pitch];
      }
    }
  }

  __syncthreads();


  // ### implement me ###
  
  if ( (x+y)%2 == red )
  {
	  if (x < nx && y < ny){


		  float alpha = 0.5 * ( g[tx+1][ty] + g[tx][ty]   );
		  float beta  = 0.5 * ( g[tx][ty]   + g[tx-1][ty] );
		  float gamma = 0.5 * ( g[tx][ty+1] + g[tx][ty]   );
		  float sigma = 0.5 * ( g[tx][ty]   + g[tx][ty-1] );

		  // B.C
		  if (x >= nx) alpha = 0;
		  if (x <= 0)  beta = 0;
		  if (y >= ny) gamma = 0;
		  if (y <= 0)  sigma = 0;

		  float RHS = d_original[idx]+ weight * (alpha * u[tx+1][ty] + beta * u[tx-1][ty] + gamma * u[tx][ty+1] + sigma * u[tx][ty-1] );

		  //	  d_output[idx] = RHS /(1 + weight * (alpha + beta + gamma + sigma));


		  d_output[idx]  = (1 - overrelaxation ) * u[tx][ty] 
		                                                 + overrelaxation *  RHS / (1 + weight * (alpha + beta + gamma + sigma));


		  //	  d_output[idx] = RHS;  // Debug 
	  }
  }


}




__global__ void sor_shared
(
 float3 *d_output,
 const float3 *d_input,
 const float3 *d_original,
 const float3 *d_diffusivity,
 float weight,
 float overrelaxation,
 int   nx,
 int   ny,
 size_t   pitchBytes,
 int   red
 )
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const char* imgP = (char*)d_input + y*pitchBytes + x*sizeof(float3);
  const char* diffP = (char*)d_diffusivity + y*pitchBytes + x*sizeof(float3);

  const int tx = threadIdx.x+1;
  const int ty = threadIdx.y+1;

  __shared__ float3 u[DIFF_BW+2][DIFF_BH+2];
  __shared__ float3 g[DIFF_BW+2][DIFF_BH+2];



  // load data into shared memory
  if (x < nx && y < ny) {
    u[tx][ty] = *( (float3*)imgP );
    g[tx][ty] = *( (float3*)diffP );

    if (x == 0)  {
      u[0][ty] = u[tx][ty];
      g[0][ty] = g[tx][ty];
    }
    else if (x == nx-1) {
      u[tx+1][ty] = u[tx][ty];
      g[tx+1][ty] = g[tx][ty];
    }
    else {
      if (threadIdx.x == 0) {
        u[0][ty] = *( ((float3*)imgP)-1 );
        g[0][ty] = *( ((float3*)diffP)-1 );
      }
      else if (threadIdx.x == blockDim.x-1) {
        u[tx+1][ty] = *( ((float3*)imgP)+1 );
        g[tx+1][ty] = *( ((float3*)diffP)+1 );
      }
    }

    if (y == 0) {
      u[tx][0] = u[tx][ty];
      g[tx][0] = g[tx][ty];
    }
    else if (y == ny-1) {
      u[tx][ty+1] = u[tx][ty];
      g[tx][ty+1] = g[tx][ty];
    }
    else {
      if (threadIdx.y == 0) {
        u[tx][0] = *( (float3*)(imgP-pitchBytes) );
        g[tx][0] = *( (float3*)(diffP-pitchBytes) );
      }
      else if (threadIdx.y == blockDim.y-1) {
        u[tx][ty+1] = *( (float3*)(imgP+pitchBytes) );
        g[tx][ty+1] = *( (float3*)(diffP+pitchBytes) );
      }
    }
  }

  __syncthreads();


  // ### implement me ###
  if ( (x+y)%2== red )
  {
	  if (x < nx && y < ny){


		  float3 alpha = make_float3( 
				  0.5 * ( g[tx+1][ty].x + g[tx][ty].x),
				  0.5 * ( g[tx+1][ty].y + g[tx][ty].y),
				  0.5 * ( g[tx+1][ty].z + g[tx][ty].z) );

		  float3 beta  = make_float3( 
				  0.5 * ( g[tx][ty].x + g[tx-1][ty].x ),
				  0.5 * ( g[tx][ty].y + g[tx-1][ty].y ),
				  0.5 * ( g[tx][ty].z + g[tx-1][ty].z ));

		  float3 gamma = make_float3(
				  0.5 * ( g[tx][ty+1].x + g[tx][ty].x ),
				  0.5 * ( g[tx][ty+1].y + g[tx][ty].y ),
				  0.5 * ( g[tx][ty+1].z + g[tx][ty].z ) );

		  float3 sigma = make_float3(
				  0.5 * ( g[tx][ty].x + g[tx][ty-1].x ),
				  0.5 * ( g[tx][ty].y + g[tx][ty-1].y ),
				  0.5 * ( g[tx][ty].z + g[tx][ty-1].z ) );


		  if (x >= nx) alpha = make_float3(0,0,0);
		  if (x <= 0)  beta  = make_float3(0,0,0);
		  if (y >= ny) gamma = make_float3(0,0,0);
		  if (y <= 0)  sigma = make_float3(0,0,0);

		  float3 Img = *((float3*)(((char*)d_original) + y*pitchBytes)+ x) ;

		  float3 RHS3 = make_float3 (
				  alpha.x * u[tx+1][ty].x + beta.x * u[tx-1][ty].x + gamma.x * u[tx][ty+1].x + sigma.x * u[tx][ty-1].x ,
				  alpha.y * u[tx+1][ty].y + beta.y * u[tx-1][ty].y + gamma.y * u[tx][ty+1].y + sigma.y * u[tx][ty-1].y ,
				  alpha.z * u[tx+1][ty].z + beta.z * u[tx-1][ty].z + gamma.z * u[tx][ty+1].z + sigma.z * u[tx][ty-1].z );

		  float3 RHSdenom = make_float3(
				  1 + weight * (alpha.x + beta.x+gamma.x+sigma.x),
				  1 + weight * (alpha.y + beta.y+gamma.y+sigma.y),
				  1 + weight * (alpha.z + beta.z+gamma.z+sigma.z) );

		  float3 xbar = make_float3(
				  (Img.x + weight * RHS3.x )/ RHSdenom.x,	  
				  (Img.y + weight * RHS3.y )/ RHSdenom.y,
				  (Img.z + weight * RHS3.z )/ RHSdenom.z) ; 

		  //	  	  	*((float3*)(((char*)d_output) + y*pitchBytes)+ x) = make_float3 (0,255,255); // Debug
		  //	  d_output[idx]  = (1 - overrelaxation ) * u[tx][ty] 
		  //	          + overrelaxation *  RHS / (1 + weight * (alpha + beta + gamma + sigma));
		  *((float3*)(((char*)d_output) + y*pitchBytes)+ x) = make_float3 (
		  (1 - overrelaxation )*u[tx][ty].x + overrelaxation * xbar.x,
		  (1 - overrelaxation )*u[tx][ty].y + overrelaxation * xbar.y,
		  (1 - overrelaxation )*u[tx][ty].z + overrelaxation * xbar.z   );
		  
//		  *((float3*)(((char*)d_output) + y*pitchBytes)+ x) = make_float3(0,0,0);
	  }
  }
  
  
  
}




//----------------------------------------------------------------------------
// Host function
//----------------------------------------------------------------------------



void gpu_diffusion
(
 const float *input,
 float *output,
 int nx, int ny, int nc, 
 float timeStep,
 int iterations,
 float weight,
 int lagged_iterations,
 float overrelaxation,
 int mode
 )
{
  int i,j;
  size_t pitchF1, pitchBytesF1, pitchBytesF3;
  float *d_input = 0;
  float *d_output = 0;
  float *d_diffusivity = 0;
  float *d_original = 0;
  float *temp = 0;

  dim3 dimGrid((int)ceil((float)nx/DIFF_BW), (int)ceil((float)ny/DIFF_BH));
  dim3 dimBlock(DIFF_BW,DIFF_BH);

  // Allocation of GPU Memory
  if (nc == 1) {

    cutilSafeCall( cudaMallocPitch( (void**)&(d_input), &pitchBytesF1, nx*sizeof(float), ny ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(d_output), &pitchBytesF1, nx*sizeof(float), ny ) );
    if (mode) cutilSafeCall( cudaMallocPitch( (void**)&(d_diffusivity), &pitchBytesF1, nx*sizeof(float), ny ) );
    if (mode >= 2) cutilSafeCall( cudaMallocPitch( (void**)&(d_original), &pitchBytesF1, nx*sizeof(float), ny ) );

    cutilSafeCall( cudaMemcpy2D(d_input, pitchBytesF1, input, nx*sizeof(float), nx*sizeof(float), ny, cudaMemcpyHostToDevice) );
    if (mode >= 2) cutilSafeCall( cudaMemcpy2D(d_original, pitchBytesF1, d_input, pitchBytesF1, nx*sizeof(float), ny, cudaMemcpyDeviceToDevice) );

    pitchF1 = pitchBytesF1/sizeof(float);

  } else if (nc == 3) {

    cutilSafeCall( cudaMallocPitch( (void**)&(d_input), &pitchBytesF3, nx*sizeof(float3), ny ) );
    cutilSafeCall( cudaMallocPitch( (void**)&(d_output), &pitchBytesF3, nx*sizeof(float3), ny ) );
    if (mode) cutilSafeCall( cudaMallocPitch( (void**)&(d_diffusivity), &pitchBytesF3, nx*sizeof(float3), ny ) );
    if (mode >= 2) cutilSafeCall( cudaMallocPitch( (void**)&(d_original), &pitchBytesF3, nx*sizeof(float3), ny ) );

    cutilSafeCall( cudaMemcpy2D(d_input, pitchBytesF3, input, nx*sizeof(float3), nx*sizeof(float3), ny, cudaMemcpyHostToDevice) );
    if (mode >= 2) cutilSafeCall( cudaMemcpy2D(d_original, pitchBytesF3, d_input, pitchBytesF3, nx*sizeof(float3), ny, cudaMemcpyDeviceToDevice) );

  }


  //Execution of the Diffusion Kernel

  if (mode == 0) {   // linear isotropic diffision
    if (nc == 1) {
      for (i=0;i<iterations;i++) {
        diffuse_linear_isotrop_shared<<<dimGrid,dimBlock>>>(d_input, d_output, timeStep, nx, ny, pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
    else if (nc == 3) {
      for (i=0;i<iterations;i++) {
        diffuse_linear_isotrop_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_output,timeStep,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
  }
  else if (mode == 1) {  // nonlinear isotropic diffusion
    if (nc == 1) {

      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        diffuse_nonlinear_isotrop_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,d_output,timeStep,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
    else if (nc == 3) {
      for (i=0;i<iterations;i++) {
    	  
    	  // Added 
//        compute_tv_diffusivity_separate_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);
    	  compute_tv_diffusivity_joined_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        diffuse_nonlinear_isotrop_shared<<<dimGrid,dimBlock>>>
          ((float3*)d_input,(float3*)d_diffusivity,(float3*)d_output,timeStep,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        temp = d_input;
        d_input = d_output;
        d_output = temp;
      }
    }
  }
  else if (mode == 2) {    // Jacobi-method
    if (nc == 1) {
      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        for (j=0;j<lagged_iterations;j++) {
          jacobi_shared<<<dimGrid,dimBlock>>> (d_output,d_input,d_original,
            d_diffusivity,weight,overrelaxation,nx,ny,pitchF1);

          cutilSafeCall( cudaThreadSynchronize() );

          temp = d_input;
          d_input = d_output;
          d_output = temp;
        }
      }
    }
    else if (nc == 3) {
      for (i=0;i<iterations;i++) {
        //--- this doesn't work with joint diffusivities ---
        //compute_tv_diffusivity_joined_shared<<<dimGrid,dimBlock>>>
        //		((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitch);
        compute_tv_diffusivity_separate_shared<<<dimGrid,dimBlock>>>
          ((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        for (j=0;j<lagged_iterations;j++) {
          jacobi_shared<<<dimGrid,dimBlock>>>
            ((float3*)d_output,(float3*)d_input,
            (float3*)d_original,(float3*)d_diffusivity,
            weight,overrelaxation,nx,ny,pitchBytesF3);

          cutilSafeCall( cudaThreadSynchronize() );

          temp = d_input;
          d_input = d_output;
          d_output = temp;
        }
      }
    }    
  }
  else if(mode == 3) {    // Successive Over Relaxation (Gauss-Seidel with extrapolation)
    if (nc == 1) {
      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_shared<<<dimGrid,dimBlock>>>(d_input,d_diffusivity,nx,ny,pitchF1);

        cutilSafeCall( cudaThreadSynchronize() );

        for(j=0;j<lagged_iterations;j++) {					
          sor_shared<<<dimGrid,dimBlock>>>(d_input,d_input,d_original,
            d_diffusivity,weight,overrelaxation,nx,ny,pitchF1, 0);

          cutilSafeCall( cudaThreadSynchronize() );

          sor_shared<<<dimGrid,dimBlock>>>(d_input,d_input,d_original,
            d_diffusivity,weight,overrelaxation,nx,ny,pitchF1, 1);

          cutilSafeCall( cudaThreadSynchronize() );
        }
      }
    }
    if (nc == 3) {
      for (i=0;i<iterations;i++) {
        compute_tv_diffusivity_separate_shared<<<dimGrid,dimBlock>>>((float3*)d_input,(float3*)d_diffusivity,nx,ny,pitchBytesF3);

        cutilSafeCall( cudaThreadSynchronize() );

        for (j=0;j<lagged_iterations;j++) {
          sor_shared<<<dimGrid,dimBlock>>>
            ((float3*)d_input,(float3*)d_input,
            (float3*)d_original,(float3*)d_diffusivity,
            weight,overrelaxation,nx,ny,pitchBytesF3, 0);

          cutilSafeCall( cudaThreadSynchronize() );

          sor_shared<<<dimGrid,dimBlock>>>
            ((float3*)d_input,(float3*)d_input,
            (float3*)d_original,(float3*)d_diffusivity,
            weight,overrelaxation,nx,ny,pitchBytesF3, 1);

          cutilSafeCall( cudaThreadSynchronize() );
        }
      }
    }
  }


  if (nc == 1) {
    if (mode == 3) cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float), d_input, pitchBytesF1, nx*sizeof(float), ny, cudaMemcpyDeviceToHost) );
    else cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float), d_output, pitchBytesF1, nx*sizeof(float), ny, cudaMemcpyDeviceToHost) );
  } else if (nc == 3) {
    if (mode == 3) cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float3), d_input, pitchBytesF3, nx*sizeof(float3), ny, cudaMemcpyDeviceToHost) );
    else cutilSafeCall( cudaMemcpy2D(output, nx*sizeof(float3), d_output, pitchBytesF3, nx*sizeof(float3), ny, cudaMemcpyDeviceToHost) );
  }


  // clean up
  if (d_original) cutilSafeCall( cudaFree(d_original) );
  if (d_diffusivity) cutilSafeCall( cudaFree(d_diffusivity) );
  if (d_output) cutilSafeCall( cudaFree(d_output) );
  if (d_input)  cutilSafeCall( cudaFree(d_input) );
}
