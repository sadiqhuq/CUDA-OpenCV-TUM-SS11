/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: sorflow
* file:    sorflow_gpu.cu
*
* 
\********* PLEASE ENTER YOUR CORRECT STUDENT NAME AND ID BELOW **************/
const char* studentName = "Ramakrishna Nanjundaiah, Sadiq Huq";
const int   studentID   = 3615831;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* sorflow_compute_motion_tensor_tex            - Done
* sorflow_linear_sor_shared                    - Done
* sorflow_update_robustifications_shared       - Done
* sorflow_nonlinear_sor_shared                 - Done
* add_flow_fields                              - Done
* bilinear_backward_warping_tex                - Done
* sorflow_update_robustifications_warp_shared  - Done
* sorflow_update_righthandside_shared          - Done
* sorflow_nonlinear_warp_sor_shared            - Done
* sorflow_gpu_nonlinear_warp                   - Done
*
\****************************************************************************/


#include "sorflow_gpu.cuh"
#include <stdio.h>
#include "sorflow.h"
#include "resample_gpu.cuh"
#include <cutil_math.h>



cudaChannelFormatDesc sorflow_float_tex = cudaCreateChannelDesc<float>();

texture<float, 2, cudaReadModeElementType> tex_sorflow_I1;
texture<float, 2, cudaReadModeElementType> tex_sorflow_I2;

#define IMAGE_FILTER_METHOD cudaFilterModeLinear;
#define TEXTURE_OFFSET 0.5f

#define SF_BW 16
#define SF_BH 16



const char* getStudentName() { return studentName; };
int         getStudentID()   { return studentID; };
bool        checkStudentNameAndID() { return strcmp(studentName, "John Doe") != 0 && studentID != 1234567; }; 



//############################## Helpers ####################################


__global__ void sorflow_hv_to_rgb_kernel
(
	float2 *device_u,
	float3 *device_rgb,
	float  pmin,
	float  pmax,
	float  ptmin,
	int    nx,
	int    ny,
	int    pitchf2,
	int    pitchf3
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float temp;
	float2 u;
	float3 rgb;


	float Pi;
	float amp;
	float phi;
	float alpha, beta;

	if(x < nx && y < ny)
	{
		u = device_u[y*pitchf2+x];
	}


  /* set pi */
  Pi = 2.0f * acos(0.0f);

  /* determine amplitude and phase (cut amp at 1) */
  amp = (sqrtf(u.x*u.x + u.y*u.y) - pmin)/(pmax - pmin);
  if (amp > 1.0f) amp = 1.0f;
  if (u.x == 0.0f)
    if (u.y >= 0.0f) phi = 0.5f * Pi;
    else phi = 1.5f * Pi;
  else if (u.x > 0.0f)
    if (u.y >= 0.0f) phi = atanf(u.y/u.x);
    else phi = 2.0f * Pi + atanf(u.y/u.x);
  else phi = Pi + atanf (u.y/u.x);

  phi = phi / 2.0f;

  // interpolation between red (0) and blue (0.25 * Pi)
  if ((phi >= 0.0f) && (phi < 0.125f * Pi))
  {
    beta  = phi / (0.125f * Pi);
    alpha = 1.0f - beta;
    rgb.x = amp * (alpha+ beta);
    rgb.y = 0.0f;
    rgb.z = amp * beta;
  }
  else if ((phi >= 0.125 * Pi) && (phi < 0.25 * Pi))
  {
    beta  = (phi-0.125 * Pi) / (0.125 * Pi);
    alpha = 1.0 - beta;
    rgb.x = amp * (alpha + beta *  0.25f);
    rgb.y = amp * (beta *  0.25f);
    rgb.z = amp * (alpha + beta);
  }
  // interpolation between blue (0.25 * Pi) and green (0.5 * Pi)
  else if ((phi >= 0.25 * Pi) && (phi < 0.375 * Pi))
  {
    beta  = (phi - 0.25 * Pi) / (0.125 * Pi);
    alpha = 1.0 - beta;
    rgb.x = amp * (alpha *  0.25f);
    rgb.y = amp * (alpha *  0.25f + beta);
    rgb.z = amp * (alpha+ beta);
  }
  else if ((phi >= 0.375 * Pi) && (phi < 0.5 * Pi))
  {
    beta  = (phi - 0.375 * Pi) / (0.125 * Pi);
    alpha = 1.0 - beta;
    rgb.x = 0.0f;
    rgb.y = amp * (alpha+ beta);
    rgb.z = amp * (alpha);
  }
  // interpolation between green (0.5 * Pi) and yellow (0.75 * Pi)
  else if ((phi >= 0.5 * Pi) && (phi < 0.75 * Pi))
  {
    beta  = (phi - 0.5 * Pi) / (0.25 * Pi);
    alpha = 1.0 - beta;
    rgb.x = amp * (beta);
    rgb.y = amp * (alpha+ beta);
    rgb.z = 0.0f;
  }
  // interpolation between yellow (0.75 * Pi) and red (Pi)
  else if ((phi >= 0.75 * Pi) && (phi <= Pi))
  {
    beta  = (phi - 0.75 * Pi) / (0.25 * Pi);
    alpha = 1.0 - beta;
    rgb.x = amp * (alpha+ beta);
    rgb.y = amp * (alpha);
    rgb.z = 0.0f;
  }
  else
  {
  	rgb.x = rgb.y = rgb.z = 0.5f;
  }

	temp = atan2f(-u.x,-u.y)*0.954929659f; // 6/(2*PI)
	if(temp < 0) temp += 6.0f;

  rgb.x = (rgb.x>=1.0f)*1.0f+((rgb.x<1.0f)&&(rgb.x>0.0f))*rgb.x;
  rgb.y = (rgb.y>=1.0f)*1.0f+((rgb.y<1.0f)&&(rgb.y>0.0f))*rgb.y;
  rgb.z = (rgb.z>=1.0f)*1.0f+((rgb.z<1.0f)&&(rgb.z>0.0f))*rgb.z;

  rgb.x *= 255.0f; rgb.y *= 255.0f; rgb.z *= 255.0f;

	if(x < nx && y < ny)
	{
		device_rgb[y*pitchf3+x] = rgb;
	}
}

void sorflow_hv_to_rgb
(
	float2 *device_u,
	float3 *device_rgb,
	float  pmin,
	float  pmax,
	float  ptmin,
	int    nx,
	int    ny,
	int    pitchf2,
	int    pitchf3
)
{
	int ngx = (nx%SF_BW) ? ((nx/SF_BW)+1) : (nx/SF_BW);
	int ngy = (ny%SF_BH) ? ((ny/SF_BH)+1) : (ny/SF_BH);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);

	sorflow_hv_to_rgb_kernel<<<dimGrid,dimBlock>>>
			(device_u,device_rgb,pmin,pmax,ptmin,nx,ny,pitchf2,pitchf3);
	catchkernel;
}

void resize_fields
(
	int    nx,
	int    ny,
	int    *pitchf1,
	int    *pitchf2,
	int    *pitchf3,
	float  **I1_g,
	float  **I2_g,
	float2 **u_g,
	float2 **du_g,
	float  **I1_resampled_g,
	float  **I2_resampled_g,
	float  **I2_resampled_warped_g,
	float3 **Aspatial_g,
	float3 **Atemporal_g,
	float2 **b_g,
	float2 **penalty_g,
	float  **output_g
)
{
	if(*I1_g)                  cutilSafeCall( cudaFree(*I1_g));
	if(*I2_g)                  cutilSafeCall( cudaFree(*I2_g));
	if(*u_g)                   cutilSafeCall( cudaFree(*u_g));
	if(*du_g)                  cutilSafeCall( cudaFree(*du_g));
	if(*I1_resampled_g)        cutilSafeCall( cudaFree(*I1_resampled_g));
	if(*I2_resampled_g)        cutilSafeCall( cudaFree(*I2_resampled_g));
	if(*I2_resampled_warped_g) cutilSafeCall( cudaFree(*I2_resampled_warped_g));
	if(*Aspatial_g)            cutilSafeCall( cudaFree(*Aspatial_g));
	if(*Atemporal_g)           cutilSafeCall( cudaFree(*Atemporal_g));
	if(*b_g)                   cutilSafeCall( cudaFree(*b_g));
	if(*penalty_g)             cutilSafeCall( cudaFree(*penalty_g));
	if(*output_g)              cutilSafeCall( cudaFree(*output_g));


	cuda_malloc2D((void**)I1_g,nx,ny,1,sizeof(float),pitchf1);
	cuda_malloc2D((void**)I2_g,nx,ny,1,sizeof(float),pitchf1);
	cuda_malloc2D((void**)u_g,nx,ny,1,sizeof(float2),pitchf2);
	cuda_malloc2D((void**)du_g,nx,ny,1,sizeof(float2),pitchf2);
	cuda_malloc2D((void**)I1_resampled_g,nx,ny,1,sizeof(float),pitchf1);
	cuda_malloc2D((void**)I2_resampled_g,nx,ny,1,sizeof(float),pitchf1);
	cuda_malloc2D((void**)I2_resampled_warped_g,nx,ny,1,sizeof(float),pitchf1);
	cuda_malloc2D((void**)Aspatial_g,nx,ny,1,sizeof(float3),pitchf3);
	cuda_malloc2D((void**)Atemporal_g,nx,ny,1,sizeof(float3),pitchf3);
	cuda_malloc2D((void**)b_g,nx,ny,1,sizeof(float2),pitchf2);
	cuda_malloc2D((void**)penalty_g,nx,ny,1,sizeof(float2),pitchf2);
	cuda_malloc2D((void**)output_g,nx,ny,1,sizeof(float3),pitchf3);
}

void delete_fields
(
	float  *I1_g,
	float  *I2_g,
	float2 *u_g,
	float2 *du_g,
	float  *I1_resampled_g,
	float  *I2_resampled_g,
	float  *I2_resampled_warped_g,
	float3 *Aspatial_g,
	float3 *Atemporal_g,
	float2 *b_g,
	float2 *penalty_g,
	float  *output_g
)
{
	if(I1_g)                  cutilSafeCall( cudaFree(I1_g));
	if(I2_g)                  cutilSafeCall( cudaFree(I2_g));
	if(u_g)                   cutilSafeCall( cudaFree(u_g));
	if(du_g)                  cutilSafeCall( cudaFree(du_g));
	if(I1_resampled_g)        cutilSafeCall( cudaFree(I1_resampled_g));
	if(I2_resampled_g)        cutilSafeCall( cudaFree(I2_resampled_g));
	if(I2_resampled_warped_g) cutilSafeCall( cudaFree(I2_resampled_warped_g));
	if(Aspatial_g)            cutilSafeCall( cudaFree(Aspatial_g));
	if(Atemporal_g)           cutilSafeCall( cudaFree(Atemporal_g));
	if(b_g)                   cutilSafeCall( cudaFree(b_g));
	if(penalty_g)             cutilSafeCall( cudaFree(penalty_g));
	if(output_g)              cutilSafeCall( cudaFree(output_g));
}























//###################### Functions for Linear Regularization #################

__global__ void sorflow_compute_motion_tensor_tex
(
	float3 *Aspatial_g,
	float3 *Atemporal_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	int    pitchf3
)
{

	// ### implement me ###

  // compute the motion tensor

  // Aspatial = products of spatial derivatives as defined in the Euler-Lagrange equations
  // Atemporal = products of spatial and temporal derivatives as defined in the Euler-Lagrange equations
  // do that by using the image textures

	//Thread indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//PitchBytes
	const int pitchBytes_3 = pitchf3 * sizeof(float3);

	//Offset in texture 
	const float xx = (float)x+TEXTURE_OFFSET;
	const float yy = (float)y+TEXTURE_OFFSET;

	//Derivatives of I1 using image texture
	const float I1_dx = (tex2D(tex_sorflow_I1,xx+1,yy) - tex2D(tex_sorflow_I1,xx-1,yy)) / (2*hx);
	const float I1_dy = (tex2D(tex_sorflow_I1,xx,yy+1) - tex2D(tex_sorflow_I1,xx,yy-1)) / (2*hy);

	//Derivatives of I2 using image texture
	const float I2_dx = (tex2D(tex_sorflow_I2,xx+1,yy) - tex2D(tex_sorflow_I2,xx-1,yy)) / (2*hx);
	const float I2_dy = (tex2D(tex_sorflow_I2,xx,yy+1) - tex2D(tex_sorflow_I2,xx,yy-1)) / (2*hy);

	//Mean values at pixels
	const float Ix = 0.5f * (I1_dx + I2_dx);
	const float Iy = 0.5f * (I1_dy + I2_dy);

	//It (time derivative)- Backward difference from assignment sheet
	const float It = tex2D(tex_sorflow_I2,xx,yy) - tex2D(tex_sorflow_I1,xx,yy);	


	//Aspatial_g storage ==> [Ix*Ix, Iy*Ix , Iy*Iy]
	//Aspatial_g storege ==> [Ix*It, Iy*It , It*It]
	if ( x < nx && y < ny) {
		*((float3*) (((char*)Aspatial_g) + y*pitchBytes_3) + x) = make_float3 (Ix*Ix,Iy*Ix,Iy*Iy);
		*((float3*) (((char*)Atemporal_g) + y*pitchBytes_3) + x) = make_float3 (Ix*It,Iy*It,It*It);
	}		

}


__global__ void sorflow_linear_sor_shared
(
	const float3 *Aspatial_g,
	const float3 *Atemporal_g,
	float2 *u_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  lambda,
	float  relaxation,
	int    red,
	int    pitchf2,
	int    pitchf3
)
{
	
  // ### implement me ###

  // solve the linear Euler-Lagrange equations resulting from quatratic penalization using shared memory 

	//Thread indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int tx = threadIdx.x+1;
	const int ty = threadIdx.y+1;

	//Pitch bytes
	const int pitchbytes_2 = pitchf2 * sizeof(float2);
	const int pitchbytes_3 = pitchf3 * sizeof(float3);

	// loading and boundary conditions
	__shared__ float2 u[SF_BW+2][SF_BH+2];
	const char* image = (char*)u_g + y*pitchbytes_2  + x*sizeof(float2);

	//Boundary conditions
	if (x < nx && y < ny) {
		u[tx][ty] = *( (float2*)image );

		if (x == 0)
			u[threadIdx.x][ty] = u[tx][ty];
		else if (x == nx-1)
			u[tx+1][ty] = u[tx][ty];
		else {
			if (threadIdx.x == 0) u[0][ty] = *(((float2*)image)-1);
			else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *(((float2*)image)+1);
		}

		if (y == 0)
			u[tx][0] = u[tx][ty];
		else if (y == ny-1)
			u[tx][ty+1] = u[tx][ty];
		else {
			if (threadIdx.y == 0) u[tx][0] = *( (float2*)(image-pitchbytes_2) );
			else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float2*)(image+pitchbytes_2) );
		}
	}


	__syncthreads();

	if((x+y)%2 == red){

		// ### implement me ###
		//{a = alpha, b = beta, c = gamma, d = delta} = 1 for linear!!!
		if (x < nx && y < ny){
			float a = 1.f;
			float b = 1.f;
			float c = 1.f;
			float d = 1.f;


			//Setting values at boundaries
			if (x >= nx) a = 0;
			if (x <= 0)  b = 0;
			if (y >= ny) c = 0;
			if (y <= 0)  d = 0;

			//term_1 = value from the four neighbouring points
			float2 term_1 =  make_float2 (
					( a * u[tx-1][ty].x + b * u[tx+1][ty].x + c * u[tx][ty-1].x + d * u[tx][ty+1].x),
					( a * u[tx-1][ty].y + b * u[tx+1][ty].y + c * u[tx][ty-1].y + d * u[tx][ty+1].y) );

			//Spatial derivatives
			const float3 s = *((float3*)(((char*)Aspatial_g) + y*pitchbytes_3)+ x);

			//Temporal derivaties
			const float3 t = *((float3*)(((char*)Atemporal_g) + y*pitchbytes_3)+ x);

			//Implemetation of iteration
			float u1 = ( (lambda * term_1.x) - (s.y * u[tx][ty].y + t.x))/ (s.x + lambda * (a+b+c+d));
			float u2 = ( (lambda * term_1.y) - (s.y * u[tx][ty].x + t.y))/ (s.z + lambda * (a+b+c+d));

			float temp1 = (1-relaxation) * u[tx][ty].x + relaxation * u1;
			float temp2 = (1-relaxation) * u[tx][ty].y + relaxation * u2;

			//copy output to u_g
			*((float2*) (((char*)u_g) + y*pitchbytes_2) + x) = make_float2(temp1,temp2);	

		}	  	  
	}	

}




void sorflow_gpu_linear
(
	const float  *I1_g,
	const float  *I2_g,
	float2 *u_g,
	float3 *Aspatial_g,
	float3 *Atemporal_g,
	int    nx,
	int    ny,
	int    pitchf1,
	int    pitchf2,
	int    pitchf3,
	float  lambda,
	int    iterations,
	float  relaxation,
	float  data_epsilon,
	float  diff_epsilon
)
{

	int i;
	float hx = 1.0f;
	float hy = 1.0f;

	int ngx = (nx%SF_BW) ? ((nx/SF_BW)+1) : (nx/SF_BW);
	int ngy = (ny%SF_BW) ? ((ny/SF_BW)+1) : (ny/SF_BW);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);


  bind_textures(I1_g, I2_g, nx, ny, pitchf1);

	cutilSafeCall( cudaMemset(u_g,0,pitchf2*ny*sizeof(float2)));


	sorflow_compute_motion_tensor_tex<<<dimGrid,dimBlock>>>
	    (Aspatial_g,Atemporal_g,nx,ny,hx,hy,pitchf3);
	catchkernel;


	for(i=0;i<iterations;i++)
	{
			sorflow_linear_sor_shared<<<dimGrid,dimBlock>>>
					(Aspatial_g,Atemporal_g,u_g,nx,ny,hx,hy,lambda,relaxation,0,pitchf2,pitchf3);
			catchkernel;
			sorflow_linear_sor_shared<<<dimGrid,dimBlock>>>
					(Aspatial_g,Atemporal_g,u_g,nx,ny,hx,hy,lambda,relaxation,1,pitchf2,pitchf3);
			catchkernel;
	}
}









//###################### Functions for Nonlinear Regularization ################



__global__ void sorflow_update_robustifications_shared
(
	const float3 *Aspatial_g,
	const float3 *Atemporal_g,
	const float2 *u_g,
	float2 *penalty_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  data_epsilon,
	float  diff_epsilon,
	int    pitchf2,
	int    pitchf3
)
{

	// ### implement me ###

  // update the penalty functions
	//Thread indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int tx = threadIdx.x+1;
	const int ty = threadIdx.y+1;

	//Pitch bytes
	const int pitchbytes_2 = pitchf2 * sizeof(float2);
	const int pitchbytes_3 = pitchf3 * sizeof(float3);

	//Loading  shared memory
	__shared__ float2 u[SF_BW+2][SF_BH+2];
	const char* image = (char*)u_g + y*pitchbytes_2  + x*sizeof(float2);

	// loading and boundary conditions
	if (x < nx && y < ny) {
		u[tx][ty] = *( (float2*)image );

		if (x == 0)
			u[threadIdx.x][ty] = u[tx][ty];
		else if (x == nx-1)
			u[tx+1][ty] = u[tx][ty];
		else {
			if (threadIdx.x == 0) u[0][ty] = *(((float2*)image)-1);
			else if (threadIdx.x == blockDim.x-1) u[tx+1][ty] = *(((float2*)image)+1);
		}

		if (y == 0)
			u[tx][0] = u[tx][ty];
		else if (y == ny-1)
			u[tx][ty+1] = u[tx][ty];
		else {
			if (threadIdx.y == 0) u[tx][0] = *( (float2*)(image-pitchbytes_2) );
			else if (threadIdx.y == blockDim.y-1) u[tx][ty+1] = *( (float2*)(image+pitchbytes_2) );
		}
	}

	__syncthreads();


	if (x < nx && y < ny){

		//Spatial derivatives
		const float3 s = *((float3*)(((char*)Aspatial_g) + y*pitchbytes_3)+ x);

		//Temporal derivaties
		const float3 t = *((float3*)(((char*)Atemporal_g) + y*pitchbytes_3)+ x);


		//Euler _ Lagrange equation, data term
		const float phi_data = 1.f/(2.f * sqrt(data_epsilon + s.x * u[tx][ty].x *u[tx][ty].x+
				s.z * u[tx][ty].y *u[tx][ty].y + t.z + 
				2.f * s.y * u[tx][ty].x *u[tx][ty].y+
				2.f * t.y * u[tx][ty].y+
				2.f * t.x * u[tx][ty].x));


		//Derivatives of u1 and u2 - central
		const float u1_dx = 0.5f*(u[tx+1][ty].x-u[tx-1][ty].x);
		const float u1_dy = 0.5f*(u[tx][ty+1].x-u[tx][ty-1].x);

		const float u2_dx = 0.5f*(u[tx+1][ty].y-u[tx-1][ty].y);
		const float u2_dy = 0.5f*(u[tx][ty+1].y-u[tx][ty-1].y);

		//Euler _ Lagrange equation, data term
		const float phi_diff =  1.f/(2.f * sqrt(diff_epsilon + u1_dx*u1_dx+ u1_dy*u1_dy
				+	u2_dx*u2_dx + u2_dy *u2_dy ));

		*((float2*) (((char*)penalty_g) + y*pitchbytes_2) + x) = make_float2(phi_data,phi_diff);
	}	  	  

}




__global__ void sorflow_nonlinear_sor_shared
(
	const float3 *Aspatial_g,
	const float3 *Atemporal_g,
	const float2 *penalty_g,
	float2 *u_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  lambda,
	float  relaxation,
	int    red,
	int    pitchf2,
	int    pitchf3
)
{

  // ### implement me ### 
	const int pitchbytes_2 = pitchf2 * sizeof(float2);
	const int pitchbytes_3 = pitchf3 * sizeof(float3);

	//Thread Indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int tx = threadIdx.x+1;
	const int ty = threadIdx.y+1;

	//Image and penalty
	const char* image = (char*)u_g + y*pitchbytes_2  + x*sizeof(float2);
	const char* p = (char*)penalty_g + y*pitchbytes_2  + x*sizeof(float2);

	//Loading into shared mmory
	__shared__ float2 u[SF_BW+2][SF_BW+2];//for image
	__shared__ float2 g[SF_BW+2][SF_BW+2];//for p


	// loading and boundary conditions
	if (x < nx && y < ny) {
		u[tx][ty] = *((float2*)image);
		g[tx][ty] = *((float2*)p);

		if (x == 0) {
			u[threadIdx.x][ty] = u[tx][ty];
			g[threadIdx.x][ty] = g[tx][ty];
		}
		else if (x == nx-1) {
			u[tx+1][ty] = u[tx][ty];
			g[tx+1][ty] = g[tx][ty];
		}
		else {
			if (threadIdx.x == 0) {
				u[0][ty] = *(((float2*)image)-1);
				g[0][ty] = *(((float2*)p)-1);
			}
			else if (threadIdx.x == blockDim.x-1) {
				u[tx+1][ty] = *(((float2*)image)+1);
				g[tx+1][ty] = *(((float2*)p)+1);
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
				u[tx][0] = *((float2*)(image-pitchbytes_2));
				g[tx][0] = *((float2*)(p-pitchbytes_2));
			}
			else if (threadIdx.y == blockDim.y-1) {
				u[tx][ty+1] = *((float2*)(image+pitchbytes_2));
				g[tx][ty+1] = *((float2*)(p+pitchbytes_2));
			}
		}

	}

	__syncthreads();


	if (x < nx && y < ny){

		//Spatial derivatives
		const float3 s = *((float3*)(((char*)Aspatial_g) + y*pitchbytes_3)+ x);

		//Temporal derivaties
		const float3 t = *((float3*)(((char*)Atemporal_g) + y*pitchbytes_3)+ x);

		//Calculation of a,b,c,d - only y field of p
		float a = 0.5 * (g[tx+1][ty].y+g[tx][ty].y);
		float b = 0.5 * (g[tx-1][ty].y+g[tx][ty].y);
		float c = 0.5 * (g[tx][ty].y+g[tx][ty+1].y);
		float d = 0.5 * (g[tx][ty].y+g[tx][ty-1].y);


		//Setting values at boundaries
		if (x >= nx) a = 0;
		if (x <= 0)  b = 0;
		if (y >= ny) c = 0;
		if (y <= 0)  d = 0;

		float2 term_1 =  make_float2 (
				( a * u[tx+1][ty].x + b * u[tx-1][ty].x + c * u[tx][ty+1].x + d * u[tx][ty-1].x),
				( a * u[tx+1][ty].y + b * u[tx-1][ty].y + c * u[tx][ty+1].y + d * u[tx][ty-1].y) );

		//Implemetation of iteration
		float u1 = ( (lambda * term_1.x) - (s.y * u[tx][ty].y*g[tx][ty].x  + t.x*g[tx][ty].x ))/ (s.x*g[tx][ty].x + lambda * (a+b+c+d));
		float u2 = ( (lambda * term_1.y) - (s.y * u[tx][ty].x*g[tx][ty].x  + t.y*g[tx][ty].x ))/ (s.z*g[tx][ty].x + lambda * (a+b+c+d));

		float temp1 = (1-relaxation) * u[tx][ty].x + relaxation * u1;
		float temp2 = (1-relaxation) * u[tx][ty].y + relaxation * u2;

		*((float2*) (((char*)u_g) + y*pitchbytes_2) + x) = make_float2(temp1,temp2);

	}

}


void sorflow_gpu_nonlinear
(
	const float  *I1_g,
	const float  *I2_g,
	float2 *u_g,
	float3 *Aspatial_g,
	float3 *Atemporal_g,
	float2 *penalty_g,
	int    nx,
	int    ny,
	int    pitchf1,
	int    pitchf2,
	int    pitchf3,
	float  lambda,
	int    outer_iterations,
	int    inner_iterations,
	float  relaxation,
	float  data_epsilon,
	float  diff_epsilon
)
{

	int i,j;
	float hx = 1.0f;
	float hy = 1.0f;

	int ngx = (nx%SF_BW) ? ((nx/SF_BW)+1) : (nx/SF_BW);
	int ngy = (ny%SF_BW) ? ((ny/SF_BW)+1) : (ny/SF_BW);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);


  bind_textures(I1_g, I2_g, nx, ny, pitchf1);

	cutilSafeCall( cudaMemset(u_g,0,pitchf2*ny*sizeof(float2)));


	sorflow_compute_motion_tensor_tex<<<dimGrid,dimBlock>>>
	    (Aspatial_g,Atemporal_g,nx,ny,hx,hy,pitchf3);
	catchkernel;


	for(i=0;i<outer_iterations;i++)
	{

		sorflow_update_robustifications_shared<<<dimGrid,dimBlock>>>
				(Aspatial_g,Atemporal_g,u_g,penalty_g,nx,ny,hx,hy,
						data_epsilon,diff_epsilon,pitchf2,pitchf3);
		catchkernel;

		for(j=0;j<inner_iterations;j++)
		{
			sorflow_nonlinear_sor_shared<<<dimGrid,dimBlock>>>
					(Aspatial_g,Atemporal_g,penalty_g,u_g,nx,ny,hx,hy,lambda,relaxation,0,pitchf2,pitchf3);
			catchkernel;
			sorflow_nonlinear_sor_shared<<<dimGrid,dimBlock>>>
					(Aspatial_g,Atemporal_g,penalty_g,u_g,nx,ny,hx,hy,lambda,relaxation,1,pitchf2,pitchf3);
			catchkernel;

		}
	}

}




























//###################### Functions for Nonlinear Regularization ##############
//###################### with warping                         ################


__global__ void add_flow_fields
(
	const float2 *u_g,
	float2 *u0_g,
	int    nx,
	int    ny,
	int    pitchf2
)
{

  // ### implement me ### 
	//pitch bytes
	const int pitchbytes_2 = pitchf2 * sizeof(float2);

	//Thread indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Adding fields
	if (x < nx && y< ny){

		*((float2*) (((char*)u0_g) + y*pitchbytes_2) + x) += *((float2*) (((char*)u_g) + y*pitchbytes_2) + x);
	}

}

__global__ void bilinear_backward_warping_tex
(
	const float2 *u_g,
	float *f2_warped_g,
	int   nx,
	int   ny,
	float hx,
	float hy,
	int   pitchf1,
	int   pitchf2
)
{

  // ### implement me ###

  // note: 
  // nx, ny is the grid size
  // hx, hy is the size of a single grid cell this method is working on

	//pitch bytes
	const int pitchbytes_2 = pitchf2 * sizeof(float2);

	//Thread indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < nx && y< ny){

		float2 u = *((float2*) (((char*)u_g) + y*pitchbytes_2) + x);
		float2 h = make_float2(hx,hy);
		float2 xy = make_float2 ( (float)x , (float)y );
		float2 tex = xy + u/h +TEXTURE_OFFSET;
		f2_warped_g[x + y * pitchf1] = tex2D(tex_sorflow_I2,tex.x,tex.y);		

	}
}



__global__ void sorflow_update_robustifications_warp_shared
(
	const float3 *Aspatial_g,
	const float3 *Atemporal_g,
	const float2 *u_g,
	const float2 *du_g,
	float2 *penalty_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  data_epsilon,
	float  diff_epsilon,
	int    pitchf2,
	int    pitchf3
)
{

  // ### implement me ###
  
  // update the penalty functions

	const int pitchbytes_2 = pitchf2 * sizeof(float2);
	const int pitchbytes_3 = pitchf3 * sizeof(float3);

	//Thread Indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int tx = threadIdx.x+1;
	const int ty = threadIdx.y+1;

	//Image and penalty
	const char* image = (char*)u_g + y*pitchbytes_2  + x*sizeof(float2);
	const char* d_image = (char*)du_g + y*pitchbytes_2  + x*sizeof(float2);

	//Loading into shared mmory
	__shared__ float2 u[SF_BW+2][SF_BW+2];//for known
	__shared__ float2 g[SF_BW+2][SF_BW+2];//for unknown 


	// load data into shared memory
	if (x < nx && y < ny) {
		u[tx][ty] = *((float2*)image);
		g[tx][ty] = *((float2*)d_image);

		if (x == 0) {
			u[threadIdx.x][ty] = u[tx][ty];
			g[threadIdx.x][ty] = g[tx][ty];
		}
		else if (x == nx-1) {
			u[tx+1][ty] = u[tx][ty];
			g[tx+1][ty] = g[tx][ty];
		}
		else {
			if (threadIdx.x == 0) {
				u[0][ty] = *(((float2*)image)-1);
				g[0][ty] = *(((float2*)d_image)-1);
			}
			else if (threadIdx.x == blockDim.x-1) {
				u[tx+1][ty] = *(((float2*)image)+1);
				g[tx+1][ty] = *(((float2*)d_image)+1);
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
				u[tx][0] = *((float2*)(image-pitchbytes_2));
				g[tx][0] = *((float2*)(d_image-pitchbytes_2));
			}
			else if (threadIdx.y == blockDim.y-1) {
				u[tx][ty+1] = *((float2*)(image+pitchbytes_2));
				g[tx][ty+1] = *((float2*)(d_image+pitchbytes_2));
			}
		}

	}

	__syncthreads();

	if (x < nx && y < ny){

		//Spatial derivatives
		const float3 s = *((float3*)(((char*)Aspatial_g) + y*pitchbytes_3)+ x);

		//Temporal derivaties
		const float3 t = *((float3*)(((char*)Atemporal_g) + y*pitchbytes_3)+ x);
		//Data term
		const float phi_data = 1.f/(2.f * sqrt(data_epsilon + s.x * g[tx][ty].x *g[tx][ty].x+
				s.z * g[tx][ty].y *g[tx][ty].y + t.z + 
				2.f * s.y * g[tx][ty].x *g[tx][ty].y+
				2.f * t.y * g[tx][ty].y+
				2.f * t.x * g[tx][ty].x));

		//Derivatives  - central
		const float u1_dx = 0.5f*( u[tx+1][ty].x + g[tx+1][ty].x - u[tx-1][ty].x - g[tx-1][ty].x)/hx;
		const float u1_dy = 0.5f*( u[tx][ty+1].x + g[tx][ty+1].x - u[tx][ty-1].x - g[tx][ty-1].x)/hy;

		const float u2_dx = 0.5f*( u[tx+1][ty].y + g[tx+1][ty].y - u[tx-1][ty].y - g[tx-1][ty].y)/hx;
		const float u2_dy = 0.5f*( u[tx][ty+1].y + g[tx][ty+1].y - u[tx][ty-1].y - g[tx][ty-1].y)/hy;

		//Diff term
		const float phi_diff =  1.f/(2.f * sqrt(diff_epsilon + u1_dx*u1_dx+ u1_dy*u1_dy
				+	u2_dx*u2_dx + u2_dy *u2_dy ));

		*((float2*) (((char*)penalty_g) + y*pitchbytes_2) + x) = make_float2(phi_data,phi_diff);

	}
}



// updates b, which is the right hand side of the system of equations
__global__ void sorflow_update_righthandside_shared
(
	const float2 *u_g,
	const float2 *penalty_g,
	const float3 *Atemporal_g,
	float2 *b_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  lambda,
	int    pitchf2,
	int    pitchf3
)
{

  // ### implement me ### 
	//Thread Indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int tx = threadIdx.x+1;
	const int ty = threadIdx.y+1;

	//Pitch bytes
	const int pitchbytes_2 = pitchf2 * sizeof(float2);
	const int pitchbytes_3 = pitchf3 * sizeof(float3);

	const char* image = (char*)u_g + y*pitchbytes_2 + x*sizeof(float2);
	const char* penalty = (char*)penalty_g + y*pitchbytes_2 + x*sizeof(float2);

	//Shared Memory
	__shared__ float2 u[SF_BW+2][SF_BH+2];
	__shared__ float2 g[SF_BW+2][SF_BH+2];

	// loading and boundary conditions
	if (x < nx && y < ny) {
		u[tx][ty] = *( (float2*)image );
		g[tx][ty] =  *( (float2*)penalty );

		if (x == 0) {
			u[0][ty] = u[tx][ty];
			g[0][ty] = g[tx][ty];
		}
		else if (x == nx-1) {
			u[tx+1][ty] = u[tx][ty];
			g[tx+1][ty] = g[tx][ty];
		}
		else {
			if (threadIdx.x == 0) {
				u[0][ty] = *( ((float2*)image)-1 );
				g[0][ty] = *( ((float2*)penalty)-1 );
			}
			else if (threadIdx.x == blockDim.x-1) {
				u[tx+1][ty] = *( ((float2*)image)+1 );
				g[tx+1][ty] = *( ((float2*)penalty)+1 );
			}
		}

		if (y == 0)  {
			u[tx][0] = u[tx][ty];
			g[tx][0] = g[tx][ty];
		}
		else if (y == ny-1) {
			u[tx][ty+1] = u[tx][ty];
			g[tx][ty+1] = g[tx][ty];
		}
		else {
			if (threadIdx.y == 0) {
				u[tx][0] = *( (float2*)(image-pitchbytes_2) );
				g[tx][0] = *( (float2*)(penalty-pitchbytes_2) );
			}
			else if (threadIdx.y == blockDim.y-1) {
				u[tx][ty+1] = *( (float2*)(image+pitchbytes_2) );
				g[tx][ty+1] = *( (float2*)(penalty+pitchbytes_2) );
			}
		}
	}
	__syncthreads();	

	if(x < nx && y < ny)
	{
		float3 t = *((float3*)(((char*)Atemporal_g) + y*pitchbytes_3)+ x);		


		float a = 0.5 * (g[tx+1][ty].y+g[tx][ty].y);
		float b = 0.5 * (g[tx-1][ty].y+g[tx][ty].y);
		float c = 0.5 * (g[tx][ty].y+g[tx][ty+1].y);
		float d = 0.5 * (g[tx][ty].y+g[tx][ty-1].y);


		//Setting values at boundaries
		if (x >= nx) a = 0;
		if (x <= 0)  b = 0;
		if (y >= ny) c = 0;
		if (y <= 0)  d = 0;


		const float2 div = 
				((a * u[tx+1][ty]) / (hx * hx)) +
				((b *  u[tx-1][ty]) / (hx * hx)) +
				((c * u[tx][ty+1]) / (hy * hy)) +
				((d * u[tx][ty-1]) / (hy * hy)) -

				(
						((a + b) * u[tx][ty] / (hx * hx)) +
						((c + d) * u[tx][ty] / (hy * hy))
				);


		*((float2*)(((char*)b_g) + y*pitchbytes_2)+ x) = make_float2(
				-g[tx][ty].x * t.x + lambda * div.x,
				-g[tx][ty].x * t.y + lambda * div.y );		

	}
}


// updates the increments du_g for each level
__global__ void sorflow_nonlinear_warp_sor_shared
(
	const float3 *Aspatial_g,
	const float2 *b_g,
	const float2 *penalty_g,
	float2 *du_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	float  lambda,
	float  relaxation,
	int    red,
	int    pitchf2,
	int    pitchf3
)
{

  // ### implement me ### 

	const int pitchbytes_2 = pitchf2 * sizeof(float2);
	const int pitchbytes_3 = pitchf3 * sizeof(float3);

	//Thread Indices
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int tx = threadIdx.x+1;
	const int ty = threadIdx.y+1;

	//Image and penalty
	const char* du = (char*)du_g + y*pitchbytes_2 + x*sizeof(float2);
	const char* penalty = (char*)penalty_g + y*pitchbytes_2 + x*sizeof(float2);
	
	//Loading into shared mmory
	__shared__ float2 u[SF_BW+2][SF_BW+2];//for known
	__shared__ float2 g[SF_BW+2][SF_BW+2];//for unknown 


	if (x < nx && y < ny) {

		u[tx][ty] = *( (float2*)du );
		g[tx][ty] = *( (float2*)penalty );

		if (x == 0) {
			u[0][ty] = u[tx][ty];
			g[0][ty] = g[tx][ty];
		}
		else if (x == nx-1) {
			u[tx+1][ty] =  u[tx][ty];
			g[tx+1][ty] = g[tx][ty];
		}
		else {
			if (threadIdx.x == 0) {
				u[0][ty] = *( ((float2*)du)-1 );
				g[0][ty] = *( ((float2*)penalty)-1 );
			}
			else if (threadIdx.x == blockDim.x-1) {
				u[tx+1][ty] = *( ((float2*)du)+1 );
				g[tx+1][ty] = *( ((float2*)penalty)+1 );
			}
		}

		if (y == 0)  {
			u[tx][0] = u[tx][ty];
			g[tx][0] = g[tx][ty];
		}
		else if (y == ny-1) {
			u[tx][ty+1] = u[tx][ty];
			g[tx][ty+1] = g[tx][ty];
		}
		else {
			if (threadIdx.y == 0) {
				u[tx][0] = *( (float2*)(du-pitchbytes_2) );
				g[tx][0] = *( (float2*)(penalty-pitchbytes_2) );
			}
			else if (threadIdx.y == blockDim.y-1) {
				u[tx][ty+1] = *( (float2*)(du+pitchbytes_2) );
				g[tx][ty+1] = *( (float2*)(penalty+pitchbytes_2) );
			}
		}// end load if
	} // end load 


	__syncthreads();

	if((x+y)%2 == red){

		// ### implement me ###
		if (x < nx && y < ny){

			float a = 0.5 * (g[tx+1][ty].y+g[tx][ty].y);
			float b = 0.5 * (g[tx-1][ty].y+g[tx][ty].y);
			float c = 0.5 * (g[tx][ty].y+g[tx][ty+1].y);
			float d = 0.5 * (g[tx][ty].y+g[tx][ty-1].y);

			//Setting values at boundaries
			if (x >= nx) a = 0;
			if (x <= 0)  b = 0;
			if (y >= ny) c = 0;
			if (y <= 0)  d = 0;


			const float2 term_1 =   
					((a * u[tx+1][ty]) / (hx * hx)) +
					((b * u[tx-1][ty]) / (hx * hx)) +
					((c * u[tx][ty+1]) / (hy * hy)) +
					((d * u[tx][ty-1]) / (hy * hy));
			
			//Spatial derivatives
			const float3 s = *((float3*)(((char*)Aspatial_g) + y*pitchbytes_3)+ x);

			//Implemetation of iteration
			const float2 bg = *((float2*)(((char*)b_g) + y*pitchbytes_2)+ x);
	
			float2 value;
			value.x = (1.0f / (g[tx][ty].x * s.x + lambda *( (a + b) / (hx * hx) + (c + d) / (hy * hy) ))) 	
													* (bg.x + lambda * term_1.x - g[tx][ty].x * s.y * u[tx][ty].y);		

			value.y = (1.0f / (g[tx][ty].x * s.z + lambda *( (a + b) / (hx * hx) + (c + d) / (hy * hy) )))	
												   * (bg.y + lambda * term_1.y - g[tx][ty].x * s.y * u[tx][ty].x);		

			*((float2*)(((char*)du_g ) + y*pitchbytes_2)+ x) = (1.0f - relaxation) * u[tx][ty] + 
					relaxation * value;
		}	  	  
	}	


} // End func


void sorflow_gpu_nonlinear_warp_level
(
 const float2 *u_g,
 float2 *du_g,
 float3 *Aspatial_g,
 float3 *Atemporal_g,
 float2 *b_g,
 float2 *penalty_g,
 int   nx,
 int   ny,
 int   pitchf2,
 int   pitchf3,
 float hx,
 float hy,
 float lambda,
 float relaxation,
 int   outer_iterations,
 int   inner_iterations,
 float data_epsilon,
 float diff_epsilon
)
{
	int i, j;

	int ngx = (nx%SF_BW) ? ((nx/SF_BW)+1) : (nx/SF_BW);
	int ngy = (ny%SF_BW) ? ((ny/SF_BW)+1) : (ny/SF_BW);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);



	sorflow_compute_motion_tensor_tex<<<dimGrid,dimBlock>>>
	    (Aspatial_g,Atemporal_g,nx,ny,hx,hy,pitchf3);
	catchkernel;

	cutilSafeCall( cudaMemset(du_g,0,pitchf2*ny*sizeof(float2)));

	for(i=0;i<outer_iterations;i++)
	{

		sorflow_update_robustifications_warp_shared<<<dimGrid,dimBlock>>>
				(Aspatial_g,Atemporal_g,u_g,du_g,penalty_g,nx,ny,hx,hy,
						data_epsilon,diff_epsilon,pitchf2,pitchf3);
		catchkernel;


		//Update of Regularity Term of Flow computed in last Warp
		sorflow_update_righthandside_shared<<<dimGrid,dimBlock>>>
				(u_g,penalty_g,Atemporal_g,b_g,nx,ny,hx,hy,lambda,pitchf2,pitchf3);
		catchkernel;


		for(j=0;j<inner_iterations;j++)
		{

			sorflow_nonlinear_warp_sor_shared<<<dimGrid,dimBlock>>>
					(Aspatial_g,b_g,penalty_g,du_g,nx,ny,hx,hy,lambda,relaxation,0,pitchf2,pitchf3);
			catchkernel;
			sorflow_nonlinear_warp_sor_shared<<<dimGrid,dimBlock>>>
					(Aspatial_g,b_g,penalty_g,du_g,nx,ny,hx,hy,lambda,relaxation,1,pitchf2,pitchf3);
			catchkernel;

		}
	}

}


void sorflow_gpu_nonlinear_warp
(
	float  *I1_g,
	float  *I2_g,
	float2 *u_g,
	float2 *du_g,
	float  *I1_resampled_g,
	float  *I2_resampled_g,
	float  *I2_resampled_warped_g,
	float3 *Aspatial_g,
	float3 *Atemporal_g,
	float2 *b_g,
	float2 *penalty_g,
	int    nx,
	int    ny,
	int    pitchf1,
	int    pitchf2,
	int    pitchf3,
	float  lambda,
	int    outer_iterations,
	int    inner_iterations,
	float  relaxation,
	float  rescale_factor,
	int    start_level,
	int    end_level,
	float  data_epsilon,
	float  diff_epsilon
)
{

	int   max_rec_depth;
	int   warp_max_levels;
	int   rec_depth;

	int ngx = (nx%SF_BW) ? ((nx/SF_BW)+1) : (nx/SF_BW);
	int ngy = (ny%SF_BW) ? ((ny/SF_BW)+1) : (ny/SF_BW);
	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(SF_BW,SF_BH);



	warp_max_levels = SORFlow::compute_maximum_warp_levels(nx,ny,rescale_factor);

	max_rec_depth = (((start_level+1) < warp_max_levels) ?
									(start_level+1) : warp_max_levels) -1;



	tex_sorflow_I1.addressMode[0] = cudaAddressModeClamp;
	tex_sorflow_I1.addressMode[1] = cudaAddressModeClamp;
	tex_sorflow_I1.filterMode = IMAGE_FILTER_METHOD ;
	tex_sorflow_I1.normalized = false;

	tex_sorflow_I2.addressMode[0] = cudaAddressModeClamp;
	tex_sorflow_I2.addressMode[1] = cudaAddressModeClamp;
	tex_sorflow_I2.filterMode = IMAGE_FILTER_METHOD;
	tex_sorflow_I2.normalized = false;

  int nx_fine, ny_fine, nx_coarse=0, ny_coarse=0;

	float hx_fine;
	float hy_fine;


	cutilSafeCall( cudaMemset(u_g,0,pitchf2*ny*sizeof(float2)));

	for(rec_depth = max_rec_depth; rec_depth >= 0; rec_depth--)
	{
    // ### implement the computation of the incremental flow du_g for each level ###
		nx_fine = (int)(pow(rescale_factor, rec_depth) * nx + 0.5f);
		ny_fine = (int)(pow(rescale_factor, rec_depth) * ny + 0.5f);

    // ### setup level grid dimensions
		hx_fine = ((float)nx/nx_fine);
		hy_fine = ((float)ny/ny_fine);

    // ## resample the images to the grid dimensions
		resample_area_2d_tex(I1_g, nx, ny, pitchf1, I1_resampled_g, nx_fine, ny_fine, pitchf1);
		resample_area_2d_tex(I2_g, nx, ny, pitchf1, I2_resampled_g, nx_fine, ny_fine, pitchf1);

    // ### bind textures to resampled images 
		bind_textures(I1_resampled_g, I2_resampled_g, nx_fine, ny_fine, pitchf1);

    // ### if not at the coarsest level, resample flow from coarser level
		if(rec_depth != max_rec_depth)
			resample_area_2d_tex(u_g, nx_coarse, ny_coarse, pitchf1, u_g, nx_fine, ny_fine, pitchf2, b_g);

		if(rec_depth >= end_level)
		{

      // ### Warp the second image towards the first one, and bind
      // ### the texture for the second image to it
			bilinear_backward_warping_tex<<<dimGrid, dimBlock>>>(
					u_g,
					I2_resampled_warped_g,
					nx_fine,
					ny_fine, 
					hx_fine, 
					hy_fine, 
					pitchf1, 
					pitchf2);


			update_textures(I2_resampled_warped_g, nx_fine, ny_fine, pitchf1);

      // ### Call the function computing the incremental flow for the level
			sorflow_gpu_nonlinear_warp_level(u_g, 
					du_g, 
					Aspatial_g, 
					Atemporal_g, 
					b_g, 
					penalty_g, 
					nx_fine, 
					ny_fine, 
					pitchf2, 
					pitchf3, 
					hx_fine, 
					hy_fine,
					lambda,
					relaxation,
					outer_iterations,
					inner_iterations,
					data_epsilon,
					diff_epsilon);

      // ### add the incremental flow du_g to the coarser flow u_g
			add_flow_fields<<<dimGrid, dimBlock>>>(du_g, u_g, nx_fine, ny_fine, pitchf2);
		}

		nx_coarse = nx_fine;
		ny_coarse = ny_fine;
	}
}


void bind_textures(const float *I1_g, const float *I2_g, int nx, int ny, int pitchf1)
{
	tex_sorflow_I1.addressMode[0] = cudaAddressModeClamp;
	tex_sorflow_I1.addressMode[1] = cudaAddressModeClamp;
	tex_sorflow_I1.filterMode = IMAGE_FILTER_METHOD ;
	tex_sorflow_I1.normalized = false;

	tex_sorflow_I2.addressMode[0] = cudaAddressModeClamp;
	tex_sorflow_I2.addressMode[1] = cudaAddressModeClamp;
	tex_sorflow_I2.filterMode = IMAGE_FILTER_METHOD;
	tex_sorflow_I2.normalized = false;

	cutilSafeCall( cudaBindTexture2D(0, &tex_sorflow_I1, I1_g,
		&sorflow_float_tex, nx, ny, pitchf1*sizeof(float)) );
	cutilSafeCall( cudaBindTexture2D(0, &tex_sorflow_I2, I2_g,
		&sorflow_float_tex, nx, ny, pitchf1*sizeof(float)) );
}


void unbind_textures()
{
  cutilSafeCall (cudaUnbindTexture(tex_sorflow_I1));
  cutilSafeCall (cudaUnbindTexture(tex_sorflow_I2));
}


void update_textures(const float *I2_resampled_warped_g, int nx_fine, int ny_fine, int pitchf1)
{
	cutilSafeCall (cudaUnbindTexture(tex_sorflow_I2));
	cutilSafeCall( cudaBindTexture2D(0, &tex_sorflow_I2, I2_resampled_warped_g,
		&sorflow_float_tex, nx_fine, ny_fine, pitchf1*sizeof(float)) );
}
