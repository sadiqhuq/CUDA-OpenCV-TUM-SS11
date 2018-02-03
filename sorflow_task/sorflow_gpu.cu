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
const char* studentName = "John Doe";
const int   studentID   = 1234567;
/****************************************************************************\
*
* In this file the following methods have to be edited or completed:
*
* sorflow_compute_motion_tensor_tex
* sorflow_linear_sor_shared
* sorflow_update_robustifications_shared
* sorflow_nonlinear_sor_shared
* add_flow_fields
* bilinear_backward_warping_tex
* sorflow_update_robustifications_warp_shared
* sorflow_update_righthandside_shared
* sorflow_nonlinear_warp_sor_shared
* sorflow_gpu_nonlinear_warp
*
\****************************************************************************/


#include "sorflow_gpu.cuh"
#include <stdio.h>
#include "sorflow.h"
#include "resample_gpu.cuh"



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

}


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

    // ### setup level grid dimensions

    // ## resample the images to the grid dimensions

    // ### bind textures to resampled images 


    // ### if not at the coarsest level, resample flow from coarser level

		if(rec_depth >= end_level)
		{

      // ### Warp the second image towards the first one, and bind
      // ### the texture for the second image to it

      // ### Call the function computing the incremental flow for the level

      // ### add the incremental flow du_g to the coarser flow u_g
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
