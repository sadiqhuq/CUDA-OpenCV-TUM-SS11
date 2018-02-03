/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: sorflow
* file:    sorflow_gpu.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef SORFLOW_GPU_CUH
#define SORFLOW_GPU_CUH

#include "cuda_basic.cuh"


const char* getStudentName();
int         getStudentID();
bool        checkStudentNameAndID();


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
);


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
);

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
);


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
);

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
);

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
);

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
);

__global__ void sorflow_compute_motion_tensor_tex
(
	float3 *Aspatial_g,
	float3 *Atemporal_g,
	int    nx,
	int    ny,
	float  hx,
	float  hy,
	int    pitchf3
);

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
);

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
);


void bind_textures(const float *I1_g, const float *I2_g, int nx, int ny, int pitchf1);
void unbind_textures();
void update_textures(const float *I2_resampled_warped_g, int nx_fine, int ny_fine, int pitchf1);

#endif
