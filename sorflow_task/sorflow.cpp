/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: sorflow
* file:    sorflow.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include "sorflow.h"
#include "sorflow_gpu.cuh"


SORFlow::SORFlow():
I1_g(0),
I2_g(0),
u_g(0),
du_g(0),
I1_resampled_g(0),
I2_resampled_g(0),
I2_resampled_warped_g(0),
Aspatial_g(0),
Atemporal_g(0),
b_g(0),
penalty_g(0),
output_g(0),
nx(-1),
ny(-1),
pitchf1(-1),
pitchf2(-1),
pitchf3(-1),
outer_iterations(SORFLOW_OUTER_ITERATIONS_INIT),
inner_iterations(SORFLOW_INNER_ITERATIONS_INIT),
lambda(SORFLOW_LAMBDA_INIT),
show_scale(SORFLOW_SHOWSCALE_INIT),
show_threshold(SORFLOW_SHOWTHRESHOLD_INIT),
data_epsilon(SORFLOW_DATAEPSILON_INIT),
diff_epsilon(SORFLOW_DIFFEPSILON_INIT),
start_level(0),
end_level(0),
size_set(false),
I1_set(false),
I2_set(false),
relaxation(1.0f),
method(2)
{}


SORFlow::SORFlow(int par_nx, int par_ny):
I1_g(0),
I2_g(0),
u_g(0),
du_g(0),
I1_resampled_g(0),
I2_resampled_g(0),
I2_resampled_warped_g(0),
Aspatial_g(0),
Atemporal_g(0),
b_g(0),
penalty_g(0),
output_g(0),
nx(-1),
ny(-1),
pitchf1(-1),
pitchf2(-1),
pitchf3(-1),
outer_iterations(SORFLOW_OUTER_ITERATIONS_INIT),
inner_iterations(SORFLOW_INNER_ITERATIONS_INIT),
lambda(SORFLOW_LAMBDA_INIT),
show_scale(SORFLOW_SHOWSCALE_INIT),
show_threshold(SORFLOW_SHOWTHRESHOLD_INIT),
data_epsilon(SORFLOW_DATAEPSILON_INIT),
diff_epsilon(SORFLOW_DIFFEPSILON_INIT),
start_level(0),
end_level(0),
size_set(false),
I1_set(false),
I2_set(false),
relaxation(1.0f),
method(2)
{
	set_size(par_nx,par_ny);

	start_level = compute_maximum_warp_levels(nx,ny,0.5f) - 1;
	end_level = 0;
}

SORFlow::~SORFlow()
{
	delete_fields(I1_g,I2_g,u_g,du_g,
			I1_resampled_g,I2_resampled_g,I2_resampled_warped_g,
			Aspatial_g,Atemporal_g,b_g,penalty_g,output_g);
}

void SORFlow::set_size(int par_nx, int par_ny)
{
	nx = par_nx;
	ny = par_ny;

	resize_fields(nx,ny,&pitchf1,&pitchf2,&pitchf3,
			&I1_g,&I2_g,&u_g,&du_g,
			&I1_resampled_g,&I2_resampled_g,&I2_resampled_warped_g,
			&Aspatial_g,&Atemporal_g,&b_g,&penalty_g,&output_g);

	if(start_level > compute_maximum_warp_levels(nx,ny,0.5f) - 1)
		start_level = compute_maximum_warp_levels(nx,ny,0.5f) - 1;


	size_set = true;
}

void SORFlow::get_output_size(int *par_nx, int *par_ny, int *par_channels, int *par_pitch)
{
	*par_nx = nx;
	*par_ny = ny;
	*par_channels = 3;
	*par_pitch = pitchf3;
}

void SORFlow::set_lambda(float par_lambda)
{
	lambda = par_lambda;
}


void SORFlow::set_outer_iterations(int par_iterations)
{
	outer_iterations = par_iterations;
}

void SORFlow::set_inner_iterations(int par_iterations)
{
	inner_iterations = par_iterations;
}

void SORFlow::set_show_threshold(float par_threshold)
{
	if(par_threshold >= 0.0f)
		show_threshold = par_threshold;

}

void SORFlow::set_show_scale(float par_show_scale)
{
	if(par_show_scale >= 0.0f) show_scale = par_show_scale;
	fprintf(stderr,"\nDisplay Scale set to %f",show_scale);
}

void SORFlow::set_relaxation(float par_relaxation)
{
	relaxation = par_relaxation;
}

void SORFlow::set_method(int par_method)
{
	method = par_method;
}

void SORFlow::set_input(float *par_input)
{
	if(!size_set)
	{
		fprintf(stderr,"\nERROR: FlowLib::setInput - Size has to be set first!");
		exit(1);
	}
	if(!I1_set)
	{
		cuda_copy_h2d_2D(par_input,I1_g,nx,ny,1,sizeof(float),pitchf1);
		I1_set = true;
	}
	else if(!I2_set)
	{
		cuda_copy_h2d_2D(par_input,I2_g,nx,ny,1,sizeof(float),pitchf1);
		I2_set = true;
	}
	else
	{
		float *temp = I1_g;
		I1_g = I2_g;
		I2_g = temp;

		cuda_copy_h2d_2D(par_input,I2_g,nx,ny,1,sizeof(float),pitchf1);
	}

}

void SORFlow::compute_flow()
{

	if(method == 0)
	{
		sorflow_gpu_linear(I1_g,I2_g,u_g,Aspatial_g,Atemporal_g,
				nx,ny,pitchf1,pitchf2,pitchf3,
				lambda,outer_iterations*inner_iterations,relaxation,
				data_epsilon,diff_epsilon);
	}
	else if(method == 1)
	{
		sorflow_gpu_nonlinear(I1_g,I2_g,u_g,Aspatial_g,Atemporal_g,penalty_g,
				nx,ny,pitchf1,pitchf2,pitchf3,
				lambda,outer_iterations,inner_iterations,relaxation,
				data_epsilon,diff_epsilon);
	}
	else if(method == 2)
	{
		sorflow_gpu_nonlinear_warp(I1_g,I2_g,u_g,du_g,I1_resampled_g,I2_resampled_g,I2_resampled_warped_g,Aspatial_g,Atemporal_g,b_g,penalty_g,
				nx,ny,pitchf1,pitchf2,pitchf3,
				lambda,outer_iterations,inner_iterations,relaxation,
				0.5f,start_level,end_level,
				data_epsilon,diff_epsilon);
	}

}

void SORFlow::get_output(float *par_output)
{
	cuda_copy_d2h_2D((float*)u_g,par_output,nx,ny,1,sizeof(float2),pitchf2);
}

void SORFlow::get_output_RGB(float *par_output)
{
	sorflow_hv_to_rgb((float2*)u_g,(float3*)output_g,0.0f,show_scale,show_threshold,nx,ny,pitchf2,pitchf3);
	cuda_copy_d2h_2D(output_g,par_output,nx,ny,3,sizeof(float),pitchf3);
}

void SORFlow::get_output_RGBGPU(float *par_output_g)
{
	sorflow_hv_to_rgb((float2*)u_g,(float3*)par_output_g,0.0f,show_scale,show_threshold,nx,ny,pitchf2,pitchf3);
}

void SORFlow::setinput_computeflow_getoutputRGBGPU
(
	float *par_output_g,
	float *par_input
)
{
	if(par_input) set_input(par_input);
	compute_flow();
	get_output_RGBGPU(par_output_g);
}

int SORFlow::compute_maximum_warp_levels
(
 int   nx_orig,
 int   ny_orig,
 float n_eta
)
{
	int i;
	int nx,ny;
	int nx_old,ny_old;

	nx_old = nx_orig;
	ny_old = ny_orig;

	for(i=1;;i++)
	{
		nx=(int)ceil(nx_orig*pow(n_eta,i));
		ny=(int)ceil(ny_orig*pow(n_eta,i));

		if((nx<4)||(ny<4)) break;

		nx_old = nx;
		ny_old = ny;

	}

	if((nx==1)||(ny==1)) i--;
	return i;

}

void SORFlow::set_levels(int par_start_level, int par_end_level)
{
	if(par_start_level == -1) par_start_level = compute_maximum_warp_levels(nx,ny,0.5f) - 1;
	if(par_start_level == -2) par_start_level = start_level;
	if(par_start_level >= compute_maximum_warp_levels(nx,ny,0.5f) - 1)
		par_start_level = compute_maximum_warp_levels(nx,ny,0.5f) - 1;
	if(par_end_level < 0)
	{
		if(par_end_level == -2)
			par_end_level = end_level;
		else
			par_end_level = 0;
	}
	if(par_start_level >= par_end_level)
	{
		fprintf(stderr,"\nSetting Levels %d -> %d",par_start_level,par_end_level);
		start_level = par_start_level;
		end_level = par_end_level;
	}
}
