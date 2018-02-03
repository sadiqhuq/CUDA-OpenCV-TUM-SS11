/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: sorflow
* file:    sorflow.h
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef SORFLOW_H
#define SORFLOW_H

#include <cutil.h>
#include <cutil_inline.h>


#define SORFLOW_OUTER_ITERATIONS_INIT 40
#define SORFLOW_INNER_ITERATIONS_INIT 2
#define SORFLOW_LAMBDA_INIT 5.0f
#define SORFLOW_SHOWSCALE_INIT 1.0f
#define SORFLOW_SHOWTHRESHOLD_INIT 0.0f
#define SORFLOW_DATAEPSILON_INIT 0.1f
#define SORFLOW_DIFFEPSILON_INIT 0.001f




class SORFlow
{
public:
	SORFlow();
	SORFlow(int par_nx, int par_ny);
	~SORFlow();

	void set_size(int par_nx, int par_ny);
	void get_output_size(int *par_nx, int *par_ny, int *par_channels, int *par_pitch);


	void set_lambda(float par_lambda);
	void set_levels(int par_start_level = -1, int par_end_level = 0 );
	void set_outer_iterations(int par_iterations);
	void set_inner_iterations(int par_iterations);
	void set_show_scale(float par_show_scale = 1.0f);
	void set_show_threshold(float par_threshold = 0.0f);
	void set_relaxation(float par_relaxation);
	void set_method(int par_method);

	void set_input(float *par_input);
	void compute_flow();

	void get_output(float *par_output);
	void get_output_RGB(float *par_output);
	void get_output_RGBGPU(float *par_output_g);


	void setinput_computeflow_getoutputRGBGPU(float *par_output_g,float *par_input = 0);

	static int compute_maximum_warp_levels
	(
	 int   nx_orig,
	 int   ny_orig,
	 float n_eta
	);


	float *I1_g;
	float *I2_g;

	float2 *u_g;
	float2 *du_g;

	float *I1_resampled_g;
	float *I2_resampled_g;
	float *I2_resampled_warped_g;

	float3 *Aspatial_g;
	float3 *Atemporal_g;
	float2 *b_g;
	float2 *penalty_g;

	float *output_g;

	int nx;
	int ny;
	int pitchf1;
	int pitchf2;
	int pitchf3;

	int outer_iterations;
	int inner_iterations;

	float lambda;
	float show_scale;
	float show_threshold;

	float data_epsilon;
	float diff_epsilon;

private:
	int start_level;
	int end_level;

	bool size_set;
	bool I1_set;
	bool I2_set;

	float relaxation;

	int method;

};


#endif

