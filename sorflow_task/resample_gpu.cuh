/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: sorflow
* file:    resample_gpu.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef RESAMPLE_GPU_H
#define RESAMPLE_GPU_H
#include <cutil.h>
#include <cutil_inline.h>

void resample_area_2d_tex
(
	float *in_g,
	int   nx_in,
	int   ny_in,
	int   pitch_in,
	float *out_g,
	int   nx_out,
	int   ny_out,
	int   pitch_out,
	float *help_g = 0
);

void resample_area_2d_tex
(
	float2 *in_g,
	int   nx_in,
	int   ny_in,
	int   pitch_in,
	float2 *out_g,
	int   nx_out,
	int   ny_out,
	int   pitch_out,
	float2 *help_g = 0
);

void resample_area_2d_tex
(
	float3 *in_g,
	int   nx_in,
	int   ny_in,
	int   pitch_in,
	float3 *out_g,
	int   nx_out,
	int   ny_out,
	int   pitch_out,
	float4 *help_g,
	int   pitch_help
);

void resample_area_2d_tex
(
	float4 *in_g,
	int   nx_in,
	int   ny_in,
	int   pitch_in,
	float4 *out_g,
	int   nx_out,
	int   ny_out,
	int   pitch_out,
	float4 *help_g = 0
);


#endif
