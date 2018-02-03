/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: sorflow
* file:    resample_gpu.cu
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include "resample_gpu.cuh"
#include "cuda_basic.cuh"


#ifndef RES_BW
#define RES_BW 16
#endif

#ifndef RES_BH
#define RES_BH 16
#endif


#ifndef SAMPLE_MODE_AREA
#define SAMPLE_MODE_AREA cudaFilterModeLinear
//Areabased Resample Texture Offset
#define ARTO 0.5f
#endif

#ifndef INIT_AREA
#define INIT_AREA 2
#endif



cudaChannelFormatDesc resample_float_tex = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc resample_float2_tex = cudaCreateChannelDesc<float2>();
cudaChannelFormatDesc resample_float4_tex = cudaCreateChannelDesc<float4>();

texture<float, 2, cudaReadModeElementType> tex_resample_f1;
texture<float2, 2, cudaReadModeElementType> tex_resample_f2;
texture<float4, 2, cudaReadModeElementType> tex_resample_f4;

int tex_resample_f1_initialized = 0;
int tex_resample_f2_initialized = 0;
int tex_resample_f4_initialized = 0;





__global__ void resample_area_tex_kernel
(
	float *out_g,
	int   nx,
	int   ny,
	int   pitchf1,
	float hx,
	float hy
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float result = 0.0f;
	float sum;

	float px, py, left, midx, right, top, midy, bottom;
	if(x < nx && y < ny)
	{
		py = (float)y * hy;
		//top = py - floorf(py);
		top = ceilf(py) - py;
		if(top > hy) top = hy;
		midy = hy - top;
		bottom = midy - floorf(midy);
		midy = midy - bottom;

		px = (float)x * hx;
		//left = px - floorf(px);
		left = ceilf(px) - px;
		if(left > hx) left = hx;
		midx = hx - left;
		right = midx - floorf(midx);
		midx = midx - right;

		float midx_backup = midx;

		//Top Row
		sum = left * tex2D(tex_resample_f1,px+ARTO,py+ARTO);
		px += 1.0f * (left > 0.0f);
		while(midx > 0.0f)
		{
			sum += tex2D(tex_resample_f1,px+ARTO,py+ARTO);
			px += 1.0f;
			midx -= 1.0f;
		}
		sum +=  right * tex2D(tex_resample_f1,px+ARTO,py+ARTO);

		result += top * sum;

		//Middle Rows
		py += 1.0f * (top > 0.0f);
		while(midy > 0.0f)
		{
			px = (float)x * hx;
			midx = midx_backup;
			sum = left * tex2D(tex_resample_f1,px+ARTO,py+ARTO);
			px += 1.0f * (left > 0.0f);
			while(midx > 0.0f)
			{
				sum += tex2D(tex_resample_f1,px+ARTO,py+ARTO);
				px += 1.0f;
				midx -= 1.0f;
			}
			sum +=  right * tex2D(tex_resample_f1,px+ARTO,py+ARTO);

			result += sum;

			py += 1.0f;
			midy -= 1.0f;
		}

		//Bottom Row
		px = (float)x * hx;
		sum = left * tex2D(tex_resample_f1,px+ARTO,py+ARTO);
		px += 1.0f * (left > 0.0f);
		while(midx > 0.0f)
		{
			sum += tex2D(tex_resample_f1,px+ARTO,py+ARTO);
			px += 1.0f;
			midx -= 1.0f;
		}
		sum +=  right * tex2D(tex_resample_f1,px+ARTO,py+ARTO);

		result += bottom * sum;

		result /= hx*hy;
		//result = tex2D(tex_resample_f1,(float)x*hx,(float)y*hy);

		out_g[y*pitchf1+x] = result;
	}
}



__global__ void resample_area_tex_kernel
(
	float2 *out_g,
	int   nx,
	int   ny,
	int   pitchf2,
	float hx,
	float hy
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float2 result;
	float2 sum;

	float px, py, left, midx, right, top, midy, bottom;

	result.x = result.y = 0.0f;
	if(x < nx && y < ny)
	{
		py = (float)y * hy;
		//top = py - floorf(py);
		top = ceilf(py) - py;
		if(top > hy) top = hy;
		midy = hy - top;
		bottom = midy - floorf(midy);
		midy = midy - bottom;

		px = (float)x * hx;
		//left = px - floorf(px);
		left = ceilf(px) - px;
		if(left > hx) left = hx;
		midx = hx - left;
		right = midx - floorf(midx);
		midx = midx - right;

		float midx_backup = midx;

		//Top Row
		sum.x = left * tex2D(tex_resample_f2,px+ARTO,py+ARTO).x;
		sum.y = left * tex2D(tex_resample_f2,px+ARTO,py+ARTO).y;
		px += 1.0f * (left > 0.0f);
		while(midx > 0.0f)
		{
			sum.x += tex2D(tex_resample_f2,px+ARTO,py+ARTO).x;
			sum.y += tex2D(tex_resample_f2,px+ARTO,py+ARTO).y;
			px += 1.0f;
			midx -= 1.0f;
		}
		sum.x +=  right * tex2D(tex_resample_f2,px+ARTO,py+ARTO).x;
		sum.y +=  right * tex2D(tex_resample_f2,px+ARTO,py+ARTO).y;

		result.x += top * sum.x;
		result.y += top * sum.y;

		//Middle Rows
		py += 1.0f * (top > 0.0f);
		while(midy > 0.0f)
		{
			px = (float)x * hx;
			midx = midx_backup;
			sum.x = left * tex2D(tex_resample_f2,px+ARTO,py+ARTO).x;
			sum.y = left * tex2D(tex_resample_f2,px+ARTO,py+ARTO).y;
			px += 1.0f * (left > 0.0f);
			while(midx > 0.0f)
			{
				sum.x += tex2D(tex_resample_f2,px+ARTO,py+ARTO).x;
				sum.y += tex2D(tex_resample_f2,px+ARTO,py+ARTO).y;
				px += 1.0f;
				midx -= 1.0f;
			}
			sum.x +=  right * tex2D(tex_resample_f2,px+ARTO,py+ARTO).x;
			sum.y +=  right * tex2D(tex_resample_f2,px+ARTO,py+ARTO).y;

			result.x += sum.x;
			result.y += sum.y;

			py += 1.0f;
			midy -= 1.0f;
		}

		//Bottom Row
		px = (float)x * hx;
		sum.x = left * tex2D(tex_resample_f2,px+ARTO,py+ARTO).x;
		sum.y = left * tex2D(tex_resample_f2,px+ARTO,py+ARTO).y;
		px += 1.0f * (left > 0.0f);
		while(midx > 0.0f)
		{
			sum.x += tex2D(tex_resample_f2,px+ARTO,py+ARTO).x;
			sum.y += tex2D(tex_resample_f2,px+ARTO,py+ARTO).y;
			px += 1.0f;
			midx -= 1.0f;
		}
		sum.x +=  right * tex2D(tex_resample_f2,px+ARTO,py+ARTO).x;
		sum.y +=  right * tex2D(tex_resample_f2,px+ARTO,py+ARTO).y;

		result.x += bottom * sum.x;
		result.y += bottom * sum.y;

		result.x /= hx*hy;
		result.y /= hx*hy;
		//result = tex2D(tex_resample_f2,(float)x*hx,(float)y*hy);

		out_g[y*pitchf2+x] = result;
	}
}

__global__ void resample_area_tex_kernel
(
	float3 *out_g,
	int   nx,
	int   ny,
	int   pitchf3,
	float hx,
	float hy
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float3 result;
	float3 sum;

	float px, py, left, midx, right, top, midy, bottom;

	result.x = result.y = result.z = 0.0f;
	if(x < nx && y < ny)
	{
		py = (float)y * hy;
		//top = py - floorf(py);
		top = ceilf(py) - py;
		if(top > hy) top = hy;
		midy = hy - top;
		bottom = midy - floorf(midy);
		midy = midy - bottom;

		px = (float)x * hx;
		//left = px - floorf(px);
		left = ceilf(px) - px;
		if(left > hx) left = hx;
		midx = hx - left;
		right = midx - floorf(midx);
		midx = midx - right;

		float midx_backup = midx;

		//Top Row
		sum.x = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
		sum.y = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
		sum.z = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
		px += 1.0f * (left > 0.0f);
		while(midx > 0.0f)
		{
			sum.x += tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
			sum.y += tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
			sum.z += tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
			px += 1.0f;
			midx -= 1.0f;
		}
		sum.x +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
		sum.y +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
		sum.z +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;

		result.x += top * sum.x;
		result.y += top * sum.y;
		result.z += top * sum.z;

		//Middle Rows
		py += 1.0f * (top > 0.0f);
		while(midy > 0.0f)
		{
			px = (float)x * hx;
			midx = midx_backup;
			sum.x = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
			sum.y = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
			sum.z = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
			px += 1.0f * (left > 0.0f);
			while(midx > 0.0f)
			{
				sum.x += tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
				sum.y += tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
				px += 1.0f;
				midx -= 1.0f;
			}
			sum.x +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
			sum.y +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
			sum.z +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;

			result.x += sum.x;
			result.y += sum.y;
			result.z += sum.z;

			py += 1.0f;
			midy -= 1.0f;
		}

		//Bottom Row
		px = (float)x * hx;
		sum.x = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
		sum.y = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
		sum.z = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
		px += 1.0f * (left > 0.0f);
		while(midx > 0.0f)
		{
			sum.x += tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
			sum.y += tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
			sum.z += tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
			px += 1.0f;
			midx -= 1.0f;
		}
		sum.x +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
		sum.y +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
		sum.z +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;

		result.x += bottom * sum.x;
		result.y += bottom * sum.y;
		result.z += bottom * sum.z;

		result.x /= hx*hy;
		result.y /= hx*hy;
		result.z /= hx*hy;
		//result = tex2D(tex_resample_f1,(float)x*hx,(float)y*hy);

		out_g[y*pitchf3+x] = result;
	}
}


__global__ void resample_area_tex_kernel
(
	float4 *out_g,
	int   nx,
	int   ny,
	int   pitchf4,
	float hx,
	float hy
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float4 result;
	float4 sum;

	float px, py, left, midx, right, top, midy, bottom;

	result.x = result.y = result.z = result.w = 0.0f;
	if(x < nx && y < ny)
	{
		py = (float)y * hy;
		//top = py - floorf(py);
		top = ceilf(py) - py;
		if(top > hy) top = hy;
		midy = hy - top;
		bottom = midy - floorf(midy);
		midy = midy - bottom;

		px = (float)x * hx;
		//left = px - floorf(px);
		left = ceilf(px) - px;
		if(left > hx) left = hx;
		midx = hx - left;
		right = midx - floorf(midx);
		midx = midx - right;

		float midx_backup = midx;

		//Top Row
		sum.x = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
		sum.y = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
		sum.z = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
		sum.w = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).w;
		px += 1.0f * (left > 0.0f);
		while(midx > 0.0f)
		{
			sum.x += tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
			sum.y += tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
			sum.z += tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
			sum.w += tex2D(tex_resample_f4,px+ARTO,py+ARTO).w;
			px += 1.0f;
			midx -= 1.0f;
		}
		sum.x +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
		sum.y +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
		sum.z +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
		sum.w +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).w;

		result.x += top * sum.x;
		result.y += top * sum.y;
		result.z += top * sum.z;
		result.w += top * sum.w;

		//Middle Rows
		py += 1.0f * (top > 0.0f);
		while(midy > 0.0f)
		{
			px = (float)x * hx;
			midx = midx_backup;
			sum.x = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
			sum.y = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
			sum.z = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
			sum.w = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).w;
			px += 1.0f * (left > 0.0f);
			while(midx > 0.0f)
			{
				sum.x += tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
				sum.y += tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
				sum.z += tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
				sum.w += tex2D(tex_resample_f4,px+ARTO,py+ARTO).w;
				px += 1.0f;
				midx -= 1.0f;
			}
			sum.x +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
			sum.y +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
			sum.z +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
			sum.w +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).w;

			result.x += sum.x;
			result.y += sum.y;
			result.z += sum.z;
			result.w += sum.w;

			py += 1.0f;
			midy -= 1.0f;
		}

		//Bottom Row
		px = (float)x * hx;
		sum.x = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
		sum.y = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
		sum.z = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
		sum.w = left * tex2D(tex_resample_f4,px+ARTO,py+ARTO).w;
		px += 1.0f * (left > 0.0f);
		while(midx > 0.0f)
		{
			sum.x += tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
			sum.y += tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
			sum.z += tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
			sum.w += tex2D(tex_resample_f4,px+ARTO,py+ARTO).w;
			px += 1.0f;
			midx -= 1.0f;
		}
		sum.x +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).x;
		sum.y +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).y;
		sum.z +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).z;
		sum.w +=  right * tex2D(tex_resample_f4,px+ARTO,py+ARTO).w;

		result.x += bottom * sum.x;
		result.y += bottom * sum.y;
		result.z += bottom * sum.z;
		result.w += bottom * sum.w;

		result.x /= hx*hy;
		result.y /= hx*hy;
		result.z /= hx*hy;
		result.w /= hx*hy;
		//result = tex2D(tex_resample_f1,(float)x*hx,(float)y*hy);

		out_g[y*pitchf4+x] = result;
	}
}

__global__ void stretch_f3f4
(
	float3 *in_g,
	float4 *out_g,
	int    nx,
	int    ny,
	int    pitchf3,
	int    pitchf4
)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	float3 in;
	float4 result;
	if(x < nx && y < ny)
	{
		in = in_g[y*pitchf3+x];
		result.x = in.x;
		result.y = in.y;
		result.z = in.z;
		result.w = 0.0f;
		out_g[y*pitchf4+x] = result;
	}
}



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
	float *help_g
)
{

	int pitch_help;

	if(tex_resample_f1_initialized != INIT_AREA)
	{
		tex_resample_f1.addressMode[0] = cudaAddressModeClamp;
		tex_resample_f1.addressMode[1] = cudaAddressModeClamp;
		tex_resample_f1.filterMode = SAMPLE_MODE_AREA;
		tex_resample_f1.normalized = false;
		tex_resample_f1_initialized = INIT_AREA;
	}

	int ngx = (nx_out%RES_BW) ? ((nx_out/RES_BW)+1) : (nx_out/RES_BW);
	int ngy = (ny_out%RES_BH) ? ((ny_out/RES_BH)+1) : (ny_out/RES_BH);

	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(RES_BW,RES_BH);

	cutilSafeCall( cudaBindTexture2D(0, &tex_resample_f1, in_g,
																		&resample_float_tex, nx_in, ny_in,
																		pitch_in*sizeof(float)) );

	if(in_g != out_g)
	{
		resample_area_tex_kernel<<<dimGrid,dimBlock>>>
		(out_g,nx_out,ny_out,pitch_out,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
		catchkernel;
	}
	else
	{
		if(!help_g)
		{
			cuda_malloc2D((void**)&help_g,nx_out,ny_out,1,sizeof(float),&pitch_help);
			resample_area_tex_kernel<<<dimGrid,dimBlock>>>
			(help_g,nx_out,ny_out,pitch_help,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
			catchkernel;
			cuda_copy_d2d_repitch(help_g,out_g,nx_out,ny_out,1,sizeof(float),pitch_help,pitch_out);
			cuda_free(help_g);
		}
		else
		{
			resample_area_tex_kernel<<<dimGrid,dimBlock>>>
			(help_g,nx_out,ny_out,pitch_out,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
			catchkernel;
			cuda_copy_d2d(help_g,out_g,nx_out,ny_out,1,sizeof(float),pitch_out);
		}
	}
}



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
	float2 *help_g
)
{

	int pitch_help;

	if(tex_resample_f2_initialized != INIT_AREA)
	{
		tex_resample_f2.addressMode[0] = cudaAddressModeClamp;
		tex_resample_f2.addressMode[1] = cudaAddressModeClamp;
		tex_resample_f2.filterMode = SAMPLE_MODE_AREA;
		tex_resample_f2.normalized = false;
		tex_resample_f2_initialized = INIT_AREA;
	}

	int ngx = (nx_out%RES_BW) ? ((nx_out/RES_BW)+1) : (nx_out/RES_BW);
	int ngy = (ny_out%RES_BH) ? ((ny_out/RES_BH)+1) : (ny_out/RES_BH);

	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(RES_BW,RES_BH);




	//fprintf(stderr,"\n ResampleareaF2: %i %i %i",nx_in,ny_in,pitch_in);
	cutilSafeCall( cudaBindTexture2D(0, &tex_resample_f2, in_g,
																		&resample_float2_tex, nx_in, ny_in,
																		pitch_in*sizeof(float2)) );

	if(in_g != out_g)
	{
		resample_area_tex_kernel<<<dimGrid,dimBlock>>>
		(out_g,nx_out,ny_out,pitch_out,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
		catchkernel;
	}
	else
	{
		if(!help_g)
		{
			//fprintf(stderr,"\nSelfalloc Help Resample");
			cuda_malloc2D((void**)&help_g,nx_out,ny_out,1,sizeof(float2),&pitch_help);
			resample_area_tex_kernel<<<dimGrid,dimBlock>>>
			(help_g,nx_out,ny_out,pitch_help,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
			catchkernel;
			cuda_copy_d2d_repitch((float*)help_g,(float*)out_g,nx_out,ny_out,1,sizeof(float2),pitch_help,pitch_out);
			cuda_free(help_g);
		}
		else
		{
			//fprintf(stderr,"\nPrealloc Help Resample");
			resample_area_tex_kernel<<<dimGrid,dimBlock>>>
			(help_g,nx_out,ny_out,pitch_out,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
			catchkernel;
			//cuda_copy_d2d((float*)in_g,(float*)help_g,nx_out,ny_out,1,sizeof(float2),pitch_out);
			cuda_copy_d2d((float*)help_g,(float*)out_g,nx_out,ny_out,1,sizeof(float2),pitch_out);
		}
	}
}

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
)
{

	if(tex_resample_f4_initialized != INIT_AREA)
	{
		tex_resample_f4.addressMode[0] = cudaAddressModeClamp;
		tex_resample_f4.addressMode[1] = cudaAddressModeClamp;
		tex_resample_f4.filterMode = SAMPLE_MODE_AREA;
		tex_resample_f4.normalized = false;
		tex_resample_f4_initialized = INIT_AREA;
	}

	int ngx = (nx_out%RES_BW) ? ((nx_out/RES_BW)+1) : (nx_out/RES_BW);
	int ngy = (ny_out%RES_BH) ? ((ny_out/RES_BH)+1) : (ny_out/RES_BH);

	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(RES_BW,RES_BH);


	bool selfalloc = (help_g == 0);

	if (selfalloc)
		cuda_malloc2D((void**)&help_g,nx_in,ny_in,1,sizeof(float4),&pitch_help);


	cutilSafeCall( cudaBindTexture2D(0, &tex_resample_f4, help_g,
																		&resample_float4_tex, nx_in, ny_in,
																		pitch_help*sizeof(float4)) );

	stretch_f3f4<<<dimGrid,dimBlock>>>
	(in_g,help_g,nx_in,ny_in,pitch_in,pitch_help);
	catchkernel;

	resample_area_tex_kernel<<<dimGrid,dimBlock>>>
	(out_g,nx_out,ny_out,pitch_out,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
	catchkernel;

	if (selfalloc)
		cuda_free(help_g);

}

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
	float4 *help_g
)
{

	int pitch_help;

	if(tex_resample_f4_initialized != INIT_AREA)
	{
		tex_resample_f4.addressMode[0] = cudaAddressModeClamp;
		tex_resample_f4.addressMode[1] = cudaAddressModeClamp;
		tex_resample_f4.filterMode = SAMPLE_MODE_AREA;
		tex_resample_f4.normalized = false;
		tex_resample_f4_initialized = INIT_AREA;
	}

	int ngx = (nx_out%RES_BW) ? ((nx_out/RES_BW)+1) : (nx_out/RES_BW);
	int ngy = (ny_out%RES_BH) ? ((ny_out/RES_BH)+1) : (ny_out/RES_BH);

	dim3 dimGrid(ngx,ngy);
	dim3 dimBlock(RES_BW,RES_BH);

	cutilSafeCall( cudaBindTexture2D(0, &tex_resample_f4, in_g,
																		&resample_float4_tex, nx_in, ny_in,
																		pitch_in*sizeof(float4)) );

	if(in_g != out_g)
	{
		resample_area_tex_kernel<<<dimGrid,dimBlock>>>
		(out_g,nx_out,ny_out,pitch_out,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
		catchkernel;
	}
	else
	{
		if(!help_g)
		{
			cuda_malloc2D((void**)&help_g,nx_out,ny_out,1,sizeof(float4),&pitch_help);
			resample_area_tex_kernel<<<dimGrid,dimBlock>>>
			(help_g,nx_out,ny_out,pitch_help,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
			catchkernel;
			cuda_copy_d2d_repitch((float*)help_g,(float*)out_g,nx_out,ny_out,1,sizeof(float4),pitch_help,pitch_out);
			cuda_free(help_g);
		}
		else
		{
			resample_area_tex_kernel<<<dimGrid,dimBlock>>>
			(help_g,nx_out,ny_out,pitch_out,(float)nx_in/(float)nx_out,(float)ny_in/(float)ny_out);
			catchkernel;
			cuda_copy_d2d((float*)help_g,(float*)out_g,nx_out,ny_out,1,sizeof(float4),pitch_out);
		}
	}
}
