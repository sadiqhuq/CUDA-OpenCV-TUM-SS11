/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    summer term 2011 / 19-26th September
*
* project: sorflow
* file:    cuda_basic.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef FILLIB_BASIC_CUH
#define FILLIB_BASIC_CUH

#include <cutil.h>
#include <cutil_inline.h>



//#define DEBUG_MODUS
#ifdef DEBUG_MODUS
#define catchkernel cutilSafeCall(cudaThreadSynchronize())
#else
#define catchkernel
#endif



#ifdef SM_13
#define ATOMICADD(a,b) atomicAdd(((a)),((b)))
#else
#define ATOMICADD(a,b) *((a)) += ((b))
#endif

#ifndef ABS
//#define ABS(a) ( ( (a) < 0 ) ? ( - (a) ) : (a) )
#define ABS(a) fabsf((a))
#endif

#ifndef DIVCEIL_HOST
#define DIVCEIL_HOST
inline int divceil(int a, int b)
{
	return (a%b == 0) ? (a/b) : (a/b + 1);
}
#endif

#define TEXTURE_OFFSET 0.5f


bool cuda_malloc2D(void **device_image, int nx, int ny, int nc,
									 size_t type_size, int *pitch);

bool cuda_malloc2D_manual(void **device_image, int nx, int ny, int nc,
									 size_t type_size, int pitch);

void compute_pitch_manual(int nx, int ny, int nc,
									 size_t type_size, int *pitch);

int compute_pitch_manual(int nx, int ny, int nc,
									 size_t type_size);

void compute_pitch_alloc(void **device_image, int nx, int ny, int nc,
									 size_t type_size, int *pitch);

int compute_pitch_alloc(void **device_image, int nx, int ny, int nc,
									 size_t type_size);

bool cuda_copy_h2d_2D(float *host_ptr, float *device_ptr,
											int nx, int ny, int nc,
										  size_t type_size, int pitch);

bool cuda_copy_d2h_2D(float *device_ptr, float *host_ptr,
											int nx, int ny, int nc,
										  size_t type_size, int pitch);

bool cuda_copy_d2d(float *device_in, float *device_out,
											int nx, int ny, int nc,
										  size_t type_size, int pitch);

bool cuda_copy_d2d_repitch(float *device_in, float *device_out,
											int nx, int ny, int nc,
										  size_t type_size, int pitch_in, int pitch_out);

void cuda_zero(float *device_ptr, size_t size);

void cuda_value
(
	float *device_prt,
	int size,
	float value
);

bool cuda_free(void *device_ptr);


bool device_query_and_select(int request);

bool cuda_memory_available
(
	size_t *total,
	size_t *free,
	unsigned int device
);








#endif // #ifndef FILLIB_BASIC_CUH


