#ifndef _CUDA_UTILS_
#define _CUDA_UTILS_
/*****************************************************************
 * Utils for Cuda code
 *
 * -1 CONFIG
 *
 * 0 Host Device loop bounds and sync code as well as half math conversions
 *
 * 1 CUDA ERROR CHECKING CODE
 *
 * Adapted from:
 * https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 * https://stackoverflow.com/questions/22399794/qr-decomposition-to-solve-linear-systems-in-cuda
 *
 * error checking usage for library functions:
 *    gpuErrchk(cudaMalloc((void**)&a_d, size*sizeof(int)));
 *    cusolveErrchk(<cusolvefunctioncall>)
 *    cublasErrchk(<cubalsfunctioncall>)
 * error checking usage for custom kernels:
 *    kernel<<<1,1>>>(a);
 *    gpuErrchk(cudaPeekAtLastError());
 *    gpuErrchk(cudaDeviceSynchronize());
 *
 * 2 Matrix printing code
 *
 * 3 Templated External Memory Wrapper
 *
 *   From: https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
 *
 * 4 Parallel Reductions
 *
 * 5 MATRIX INVERSION CODE
 *
 * 6 GENERAL MATRIX MATH / MOVING CODE
 *****************************************************************/
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

/*** -1 CONFIG -1 ***/
	// options for examples depend on the plant
	#define NUM_POS 7
	#define STATE_SIZE (2*NUM_POS)
	#define CONTROL_SIZE 7
	#define EE_TYPE 1 // flange but no EE on it
	#define TOTAL_TIME 0.5
	#define NUM_TIME_STEPS 64

	// optiomizer options
	#define DEBUG_SWITCH 1 // 1 for on 0 for off
	#define USE_ALG_TRACE 1 // 1 for on 0 for off
	#define USE_FINITE_DIFF 0 // 1 for on 0 for off (analytical vs. finite diff derivatives if needed)
	#define FINITE_DIFF_EPSILON 0.00001

	// define if we are working in doubles or floats
	// typedef double algType;
	typedef float algType;

	// algorithmic options
	#define LINEAR_TRANSFORM_SWITCH 1 // 1 for on 0 for off
	#define ALPHA_BEST_SWITCH 1 // 1 for on 0 for off
	#define MAX_ITER 100
	#define MAX_SOLVER_TIME 10000.0
	#define TOL_COST 0.0001 // % decrease

	// parallelization options
	#define M_BLOCKS 4
	#define M_BLOCKS_B 4//M_BLOCKS // how many time steps to do in parallel on back pass
	#define M_BLOCKS_F 2//M_BLOCKS // how many multiple shooting intervals to use in the forward pass
	#define N_BLOCKS_B (NUM_TIME_STEPS/M_BLOCKS_B)
	#define N_BLOCKS_F (NUM_TIME_STEPS/M_BLOCKS_F)
	#define FORCE_PARALLEL 1 // 0 for better performance 1 for identical output for comp b/t CPU and GPU

	// regularizer options
	#define STATE_REG 1 // use Tassa state regularization (0 indicates standard Huu regularization)
	#define RHO_INIT 12.5
	#define RHO_MAX 10000000.0 //100000.0
	#define RHO_MIN 0.01 // 0.000001
	#define RHO_FACTOR 1.25
	#ifndef IGNORE_MAX_ROX_EXIT
		#define IGNORE_MAX_ROX_EXIT 1
	#endif

	// line search options
	#define ALPHA_BASE 0.5
	#define NUM_ALPHA 16
	#define USE_EXP_RED 1 // if 0 accept if dJ > 0 else use expected reduction formula (MIN < dJ/exp < MAX)
	#define EXP_RED_MIN 0.05 // use to not accept very small changes (set to 0 to ignore)
	#define EXP_RED_MAX 1.25 // use to not allow large bad jumps (set to \infty to ignore)
	#define USE_MAX_DEFECT 1 // 0 to always accept if dJ/ExpRed is good and X will also limit to a reasonable defect per MAX_DEFECT_SIZE once a defect under X is acheived
	#define MAX_DEFECT_SIZE 1.0 // use to not allow new traj with non-physical jump artifacts in it from multiple shooting
	#define onDefectBoundary(k) ((((k+1) % N_BLOCKS_F) == 0) && (k < NUM_TIME_STEPS - 1))

	// task length / time
	#define TIME_STEP (TOTAL_TIME/(NUM_TIME_STEPS-1))
	#define get_time_us(time) (static_cast<double>(time.tv_sec * 1000000.0 + time.tv_usec))
	#define get_time_ms(time) (get_time_us(time) / 1000.0)
	#define time_delta_us(start,end) (static_cast<double>(get_time_us(end) - get_time_us(start)))
	#define time_delta_ms(start,end) (time_delta_us(start,end)/1000.0)
	#define time_delta_s(start,end) (time_delta_ms(start,end)/1000.0)

	// GPU Stream Options
	#define NUM_STREAMS (std::max(18,4+NUM_ALPHA))
	// CPU Threading Options
	#define USE_HYPER_THREADING 0 // assumes pairs are 0,CPU_CORES/2, etc. test with cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list
	#define FORCE_CORE_SWITCHES 0 // set to 1 to force a cycle across the cores (may improve speed b/c we know that tasks are independent and don't share cache in general)
	#define CPU_CORES (std::thread::hardware_concurrency())
	#if USE_HYPER_THREADING
		#define COST_THREADS (std::max(CPU_CORES,1))
		#define INTEGRATOR_THREADS (std::max(CPU_CORES,1))
		#define BP_THREADS (std::max(std::min(M_BLOCKS_B,2*CPU_CORES),1))
		#define FSIM_THREADS (std::max(std::min(M_BLOCKS_F,2*CPU_CORES),1))
		#define FSIM_ALPHA_THREADS (std::max(2*CPU_CORES/M_BLOCKS_F,1))
	#else
		#define COST_THREADS (std::max(CPU_CORES/2,1))
		#define INTEGRATOR_THREADS (std::max(CPU_CORES/2,1))
		#define BP_THREADS (std::max(std::min(M_BLOCKS_B,CPU_CORES),1))
		#define FSIM_THREADS (std::max(std::min(M_BLOCKS_F,CPU_CORES),1))
		#define FSIM_ALPHA_THREADS (std::max(CPU_CORES/M_BLOCKS_F,1))
	#endif
	#define MAX_CPU_THREADS (std::max(std::max(13,std::max((std::max(FSIM_ALPHA_THREADS,FSIM_THREADS)+1)*COST_THREADS + INTEGRATOR_THREADS + 3, FSIM_ALPHA_THREADS*FSIM_ALPHA_THREADS + 1)),3*NUM_ALPHA+3))

	// cost type options -- only applicable to Kuka arm (so default to 0)
	#ifndef EE_COST
		#define EE_COST 0 // 1 for end effector based cost / goal and 0 for joint based
	#endif
	#ifndef USE_EE_VEL_COST
		#define USE_EE_VEL_COST 0 // turn on or off the eeVel code path
	#endif
	#ifndef USE_LIMITS_FLAG
		#define USE_LIMITS_FLAG 0 // use joint vel torque limits (quad pen)
	#endif
	#ifndef USE_SMOOTH_ABS
		#define USE_SMOOTH_ABS 0 // use smooth abs cost (only applicable to EE cost)
	#endif
	#ifndef CONSTRAINTS_ON
		#define CONSTRAINTS_ON 0 // AL style constraints
	#endif

	// dynamics URDF options (only applies to Kuka)
	#ifndef USE_WAFR_URDF
		#define USE_WAFR_URDF 0
	#endif
	#ifndef MPC_MODE
		#define MPC_MODE 0 // sets gravity to 0
	#endif

	// MPC options
	#ifndef USE_MAX_SOLVER_TIME
		#define USE_MAX_SOLVER_TIME 1
	#endif

	// Matrix Dimms
	#define DIM_x_r STATE_SIZE
	#define DIM_x_c 1
	#define DIM_u_r CONTROL_SIZE
	#define DIM_u_c 1
	#define DIM_d_r STATE_SIZE
	#define DIM_d_c 1
	#define DIM_AB_r STATE_SIZE
	#define DIM_AB_c (STATE_SIZE + CONTROL_SIZE)
	#define DIM_ABT_r (STATE_SIZE + CONTROL_SIZE)
	#define DIM_ABT_c STATE_SIZE
	#define DIM_A_r STATE_SIZE
	#define DIM_A_c STATE_SIZE
	#define DIM_H_r (STATE_SIZE + CONTROL_SIZE)
	#define DIM_H_c (STATE_SIZE + CONTROL_SIZE)
	#define DIM_Hxx_r STATE_SIZE
	#define DIM_Hxx_c STATE_SIZE
	#define DIM_Hux_r CONTROL_SIZE
	#define DIM_Hux_c STATE_SIZE
	#define DIM_Hxu_r STATE_SIZE
	#define DIM_Hxu_c CONTROL_SIZE
	#define DIM_Huu_r CONTROL_SIZE
	#define DIM_Huu_c CONTROL_SIZE
	#define DIM_g_r (STATE_SIZE + CONTROL_SIZE)
	#define DIM_g_c 1
	#define DIM_gx_r STATE_SIZE
	#define DIM_gx_c 1
	#define DIM_gu_r CONTROL_SIZE
	#define DIM_gu_c 1
	#define DIM_P_r STATE_SIZE
	#define DIM_P_c STATE_SIZE
	#define DIM_p_r STATE_SIZE
	#define DIM_p_c 1
	#define DIM_K_r CONTROL_SIZE
	#define DIM_K_c STATE_SIZE
	#define DIM_KT_r DIM_K_c
	#define DIM_KT_c DIM_K_r
	#define DIM_du_r CONTROL_SIZE
	#define DIM_du_c 1
	#define OFFSET_HXU (DIM_x_r*(DIM_x_r+DIM_u_r))
	#define OFFSET_HUU (OFFSET_HXU + DIM_x_r)
	#define OFFSET_HUX_GU DIM_x_r
	#define OFFSET_B (DIM_AB_r*DIM_x_r)
	// Matrix Dimms
/*** -1 CONFIG -1 ***/

/*** 0 Host Device loop bounds and sync code 0 ***/
	__host__ __device__ __forceinline__
	void doubleLoopVals(int *starty, int *dy, int *startx, int *dx){
		#ifdef  __CUDA_ARCH__
			*starty = threadIdx.y; *dy = blockDim.y; *startx = threadIdx.x; *dx = blockDim.x;
		#else
			*starty = 0; *dy = 1; *startx = 0; *dx = 1;
		#endif
	}

	__host__ __device__ __forceinline__
	void singleLoopVals(int *start, int *delta){
		#ifdef  __CUDA_ARCH__
			*start = threadIdx.x + threadIdx.y*blockDim.x; *delta = blockDim.x*blockDim.y;
		#else
			*start = 0; *delta = 1;
		#endif
	}

	__host__ __device__ __forceinline__
	void hd__syncthreads(){
		#ifdef __CUDA_ARCH__
			__syncthreads();
  		#endif
	}

	template <int tx = 0, int ty = 0, int bx = 0, int by = 0>
	__host__ __device__ __forceinline__
	int hd__printOnce(){
		#ifdef __CUDA_ARCH__
			if(threadIdx.x != tx || threadIdx.y != ty || blockIdx.x != bx || blockIdx.y != by){return 0;}
  		#endif
		return 1;

	}
/*** 0 Host Device loop bounds and sync code 0 ***/

/*** 1 CUDA ERROR CHECKING CODE 1 ***/

  // For CUDA calls
    __host__ void gpuAssert(cudaError_t code, const char *file, const int line, bool abort=true){
        if (code != cudaSuccess){
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort){cudaDeviceReset(); exit(code);}
        }
    }
    #define gpuErrchk(err) {gpuAssert(err, __FILE__, __LINE__);}

/*** 1 CUDA ERROR CHECKING CODE 1 ***/

/*** 2 MATRIX PRINTING CODE 2 ***/
	#include <type_traits>
	template <typename T, int M, int N>
	__host__ __device__ void printMat(T *A, int lda, int newlnflag = 0){
		#pragma unroll
		for(int i=0; i<M; i++){
			#pragma unroll
            for(int j=0; j<N; j++){
            	printf("%.4f ",(double)A[i + lda*j]);
            }
            printf("\n");
        }
        if (newlnflag){printf("\n");}
	} 
	template <typename T, int M, int N>
	__global__ void printMatKern(T *A, int lda, int newlnflag = 0){
		if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){printMat<T,M,N>(A,lda,newlnflag);}
	}
	template <typename T, int M, int N>
	__global__ void printMatKern2(T **A, int i, int lda, int offset = 0, int newlnflag = 0){
		if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){T *Ai = A[i]; T *Aik = Ai + offset; printMat<T,M,N>(Aik,lda,newlnflag);}
    }
/*** 2 MATRIX PRINTING CODE 2 ***/

/*** 3 Templated External Memory Wrapper 3 ***/
	template <typename T>
	__device__ T* shared_memory_proxy()
	{
		// __align__(sizeof(T)) -- this will break if multiple Ts chosen
	    extern __shared__ unsigned char memory[];
	    return reinterpret_cast<T*>(memory);
	}
/*** 3 Templated External Memory Wrapper 3 ***/

/*** 4 Parallel Reductions 4 ***/
	// some sort of shfl_down_sync would be better but can't get that to work right now
	template <typename T>
	__device__ __forceinline__
	void reduceMax(T *data){
	   if (blockDim.x >= 1024){ if (threadIdx.x < 512) {data[threadIdx.x] = max(data[threadIdx.x],data[threadIdx.x + 512]);} __syncthreads();}
	   if (blockDim.x >= 512) { if (threadIdx.x < 256) {data[threadIdx.x] = max(data[threadIdx.x],data[threadIdx.x + 256]);} __syncthreads();}
	   if (blockDim.x >= 256) { if (threadIdx.x < 128) {data[threadIdx.x] = max(data[threadIdx.x],data[threadIdx.x + 128]);} __syncthreads();}
	   if (blockDim.x >= 128) { if (threadIdx.x < 64)  {data[threadIdx.x] = max(data[threadIdx.x],data[threadIdx.x + 64]);}  __syncthreads();}
	   if (blockDim.x >= 64)  { if (threadIdx.x < 32)  {data[threadIdx.x] = max(data[threadIdx.x],data[threadIdx.x + 32]);}  __syncthreads();}
	   if (blockDim.x >= 32)  { if (threadIdx.x < 16)  {data[threadIdx.x] = max(data[threadIdx.x],data[threadIdx.x + 16]);}  __syncthreads();}
	   if (blockDim.x >= 16)  { if (threadIdx.x < 8)   {data[threadIdx.x] = max(data[threadIdx.x],data[threadIdx.x + 8]);}   __syncthreads();}
	   if (blockDim.x >= 8)   { if (threadIdx.x < 4)   {data[threadIdx.x] = max(data[threadIdx.x],data[threadIdx.x + 4]);}   __syncthreads();}
	   if (blockDim.x >= 4)   { if (threadIdx.x < 2)   {data[threadIdx.x] = max(data[threadIdx.x],data[threadIdx.x + 2]);}   __syncthreads();}
	   if (threadIdx.x == 0)  {data[0] = max(data[0],data[1]);}
	}

	template <typename T>
	__device__ __forceinline__
	void reduceSum(T *data){
	   if (blockDim.x >= 1024){ if (threadIdx.x < 512) {data[threadIdx.x] += data[threadIdx.x + 512];} __syncthreads();}
	   if (blockDim.x >= 512) { if (threadIdx.x < 256) {data[threadIdx.x] += data[threadIdx.x + 256];} __syncthreads();}
	   if (blockDim.x >= 256) { if (threadIdx.x < 128) {data[threadIdx.x] += data[threadIdx.x + 128];} __syncthreads();}
	   if (blockDim.x >= 128) { if (threadIdx.x < 64)  {data[threadIdx.x] += data[threadIdx.x + 64];}  __syncthreads();}
	   if (blockDim.x >= 64)  { if (threadIdx.x < 32)  {data[threadIdx.x] += data[threadIdx.x + 32];}  __syncthreads();}
	   if (blockDim.x >= 32)  { if (threadIdx.x < 16)  {data[threadIdx.x] += data[threadIdx.x + 16];}  __syncthreads();}
	   if (blockDim.x >= 16)  { if (threadIdx.x < 8)   {data[threadIdx.x] += data[threadIdx.x + 8];}   __syncthreads();}
	   if (blockDim.x >= 8)   { if (threadIdx.x < 4)   {data[threadIdx.x] += data[threadIdx.x + 4];}   __syncthreads();}
	   if (blockDim.x >= 4)   { if (threadIdx.x < 2)   {data[threadIdx.x] += data[threadIdx.x + 2];}   __syncthreads();}
	   if (threadIdx.x == 0)  {data[0] += data[1];}
	}

	template <typename T>
	__device__ __forceinline__
	void reduceOr(T *data){
	   if (blockDim.x >= 1024){ if (threadIdx.x < 512) {data[threadIdx.x] |= data[threadIdx.x + 512];} __syncthreads();}
	   if (blockDim.x >= 512) { if (threadIdx.x < 256) {data[threadIdx.x] |= data[threadIdx.x + 256];} __syncthreads();}
	   if (blockDim.x >= 256) { if (threadIdx.x < 128) {data[threadIdx.x] |= data[threadIdx.x + 128];} __syncthreads();}
	   if (blockDim.x >= 128) { if (threadIdx.x < 64)  {data[threadIdx.x] |= data[threadIdx.x + 64];}  __syncthreads();}
	   if (blockDim.x >= 64)  { if (threadIdx.x < 32)  {data[threadIdx.x] |= data[threadIdx.x + 32];}  __syncthreads();}
	   if (blockDim.x >= 32)  { if (threadIdx.x < 16)  {data[threadIdx.x] |= data[threadIdx.x + 16];}  __syncthreads();}
	   if (blockDim.x >= 16)  { if (threadIdx.x < 8)   {data[threadIdx.x] |= data[threadIdx.x + 8];}   __syncthreads();}
	   if (blockDim.x >= 8)   { if (threadIdx.x < 4)   {data[threadIdx.x] |= data[threadIdx.x + 4];}   __syncthreads();}
	   if (blockDim.x >= 4)   { if (threadIdx.x < 2)   {data[threadIdx.x] |= data[threadIdx.x + 2];}   __syncthreads();}
	   if (threadIdx.x == 0)  {data[0] |= data[1];}
	}
/*** 4 Parallel Reductions 4 ***/

/*** 5 MATRIX INVERSION CODE 5 ***/
	// invert a [DIMxDIM|I] square matrix with I already loaded assumes threads >= DIM+1 x DIM
   	template <typename T, int DIM, int GUARANTEE_ENOUGH_THREADS = 0>
   	__host__ __device__ __forceinline__
   	int invertMatrix(T *A, T *s_mem = NULL){
   		// if we know we have enough threads we can avoid using shared mem and go super fast (only on CUDA)
		if (GUARANTEE_ENOUGH_THREADS){
      		#ifdef __CUDA_ARCH__
   				int kc = threadIdx.x;
				int kr = threadIdx.y;
				T inv_pivot, sRow, C;
				#pragma unroll
				for (int piv_col = 0; piv_col < DIM; piv_col++){
					// load in values
					if (kr < DIM && kc < DIM + 1){
					   inv_pivot = static_cast<T>(1)/A[piv_col + piv_col*DIM];
					   C = A[kr + piv_col*DIM];
					   sRow = A[piv_col + (piv_col + kc)*DIM];
					}
					__syncthreads();
					// compute
					if (kr < DIM && kc < DIM + 1){
					   if (kr == piv_col){A[kr + (kc+piv_col)*DIM] *= inv_pivot;}
					   else {A[kr + (kc+piv_col)*DIM] -= C*inv_pivot*sRow;}
					}
					__syncthreads();
				}	
			#else
				printf("[!] ERROR: SPEED INVERSE ONLY AVAILABLE ON GPU THIS WILL FAIL ON CPU!");
			#endif
		}
		// else use the looped version compatable on either CPU or GPU
		else{
			if (s_mem == NULL){T temp[2*DIM+1]; s_mem = temp;} // make sure to get some local mem if not passed in
			T *C = s_mem; T *sRow = &s_mem[DIM];
			int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
			#pragma unroll
			for (int piv_col = 0; piv_col < DIM; piv_col++){
				T inv_pivot = static_cast<T>(1)/A[piv_col + piv_col*DIM];
				// load in values
				#pragma unroll
				for (int kr = starty; kr < DIM; kr += dy){C[kr] = A[kr + piv_col*DIM];}
				#pragma unroll
				for (int kc = startx; kc < DIM + 1; kc += dx){sRow[kc] = A[piv_col + (piv_col + kc)*DIM];}
				hd__syncthreads();
				// compute
				#pragma unroll
				for (int kr = starty; kr < DIM; kr += dy){
				   	#pragma unroll
				   	for (int kc = startx; kc < DIM + 1; kc += dx){
				      	if (kr == piv_col){A[kr + (kc+piv_col)*DIM] *= inv_pivot;}
				      	else {A[kr + (kc+piv_col)*DIM] -= C[kr]*inv_pivot*sRow[kc];}
				   	}
				}
				hd__syncthreads();
			}
		}
		return 0;
	}
/*** 5 MATRIX INVERSION CODE 5 ***/

/*** 6 GENERAL MATRIX MATH / MOVING CODE 6 ***/
	// loads data with a given value
	template <typename T>
	__global__
	void loadWithValKern(T *A, int sizeA, T val){
		int start, delta; singleLoopVals(&start,&delta);
		for(int i = start; i < sizeA; i += delta){A[i] = val;}
	}
	template <typename T>
	__global__
	void perturbWithValKern(T *A, int sizeA, T val){
		int start, delta; singleLoopVals(&start,&delta);
		for(int i = start; i < sizeA; i += delta){A[i] += val;}
	}

	// computes the norm of a matrix
	template <typename T, int M, int N >
	__host__ __device__ __forceinline__
	T matNorm(T *A, int ld_A){
		T norm = 0;
	    #pragma unroll
	    for (int ky = 0; ky < N; ky++){
	     	#pragma unroll
			for (int kx = 0; kx < M; kx++){
	        	norm += abs(A[kx + ld_A*ky]);
	        }
		}
		return norm;
	}
	template <typename T, int M, int N >
	__global__
	void matNormKern(T *norm, T *A, int ld_A){if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){*norm = matNorm<T,M,N>(A,ld_A);}}

	// copies a matrix (optionally scales by alpha)
	template <typename T, int M, int N>
	__host__ __device__ __forceinline__
	void copyMat(T *dst, T *src, int ld_dst, int ld_src, T alpha = 1.0){
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
	    #pragma unroll
	    for (int ky = starty; ky < N; ky += dy){
	     	#pragma unroll
			for (int kx = startx; kx < M; kx += dx){
	        	dst[kx + ld_dst*ky] = alpha*src[kx + ld_src*ky];
	        }
		}
	}
	// copies blocks of array of matricies
	template <typename T, int M, int N_DST, int N_SRC, int NUM>
	__host__ __device__ __forceinline__
	void copyNMatBlock(T *dst, T *src, int ld_dst, int ld_src){
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		// y goes over the matricies
	    #pragma unroll
	    for (int k = starty; k < NUM; k += dy){
	    	// x goes across the columns
	     	#pragma unroll
			for (int c = startx; c < N_DST; c += dx){
				// then we copy the rows
				#pragma unroll
				for (int r=0; r<M; r++){dst[k*N_DST*ld_dst + ld_dst*c + r] = src[k*N_SRC*ld_src + ld_src*c + r];}
	        }
		}
	}
	template <typename T, int M, int N_DST, int N_SRC, int NUM>
	__global__
	void copyNMatBlockKern(T *dst, T *src, int ld_dst, int ld_src){copyNMatBlock<T,M,N_DST,N_SRC,NUM>(dst,src,ld_dst,ld_src);}
	// loads a matrix into shared memory
	// special case of copyMat, assumes shared memory is of size m*n and original matrix is on disk size ld*n
	template <typename T, int M, int N>
	__host__  __device__ __forceinline__
	void loadMatToShared(T *dst, T *src, int ld){
		copyMat<T,M,N>(dst, src, M, ld);
	}
	// save a matrix from shared mem
	// special case of copyMat, assumes shared memory is of size m*n and original matrix is on disk size ld*n
	template <typename T, int M, int N>
	__host__  __device__ __forceinline__
	void saveMatFromShared(T *dst, T *src, int ld){
		copyMat<T,M,N>(dst, src, ld, M);
	}

	// copies a matrix and stores the transpose
	template <typename T, int M, int N>
	__host__ __device__ __forceinline__
	void copyMatT(T *dst, T *src, int ld_dst, int ld_src){
		#ifdef __CUDA_ARCH__
			__shared__ T temp[M*N];
		    #pragma unroll
		    for (int ky = threadIdx.y; ky < N; ky += blockDim.y){
		     	#pragma unroll
				for (int kx = threadIdx.x; kx < M; kx += blockDim.x){
		        	temp[ky + N*kx] = src[kx + ld_src*ky];
		        }
			}
			#pragma unroll
		    for (int ky = threadIdx.y; ky < N; ky += blockDim.y){
				#pragma unroll
				for (int kx = threadIdx.x; kx < M; kx += blockDim.x){
					dst[kx + ld_dst*ky] = temp[kx + M*ky];
				}
			}
		#else
			for (int ky = 0; ky < N; ky++){
		     	#pragma unroll
				for (int kx = 0; kx < M; kx++){
		        	dst[ky + ld_dst*kx] = src[kx + ld_src*ky];
		        }
			}
		#endif
	}
	// save a matrix from shared mem
	// special case of copyMatT, assumes shared memory is of size m*n and original matrix is on disk size ld*n -- skips the temp step
	template <typename T, int M, int N>
	__host__  __device__ __forceinline__
	void saveMatFromSharedT(T *dst, T *src, int ld){
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		#pragma unroll
	    for (int ky = starty; ky < N; ky += dy){
	     	#pragma unroll
			for (int kx = startx; kx < M; kx += dx){
	        	dst[kx + ld*ky] = src[ky + N*kx];
	        }
		}
	}

	// loads the identiy matrix into a variable
	template <typename T, int M, int N>
	__host__  __device__ __forceinline__
	void loadIdentity(T *A, int ld_A){
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		#pragma unroll
	  	for (int ky = starty; ky < N; ky += dy){
	    	#pragma unroll
	    	for (int kx = startx; kx < M; kx += dx){
	    		A[ky*ld_A + kx] = static_cast<T>(kx == ky ? 1 : 0);
	    	}
	  	}
	}

	// loads and regularizes a matrix
	template <typename T, int M, int N>
	__host__  __device__ __forceinline__
	void loadAndReg(T *dst, T *src, int ld_dst, int ld_src, T reg){
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
	    #pragma unroll
	    for (int ky = starty; ky < N; ky += dy){
	     	#pragma unroll
			for (int kx = startx; kx < M; kx += dx){
	        	dst[kx + ld_dst*ky] = src[kx + ld_src*ky] + (kx == ky ? reg : static_cast<T>(0));
	        }
		}
	}
	// loads a and regularizes a matrix into shared memory
	// special case of copyMat, assumes shared memory is of size m*n and original matrix is on disk size ld*n
	template <typename T, int M, int N>
	__host__ __device__ __forceinline__
	void loadAndRegToShared(T *dst, T *src, int ld, T reg){
		loadAndReg<T,M,N>(dst, src, N, ld, reg);
	}

	// loads in dst = src1 - src2
	template <typename T, int M, int N>
	__host__ __device__ __forceinline__
	void loadDeltaM(T *dst, T *src1, T *src2, int ld_src1, int ld_src2){
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
	    #pragma unroll
	    for (int ky = starty; ky < N; ky += dy){
	     	#pragma unroll
			for (int kx = startx; kx < M; kx += dx){
	        	dst[kx + N*ky] = src1[kx + ld_src1*ky] - src1[kx + ld_src2*ky];
	        }
		}
	}

	template <typename T, int M>
	__host__ __device__ __forceinline__
	void loadDeltaV(T *dst, T *src1, T *src2){
		int start, delta; singleLoopVals(&start,&delta);
	    #pragma unroll
	    for (int ind = start; ind < M; ind += delta){
	    	dst[ind] = src1[ind] - src2[ind];
		}
	}

	// zero a piece of shared memory of size LENGTH*sizeof(T)
	template <typename T, int LENGTH>
	__host__ __device__ __forceinline__
	void zeroSharedMem(T *smem){
		int start, delta; singleLoopVals(&start,&delta);
		#pragma unroll
	    for (int ind = start; ind < LENGTH; ind += delta){
	    	smem[ind] = 0;
	    }
	}

	// dot product between two vectors with values spaced s_a, s_a
	template <typename T, int K>
	__host__ __device__ __forceinline__
	T dotProd(T *a, int s_a, T *b, int s_b){
		T val = 0;
		#pragma unroll
		for (int j=0; j < K; j++){
			val += a[s_a * j] * b[s_b * j];
    	}
    	return val;
	}

	// add two matricies C (+)= alpha*(A + B)
	template <typename T, int M, int N, int PEQFLAG = 0>
	__host__ __device__ __forceinline__
	void matAdd(T *C, int ld_C, T *A, int ld_A, T *B, int ld_B, T alpha = 1.0){
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		#pragma unroll
	    for (int ky = starty; ky < N; ky += dy){
	     	#pragma unroll
			for (int kx = startx; kx < M; kx += dx){
	        	T val = alpha*(A[ky + ld_A*kx] + B[ky + ld_B*kx]);
	        	if(PEQFLAG){C[ky + ld_C*kx] += val;}
				else{C[ky + ld_C*kx] = val;}
	        }
		}
	}

	// add two vectors c (+)= alpha*(a + b)
	template <typename T, int M, int PEQFLAG = 0>
	__host__ __device__ __forceinline__
	void vecAdd(T *c, T *a, T *b, T alpha = 1.0){
		int start, delta; singleLoopVals(&start,&delta);
		#pragma unroll
	    for (int ind = start; ind < M; ind += delta){
	    	T val = alpha*(a[ind] + b[ind]);
        	if(PEQFLAG){c[ind] += val;}
			else{c[ind] = val;}
		}
	}

	// add two vectors b += alpha*a
	template <typename T, int M>
	__host__ __device__ __forceinline__
	void vecAdd(T *b, T *a, T alpha = 1.0){
		int start, delta; singleLoopVals(&start,&delta);
		#pragma unroll
	    for (int ind = start; ind < M; ind += delta){
	    	b[ind] += alpha*a[ind];
		}
	}

	// for multiplying matrix row ky of A by col kx of B
	// s_a = ld_A and s_b = 1 and pass in &A[ky] and &B[kx*ld_B]
	// then we get A[ky + ld_A * j] * B[kx * ld_B + j]
	// to transpose B pass in $B[kx] and s_b = ld_B
	// to transpose A pass in $A[ky*ld_A] and s_a = 1
	template <typename T, int K, int T_A = 0, int T_B = 0>
	__host__ __device__ __forceinline__
	T dotProdMM(T *A, int ld_A, T *B, int ld_B, int kx, int ky){
		int ind_A = T_A ? ky * ld_A	: ky;
		int s_A   = T_A ? 1 		: ld_A;
		int ind_B = T_B ? kx 		: kx * ld_B;
		int s_B   = T_A ? ld_B 		: 1;
		return dotProd<T,K>(&A[ind_A],s_A,&B[ind_B],s_B);
	}

	// for multiplying matrix row ind of A by col vector b
	// s_a = ld_A and s_b = 1 and pass in &A[ind] and b
	// then we get A[ind + ld_A * j] * b[j]
	template <typename T, int K>
	__host__ __device__ __forceinline__
	T dotProdMv(T *A, int ld_A, T *b, int ind){
		return dotProd<T,K>(&A[ind],ld_A,b,1);
	}

	// basic matrix multiply D (+)= alpha*A*B (+ beta*C) with option to produce transposed output
	template <typename T, int M, int N, int K, int PEQFLAG = 0, int T_D = 0, int T_A = 0, int T_B = 0, int T_C = 0>
	__host__ __device__ __forceinline__
	void matMult(T *D, int ld_D, T *A, int ld_A, T *B, int ld_B, T alpha = 1.0, T *C = NULL, int ld_C = 0, T beta = 1.0){
		int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
		#pragma unroll
	    for (int ky = starty; ky < N; ky += dy){
	     	#pragma unroll
			for (int kx = startx; kx < M; kx += dx){
	        	T val = alpha*dotProdMM<T,K,T_A,T_B>(A,ld_A,B,ld_B,kx,ky);
	        	if (C != NULL){
	        		int ind_C = T_C ? ky + ld_C*kx : kx + ld_C*ky;
	        		val += 	beta*C[ind_C];
	        	}
	        	int ind_D = T_D ? ky + ld_D*kx : kx + ld_D*ky;
	        	if(PEQFLAG){D[ind_D] += val;}
				else{D[ind_D] = val;}
	        }
		}
	}
	
	// basic matrix vector multiply d (+)= alpha*A*b (+ c)
	template <typename T, int M, int K, int PEQFLAG = 0>
	__host__ __device__ __forceinline__
	void matVMult(T *d, T *A, int ld_A, T *b, T alpha = 1.0, T *c = NULL){
		int start, delta; singleLoopVals(&start, &delta);
	    #pragma unroll
	    for (int ind = start; ind < M; ind += delta){
			T val = alpha*dotProdMv<T,K>(A, ld_A, b, ind) + (c != NULL ? c[ind] : static_cast<T>(0));
			if(PEQFLAG){d[ind] += val;}
			else{d[ind] = val;}
		}
	}
/*** 6 GENERAL MATRIX MATH / MOVING CODE 6 ***/
#endif