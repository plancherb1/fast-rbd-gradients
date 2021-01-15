/***
nvcc -std=c++11 -o GPU_timing.exe time_GPU.cu -gencode arch=compute_75,code=sm_75 -O3
***/

#define NUM_STREAMS 4
#include "utils/experiment_helpers.h"
#include "helpers_CPU/dynamicsGradient.h" // GPU requires CPU for partial kernels
#include "helpers_GPU/dynamicsGradient.cuh"

template <typename T, int TEST_ITERS, bool MPC_MODE = false, bool VEL_DAMPING = true>
__global__
void kern_single_full(T *d_dqdd, T *d_mem_vol, T *d_mem_const, T *s_fext = nullptr){
	int starty, dy, startx, dx; doubleLoopVals_GPU(&starty,&dy,&startx,&dx); int start, delta; singleLoopVals_GPU(&start,&delta);
	__shared__ T s_mem_const[CONST_VALS];  T *s_T = &s_mem_const[0]; T *s_I = &s_mem_const[36*NUM_POS];
	__shared__ T s_mem_vol[COMPLETELY_FUSED_VALS]; T *s_q = &s_mem_vol[0]; T *s_qd = &s_mem_vol[NUM_POS];
						 						   T *s_qdd = &s_mem_vol[3*NUM_POS]; T *s_Minv = &s_mem_vol[4*NUM_POS]; //T *s_u = &s_mem_vol[2*NUM_POS]; 
	__shared__ T s_sinq[NUM_POS];          __shared__ T s_cosq[NUM_POS];          __shared__ T s_temp[6*NUM_POS];
	__shared__ T s_v[6*NUM_POS];           __shared__ T s_a[6*NUM_POS];           __shared__ T s_f[6*NUM_POS];
	__shared__ T s_cdu[2*NUM_POS*NUM_POS]; __shared__ T s_dqdd[2*NUM_POS*NUM_POS];
	// Load in vars to shared mem
	#pragma unroll
	for (int ind = start; ind < CONST_VALS; ind += delta){s_mem_const[ind] = d_mem_const[ind];}
	#pragma unroll
	for (int ind = start; ind < COMPLETELY_FUSED_VALS; ind += delta){s_mem_vol[ind] = d_mem_vol[ind];}
	#pragma unroll
	for (int ind = start; ind < NUM_POS; ind += delta){s_sinq[ind] = sin(s_q[ind]); s_cosq[ind] = cos(s_q[ind]);}
	__syncthreads();
	// loop for all test iters
	for (int i = 0; i < TEST_ITERS; i++){
		// build Tmats
		updateTransforms_GPU<T>(s_T,s_sinq,s_cosq); __syncthreads();
		// dqdd/dtau = Minv and  dqdd/dq(d) = -Minv*dc/dq(d) note: dM term in dq drops out if you compute c with fd qdd per carpentier
		FD_helpers_GPU_vaf<T,MPC_MODE>(s_v,s_a,s_f,s_qd,s_qdd,s_I,s_T,s_temp,s_fext); __syncthreads();
		inverseDynamicsGradient_GPU<T,VEL_DAMPING>(s_cdu,s_qd,s_v,s_a,s_f,s_I,s_T); __syncthreads();
		// finally compute the final dqdd by multiplying by Minv
		finish_dqdd_GPU<T>(s_dqdd,s_cdu,s_Minv); __syncthreads();
	}
	#pragma unroll
	for (int ind = start; ind < 2*NUM_POS*NUM_POS; ind += delta){d_dqdd[ind] = s_dqdd[ind];}
}

template <typename T, int TEST_ITERS, bool MPC_MODE = false, bool VEL_DAMPING = true>
__global__
void kern_single_vaf_dcdu(T *d_cdu, T *d_mem_vol, T *d_mem_const, T *s_fext = nullptr){
	int starty, dy, startx, dx; doubleLoopVals_GPU(&starty,&dy,&startx,&dx); int start, delta; singleLoopVals_GPU(&start,&delta);
	__shared__ T s_mem_const[CONST_VALS];  T *s_T = &s_mem_const[0]; T *s_I = &s_mem_const[36*NUM_POS];
	__shared__ T s_mem_vol[FUSED_VALS]; T *s_q = &s_mem_vol[0]; T *s_qd = &s_mem_vol[NUM_POS]; T *s_qdd = &s_mem_vol[3*NUM_POS];
						 				// T *s_u = &s_mem_vol[2*NUM_POS]; 
	__shared__ T s_sinq[NUM_POS];          __shared__ T s_cosq[NUM_POS];          __shared__ T s_temp[6*NUM_POS];
	__shared__ T s_v[6*NUM_POS];           __shared__ T s_a[6*NUM_POS];           __shared__ T s_f[6*NUM_POS];
	__shared__ T s_cdu[2*NUM_POS*NUM_POS];
	// Load in vars to shared mem
	#pragma unroll
	for (int ind = start; ind < CONST_VALS; ind += delta){s_mem_const[ind] = d_mem_const[ind];}
	#pragma unroll
	for (int ind = start; ind < FUSED_VALS; ind += delta){s_mem_vol[ind] = d_mem_vol[ind];}
	#pragma unroll
	for (int ind = start; ind < NUM_POS; ind += delta){s_sinq[ind] = sin(s_q[ind]); s_cosq[ind] = cos(s_q[ind]);}
	__syncthreads();
	// loop for all test iters
	for (int i = 0; i < TEST_ITERS; i++){
		// build Tmats
		updateTransforms_GPU<T>(s_T,s_sinq,s_cosq); __syncthreads();
		// dqdd/dtau = Minv and  dqdd/dq(d) = -Minv*dc/dq(d) note: dM term in dq drops out if you compute c with fd qdd per carpentier
		FD_helpers_GPU_vaf<T,MPC_MODE>(s_v,s_a,s_f,s_qd,s_qdd,s_I,s_T,s_temp,s_fext); __syncthreads();
		inverseDynamicsGradient_GPU<T,VEL_DAMPING>(s_cdu,s_qd,s_v,s_a,s_f,s_I,s_T); __syncthreads();
	}
	#pragma unroll
	for (int ind = start; ind < 2*NUM_POS*NUM_POS; ind += delta){d_cdu[ind] = s_cdu[ind];}
}

template <typename T, int TEST_ITERS, bool MPC_MODE = false, bool VEL_DAMPING = true>
__global__
void kern_single_vaf(T *d_vaf, T *d_mem_vol, T *d_mem_const, T *s_fext = nullptr){
	int starty, dy, startx, dx; doubleLoopVals_GPU(&starty,&dy,&startx,&dx); int start, delta; singleLoopVals_GPU(&start,&delta);
	__shared__ T s_mem_const[CONST_VALS];  T *s_T = &s_mem_const[0]; T *s_I = &s_mem_const[36*NUM_POS];
	__shared__ T s_mem_vol[FUSED_VALS]; T *s_q = &s_mem_vol[0]; T *s_qd = &s_mem_vol[NUM_POS]; T *s_qdd = &s_mem_vol[3*NUM_POS];
						 				// T *s_u = &s_mem_vol[2*NUM_POS];
	__shared__ T s_sinq[NUM_POS];          __shared__ T s_cosq[NUM_POS];          __shared__ T s_temp[6*NUM_POS];
	__shared__ T s_v[6*NUM_POS];           __shared__ T s_a[6*NUM_POS];           __shared__ T s_f[6*NUM_POS];
	// Load in vars to shared mem
	#pragma unroll
	for (int ind = start; ind < CONST_VALS; ind += delta){s_mem_const[ind] = d_mem_const[ind];}
	#pragma unroll
	for (int ind = start; ind < FUSED_VALS; ind += delta){s_mem_vol[ind] = d_mem_vol[ind];}
	#pragma unroll
	for (int ind = start; ind < NUM_POS; ind += delta){s_sinq[ind] = sin(s_q[ind]); s_cosq[ind] = cos(s_q[ind]);}
	__syncthreads();
	// loop for all test iters
	for (int i = 0; i < TEST_ITERS; i++){
		// build Tmats
		updateTransforms_GPU<T>(s_T,s_sinq,s_cosq); __syncthreads();
		// dqdd/dtau = Minv and  dqdd/dq(d) = -Minv*dc/dq(d) note: dM term in dq drops out if you compute c with fd qdd per carpentier
		FD_helpers_GPU_vaf<T,MPC_MODE>(s_v,s_a,s_f,s_qd,s_qdd,s_I,s_T,s_temp,s_fext); __syncthreads();
	}
	#pragma unroll
	for (int ind = start; ind < 6*NUM_POS; ind += delta){d_vaf[ind] = s_v[ind]; d_vaf[ind+6*NUM_POS] = s_a[ind]; d_vaf[ind+12*NUM_POS] = s_f[ind];}
}

template<typename T, int TEST_ITERS, int NUM_THREADS_TEST, int NUM_TIME_STEPS_TEST, bool MPC_MODE = false, bool VEL_DAMPING = true>
void test(){
	// allocate and load on CPU
	T *h_dqdd = (T *)malloc(2*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T));
	T *h_cdu = (T *)malloc(2*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T));
	T *h_vaf = (T *)malloc(6*NUM_POS*3*NUM_TIME_STEPS_TEST*sizeof(T));
	T *h_mem_const = (T *)malloc(CONST_VALS*NUM_THREADS_TEST*sizeof(T)); // T,I
	for(int tid = 0; tid < NUM_THREADS_TEST; tid++){
		initTransforms(&h_mem_const[CONST_VALS*tid]); initInertiaTensors(&h_mem_const[CONST_VALS*tid + 36*NUM_POS]);
	}
	traj<T,NUM_TIME_STEPS_TEST> *testTraj = new traj<T,NUM_TIME_STEPS_TEST>;
	for (int k = 0; k < NUM_TIME_STEPS_TEST; k++){
		for(int i = 0; i < NUM_POS; i++){
			testTraj->knots[k].q[i] = getRand<T>(); 
			testTraj->knots[k].qd[i] = getRand<T>();
			testTraj->knots[k].u[i] = getRand<T>();
		}
		forwardDynamics<T,MPC_MODE,VEL_DAMPING,1>(&(testTraj->knots[k]));
	}
	// and load into gpu structured memory (on the CPU)
	T *h_mem_vol_split = (T *)malloc(SPLIT_VALS*NUM_TIME_STEPS_TEST*sizeof(T)); // q,qd,u,qdd,v,a,f
	T *h_Minv = (T *)malloc(NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)); // for finish in split kernel
	T *h_mem_vol_fused = (T *)malloc(FUSED_VALS*NUM_TIME_STEPS_TEST*sizeof(T)); // q,qd,u,qdd
	T *h_mem_vol_completely_fused = (T *)malloc(COMPLETELY_FUSED_VALS*NUM_TIME_STEPS_TEST*sizeof(T)); // q,qd,u,qdd,Minv
	for (int k = 0; k < NUM_TIME_STEPS_TEST; k++){
		T *h_mem_vol_splitk = &h_mem_vol_split[SPLIT_VALS*k];
		T *h_mem_vol_fusedk = &h_mem_vol_fused[FUSED_VALS*k];
		T *h_mem_vol_completely_fusedk = &h_mem_vol_completely_fused[COMPLETELY_FUSED_VALS*k];
		for(int i = 0; i < NUM_POS; i++){
			h_mem_vol_splitk[i] = testTraj->knots[k].q[i];
			h_mem_vol_splitk[i+NUM_POS] = testTraj->knots[k].qd[i];
			h_mem_vol_splitk[i+2*NUM_POS] = testTraj->knots[k].u[i];
			h_mem_vol_splitk[i+3*NUM_POS] = testTraj->knots[k].qdd[i];
		}
		for(int i = 0; i < 4*NUM_POS; i++){
			h_mem_vol_fusedk[i] = h_mem_vol_splitk[i]; h_mem_vol_completely_fusedk[i] = h_mem_vol_splitk[i];
		}
		for(int i = 0; i < NUM_POS*NUM_POS; i++){
			h_mem_vol_completely_fusedk[4*NUM_POS + i] = testTraj->knots[k].Minv[i];
			h_Minv[NUM_POS*NUM_POS*k + i] = testTraj->knots[k].Minv[i];
		}
	}

	// allocate and copy to GPU
	T *d_dqdd; gpuErrchk(cudaMalloc((void**)&d_dqdd, 2*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_cdu; gpuErrchk(cudaMalloc((void**)&d_cdu, 2*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_vaf; gpuErrchk(cudaMalloc((void**)&d_vaf, 6*NUM_POS*3*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_mem_const; gpuErrchk(cudaMalloc((void**)&d_mem_const, CONST_VALS*sizeof(T))); // T,I
	T *d_mem_vol_split; gpuErrchk(cudaMalloc((void**)&d_mem_vol_split, SPLIT_VALS*NUM_TIME_STEPS_TEST*sizeof(T))); // q,qd,u,qdd,v,a,f
	T *d_mem_vol_fused; gpuErrchk(cudaMalloc((void**)&d_mem_vol_fused, FUSED_VALS*NUM_TIME_STEPS_TEST*sizeof(T))); // q,qd,u,qdd
	T *d_mem_vol_completely_fused; gpuErrchk(cudaMalloc((void**)&d_mem_vol_completely_fused, COMPLETELY_FUSED_VALS*NUM_TIME_STEPS_TEST*sizeof(T))); // q,qd,u,qdd,Minv
	cudaStream_t *streams = (cudaStream_t *)malloc(NUM_STREAMS*sizeof(cudaStream_t));
	int priority, minPriority, maxPriority;
	gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
	for(int i=0; i<NUM_STREAMS; i++){priority = std::min(minPriority+i,maxPriority);
		gpuErrchk(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamNonBlocking,priority));
	}
	gpuErrchk(cudaMemcpy(d_mem_const,h_mem_const,CONST_VALS*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mem_vol_split,h_mem_vol_split,SPLIT_VALS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mem_vol_fused,h_mem_vol_fused,FUSED_VALS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mem_vol_completely_fused,h_mem_vol_completely_fused,COMPLETELY_FUSED_VALS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());

	// create resusable threads for any multithreaded CPU calls
	ReusableThreads<NUM_THREADS_TEST> threads;

	// time kernel with uncompressed and compressed memory copy
	struct timespec start, end;
	if(NUM_TIME_STEPS_TEST == 1){
		#if TEST_FOR_EQUIVALENCE
			#define SINGLE_TEST_ITERS (TEST_ITERS*1)
		#else
			#define SINGLE_TEST_ITERS (TEST_ITERS*10)
		#endif
		clock_gettime(CLOCK_MONOTONIC,&start);
		kern_single_full<T,SINGLE_TEST_ITERS,MPC_MODE,VEL_DAMPING><<<1,BlockDimms>>>(d_dqdd,d_mem_vol_completely_fused,d_mem_const);
		gpuErrchk(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("Single Call vaf+dc/du+dqdd/du %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));

		gpuErrchk(cudaMemcpyAsync(h_dqdd,d_dqdd,NUM_POS*2*NUM_POS*sizeof(T),cudaMemcpyDeviceToHost, streams[0])); gpuErrchk(cudaDeviceSynchronize());
		
		#if TEST_FOR_EQUIVALENCE
			printf("q,qd,qdd,u\n");
			printMat<T,1,NUM_POS>(h_mem_vol_completely_fused,1);
			printMat<T,1,NUM_POS>(&h_mem_vol_completely_fused[NUM_POS],1);
			printMat<T,1,NUM_POS>(&h_mem_vol_completely_fused[3*NUM_POS],1);
			printMat<T,1,NUM_POS>(&h_mem_vol_completely_fused[2*NUM_POS],1);
			printf("Minv\n");
			printMat<T,NUM_POS,NUM_POS>(&h_mem_vol_completely_fused[4*NUM_POS],NUM_POS);
			printf("dqdd/dq\n");
			printMat<T,NUM_POS,NUM_POS>(h_dqdd,NUM_POS);
			printf("dqdd/dqd\n");
			printMat<T,NUM_POS,NUM_POS>(&h_dqdd[NUM_POS*NUM_POS],NUM_POS);
		#endif

		clock_gettime(CLOCK_MONOTONIC,&start);
		kern_single_vaf_dcdu<T,SINGLE_TEST_ITERS,MPC_MODE,VEL_DAMPING><<<1,BlockDimms>>>(d_dqdd,d_mem_vol_fused,d_mem_const);
		gpuErrchk(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("Single Call vaf+dc/du %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
		
		clock_gettime(CLOCK_MONOTONIC,&start);
		kern_single_vaf<T,SINGLE_TEST_ITERS,MPC_MODE,VEL_DAMPING><<<1,BlockDimms>>>(d_vaf,d_mem_vol_fused,d_mem_const);
		gpuErrchk(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("Single Call vaf %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
		printf("----------------------------------------\n");
	}
	// time each end to end option (and memory back outs etc.)
	else{
		// ----------------------------------------//
		// Split: CPU vaf > GPU dc/du > CPU finish //
		// ----------------------------------------//
		std::vector<double> times = {};
		for(int iter = 0; iter < TEST_ITERS; iter++){
			clock_gettime(CLOCK_MONOTONIC,&start);
			dynamicsGradientReusableThreaded_start<T,NUM_THREADS_TEST,NUM_TIME_STEPS_TEST,MPC_MODE,VEL_DAMPING>(h_mem_vol_split,h_mem_const,&threads);
			clock_gettime(CLOCK_MONOTONIC,&end);
			times.push_back(time_delta_us_timespec(start,end));
		}
		printf("[N:%d]: Split - CPU start: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times);
	
		std::vector<double> times2 = {};
	    for(int iter = 0; iter < TEST_ITERS; iter++){
	    	clock_gettime(CLOCK_MONOTONIC,&start);
		    gpuErrchk(cudaMemcpy(d_mem_vol_split,h_mem_vol_split,SPLIT_VALS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice));
			dynamicsGradientKernel_split<T,NUM_TIME_STEPS_TEST,VEL_DAMPING><<<NUM_TIME_STEPS_TEST,BlockDimms>>>(d_cdu,d_mem_vol_split,d_mem_const);
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaMemcpy(h_cdu,d_cdu,2*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyDeviceToHost));
			clock_gettime(CLOCK_MONOTONIC,&end);
			times2.push_back(time_delta_us_timespec(start,end));
		}		
		printf("[N:%d]: Split - GPU Compute + I/O: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times2);
		
		std::vector<double> times3 = {};
	    for(int iter = 0; iter < TEST_ITERS; iter++){
	    	clock_gettime(CLOCK_MONOTONIC,&start);
		    dynamicsGradientKernel_split<T,NUM_TIME_STEPS_TEST,VEL_DAMPING><<<NUM_TIME_STEPS_TEST,BlockDimms>>>(d_cdu,d_mem_vol_split,d_mem_const);
			gpuErrchk(cudaDeviceSynchronize());
			clock_gettime(CLOCK_MONOTONIC,&end);
			times3.push_back(time_delta_us_timespec(start,end));
		}
		printf("[N:%d]: Split - GPU Compute: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times3);
		
		std::vector<double> times4 = {};
		for(int iter = 0; iter < TEST_ITERS; iter++){
			clock_gettime(CLOCK_MONOTONIC,&start);
	    	dynamicsGradientReusableThreaded_finish<T,NUM_THREADS_TEST,NUM_TIME_STEPS_TEST,MPC_MODE,VEL_DAMPING>(h_dqdd,h_cdu,h_Minv,&threads);
	    	clock_gettime(CLOCK_MONOTONIC,&end);
	    	times4.push_back(time_delta_us_timespec(start,end));
	    }
		printf("[N:%d] Split - CPU finish: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times4);
		printf("\n");

		// ---------------------------------//
		// Fused: GPU vaf+dcdu > CPU finish //
		// ---------------------------------//
		std::vector<double> times5 = {};
	    for(int iter = 0; iter < TEST_ITERS; iter++){
	    	clock_gettime(CLOCK_MONOTONIC,&start);
		    gpuErrchk(cudaMemcpy(d_mem_vol_fused,h_mem_vol_fused,FUSED_VALS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice));
			dynamicsGradientKernel_fused<T,NUM_TIME_STEPS_TEST,VEL_DAMPING><<<NUM_TIME_STEPS_TEST,BlockDimms>>>(d_dqdd,d_mem_vol_fused,d_mem_const);
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaMemcpy(h_dqdd,d_dqdd,2*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyDeviceToHost));
			clock_gettime(CLOCK_MONOTONIC,&end);
			times5.push_back(time_delta_us_timespec(start,end));
		}
		printf("[N:%d] Fused - GPU compute + I/O: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times5);
		
		std::vector<double> times6 = {};
	    for(int iter = 0; iter < TEST_ITERS; iter++){
	    	clock_gettime(CLOCK_MONOTONIC,&start);
		    dynamicsGradientKernel_fused<T,NUM_TIME_STEPS_TEST,VEL_DAMPING><<<NUM_TIME_STEPS_TEST,BlockDimms>>>(d_dqdd,d_mem_vol_fused,d_mem_const);
			gpuErrchk(cudaDeviceSynchronize());
			clock_gettime(CLOCK_MONOTONIC,&end);
			times6.push_back(time_delta_us_timespec(start,end));
		}
		printf("[N:%d] Fused - GPU compute: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times6);
		
		std::vector<double> times7 = {};
		for(int iter = 0; iter < TEST_ITERS; iter++){
			clock_gettime(CLOCK_MONOTONIC,&start);
	    	dynamicsGradientReusableThreaded_finish<T,NUM_THREADS_TEST,NUM_TIME_STEPS_TEST,MPC_MODE,VEL_DAMPING>(h_dqdd,h_cdu,h_Minv,&threads);
	    	clock_gettime(CLOCK_MONOTONIC,&end);
	    	times7.push_back(time_delta_us_timespec(start,end));
	    }
		printf("[N:%d] Fused - CPU finish: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times7);
		printf("\n");

		// -----------------------------//
		// Completely Fused: All on GPU //
		// -----------------------------//
		std::vector<double> times8 = {};
	    for(int iter = 0; iter < TEST_ITERS; iter++){
	    	clock_gettime(CLOCK_MONOTONIC,&start);
		    gpuErrchk(cudaMemcpy(d_mem_vol_completely_fused,h_mem_vol_completely_fused,COMPLETELY_FUSED_VALS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice));
			dynamicsGradientKernel<T,NUM_TIME_STEPS_TEST,MPC_MODE,VEL_DAMPING><<<NUM_TIME_STEPS_TEST,BlockDimms>>>(d_dqdd,d_mem_vol_completely_fused,d_mem_const);
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaMemcpy(h_dqdd,d_dqdd,2*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyDeviceToHost));
			clock_gettime(CLOCK_MONOTONIC,&end);
			times8.push_back(time_delta_us_timespec(start,end));
		}
		printf("[N:%d] Completely Fused - GPU Compute + I/O: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times8);
		
		std::vector<double> times9= {};
	    for(int iter = 0; iter < TEST_ITERS; iter++){
	    	clock_gettime(CLOCK_MONOTONIC,&start);
		    dynamicsGradientKernel<T,NUM_TIME_STEPS_TEST,MPC_MODE,VEL_DAMPING><<<NUM_TIME_STEPS_TEST,BlockDimms>>>(d_dqdd,d_mem_vol_completely_fused,d_mem_const);
			gpuErrchk(cudaDeviceSynchronize());
			clock_gettime(CLOCK_MONOTONIC,&end);
			times9.push_back(time_delta_us_timespec(start,end));
		}
		printf("[N:%d] Completely Fused - GPU Compute: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times9);
		printf("\n");
		printf("----------------------------------------\n");
	}

    // free all
	free(h_vaf); free(h_cdu); free(h_dqdd); free(h_mem_const); delete testTraj;
	free(h_mem_vol_split); free(h_Minv); free(h_mem_vol_fused); free(h_mem_vol_completely_fused);
	gpuErrchk(cudaFree(d_dqdd)); gpuErrchk(cudaFree(d_cdu)); gpuErrchk(cudaFree(d_mem_const));
	gpuErrchk(cudaFree(d_mem_vol_split)); gpuErrchk(cudaFree(d_mem_vol_fused)); gpuErrchk(cudaFree(d_mem_vol_completely_fused));
	for(int i=0; i<NUM_STREAMS; i++){gpuErrchk(cudaStreamDestroy(streams[i]));} free(streams);
}

int main(void){
	test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,1,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL>();
	#if !TEST_FOR_EQUIVALENCE
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,10,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL>(); 
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,16,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL>();
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,32,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL>();
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,64,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL>();
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,128,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL>();
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,256,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL>();
	#endif
}