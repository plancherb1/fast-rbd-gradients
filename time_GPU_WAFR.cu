/***
nvcc -std=c++11 -o WAFR_timing.exe time_GPU_WAFR.cu -gencode arch=compute_75,code=sm_75 -O3
***/

#include "utils/experiment_helpers.h" // include constants and other experiment consistency helpers
#include "helpers_WAFR/dynamics_WAFR.cuh" // for GPU dynamicsGradient

#if TEST_FOR_EQUIVALENCE
	dim3 dimms(1,1);
#else
	dim3 dimms(16,16);
#endif

template<typename T, int TEST_ITERS, int NUM_TIME_STEPS_TEST, bool VEL_DAMPING = false>
void test(){
	// allocate and load on CPU
	T *h_qdd = (T *)malloc(NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T));
	T *h_dqdd = (T *)malloc(3*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T));
	T *h_dqdd2 = (T *)malloc(3*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T));
	T *h_Minv = (T *)malloc(NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T));
	T *h_x = (T *)malloc(STATE_SIZE*NUM_TIME_STEPS_TEST*sizeof(T));
	T *h_u = (T *)malloc(CONTROL_SIZE*NUM_TIME_STEPS_TEST*sizeof(T));
	T *h_I = (T *)malloc(36*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)); initI(h_I);
	T *h_T = (T *)malloc(36*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)); initT(h_T);
	T *h_mem_vol = (T *)malloc((4*NUM_POS + NUM_POS*NUM_POS)*NUM_TIME_STEPS_TEST*sizeof(T)); // x,u,qdd,Minv
	T *h_mem_const = (T *)malloc(72*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)); // I,Tbody
	for (int k = 0; k < NUM_TIME_STEPS_TEST; k++){
		T *h_qddk = &h_qdd[k*NUM_POS];
		T *h_xk = &h_x[k*2*NUM_POS];
		T *h_uk = &h_u[k*NUM_POS];
		T *h_Minvk = &h_Minv[k*NUM_POS*NUM_POS];
		T *h_mem_volk = &h_mem_vol[k*(4*NUM_POS + NUM_POS*NUM_POS)];
		for(int i = 0; i < NUM_POS; i++){h_xk[i] = getRand<T>(); h_xk[i+NUM_POS] = getRand<T>(); h_uk[i] = getRand<T>();}
		dynamicsMinv<T,VEL_DAMPING>(h_qddk,h_xk,h_uk,h_I,h_T,h_Minvk);
		for(int i = 0; i < 4*NUM_POS + NUM_POS*NUM_POS; i++){
			if (i < 2*NUM_POS){h_mem_volk[i] = h_xk[i];}
			else if (i < 3*NUM_POS){h_mem_volk[i] = h_uk[i-2*NUM_POS];}
			else if (i < 4*NUM_POS){h_mem_volk[i] = h_qddk[i-3*NUM_POS];}
			else {h_mem_vol[i] = h_Minvk[i-4*NUM_POS];}
		}
	}
	memcpy(h_mem_const,h_I,36*NUM_POS*sizeof(T));
	memcpy(&h_mem_const[36*NUM_POS],h_T,36*NUM_POS*sizeof(T));

	// allocate and copy to GPU
	T *d_dqdd; gpuErrchk(cudaMalloc((void**)&d_dqdd, 3*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_dqdd2; gpuErrchk(cudaMalloc((void**)&d_dqdd2, 3*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_Minv; gpuErrchk(cudaMalloc((void**)&d_Minv, NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_x; gpuErrchk(cudaMalloc((void**)&d_x, 2*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_u; gpuErrchk(cudaMalloc((void**)&d_u, NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_qdd; gpuErrchk(cudaMalloc((void**)&d_qdd, NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_I; gpuErrchk(cudaMalloc((void**)&d_I, 36*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_T; gpuErrchk(cudaMalloc((void**)&d_T, 36*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_mem_vol; gpuErrchk(cudaMalloc((void**)&d_mem_vol, (4*NUM_POS + NUM_POS*NUM_POS)*NUM_TIME_STEPS_TEST*sizeof(T)));
	T *d_mem_const; gpuErrchk(cudaMalloc((void**)&d_mem_const, 72*NUM_POS*sizeof(T)));
	cudaStream_t *streams = (cudaStream_t *)malloc(NUM_STREAMS*sizeof(cudaStream_t));
	int priority, minPriority, maxPriority;
	gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
	for(int i=0; i<NUM_STREAMS; i++){priority = std::min(minPriority+i,maxPriority);
		gpuErrchk(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamNonBlocking,priority));
	}
	gpuErrchk(cudaMemcpy(d_I,h_I,36*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_T,h_T,36*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mem_const,h_mem_const,72*NUM_POS*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());

	// time kernel with uncompressed and compressed memory copy
	struct timespec start, end;
	if(NUM_TIME_STEPS_TEST == 1){
		#if TEST_FOR_EQUIVALENCE
			#define SINGLE_TEST_ITERS (TEST_ITERS*1)
		#else
			#define SINGLE_TEST_ITERS (TEST_ITERS*10)
		#endif
		gpuErrchk(cudaMemcpyAsync(d_Minv,h_Minv,NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice, streams[0]));
		gpuErrchk(cudaMemcpyAsync(d_x,h_x,2*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice, streams[1]));
		gpuErrchk(cudaMemcpyAsync(d_u,h_u,NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice, streams[2]));
		gpuErrchk(cudaMemcpyAsync(d_qdd,h_qdd,NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice, streams[3]));
		gpuErrchk(cudaDeviceSynchronize());

		#if TEST_FOR_EQUIVALENCE
			printf("q,qd,qdd,u\n");
			printMat<T,1,NUM_POS>(h_x,1);
			printMat<T,1,NUM_POS>(&h_x[NUM_POS],1);
			printMat<T,1,NUM_POS>(h_qdd,1);
			printMat<T,1,NUM_POS>(h_u,1);
			printf("Minv\n");
			printMat<T,NUM_POS,NUM_POS>(h_Minv,NUM_POS);
			dynamicsGradient_v2<T,VEL_DAMPING>(h_dqdd,h_qdd,h_x,h_u,h_Minv,h_I,h_T);
			printf("dqdd_dq\n");
			printMat<T,NUM_POS,NUM_POS>(h_dqdd,NUM_POS);
			printf("dqdd_dqd\n");
			printMat<T,NUM_POS,NUM_POS>(&h_dqdd[NUM_POS*NUM_POS],NUM_POS);
		#endif


		clock_gettime(CLOCK_MONOTONIC,&start);
		kern_single<T,SINGLE_TEST_ITERS,VEL_DAMPING><<<1,dimms>>>(d_dqdd,d_qdd,d_x,d_u,d_Minv,d_I,d_T);
		gpuErrchk(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("Single Call vaf+dc/du+dqdd/du %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));

		#if TEST_FOR_EQUIVALENCE
			gpuErrchk(cudaMemcpy(h_dqdd,d_dqdd,3*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyDeviceToHost));
			printf("dqdd_dq - kernel\n");
			printMat<T,NUM_POS,NUM_POS>(h_dqdd,NUM_POS);
			printf("dqdd_dqd - kernel\n");
			printMat<T,NUM_POS,NUM_POS>(&h_dqdd[NUM_POS*NUM_POS],NUM_POS);
		#endif

		clock_gettime(CLOCK_MONOTONIC,&start);
		kern_single_vaf_dcdu<T,SINGLE_TEST_ITERS,VEL_DAMPING><<<1,dimms>>>(d_dqdd,d_qdd,d_x,d_u,d_Minv,d_I,d_T);
		gpuErrchk(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("Single Call vaf+dc/du %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
		
		clock_gettime(CLOCK_MONOTONIC,&start);
		kern_single_vaf<T,SINGLE_TEST_ITERS,VEL_DAMPING><<<1,dimms>>>(d_dqdd,d_qdd,d_x,d_u,d_Minv,d_I,d_T);
		gpuErrchk(cudaDeviceSynchronize());
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("Single Call vaf %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
		printf("----------------------------------------\n");
	}
	else{
		std::vector<double> times = {};
	    for(int iter = 0; iter < TEST_ITERS; iter++){
	    	clock_gettime(CLOCK_MONOTONIC,&start);
		    gpuErrchk(cudaMemcpy(d_mem_vol,h_mem_vol,(4*NUM_POS + NUM_POS*NUM_POS)*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyHostToDevice));
			kern<T,NUM_TIME_STEPS_TEST,VEL_DAMPING><<<NUM_TIME_STEPS_TEST,dimms>>>(d_dqdd2,d_mem_vol,d_mem_const);
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaMemcpy(h_dqdd2,d_dqdd2,3*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T),cudaMemcpyDeviceToHost));
			clock_gettime(CLOCK_MONOTONIC,&end);
			times.push_back(time_delta_us_timespec(start,end));
		}
		printf("[N=%d] GPU Compute + I/O: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times);
		std::vector<double> times2 = {};
	    for(int iter = 0; iter < TEST_ITERS; iter++){
	    	clock_gettime(CLOCK_MONOTONIC,&start);
		    kern<T,NUM_TIME_STEPS_TEST,VEL_DAMPING><<<NUM_TIME_STEPS_TEST,dimms>>>(d_dqdd2,d_mem_vol,d_mem_const);
			gpuErrchk(cudaDeviceSynchronize());
			clock_gettime(CLOCK_MONOTONIC,&end);
			times2.push_back(time_delta_us_timespec(start,end));
		}
		printf("[N:%d] GPU Compute: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times2);
		printf("----------------------------------------\n");
	}

    // free all
	free(h_qdd);
	free(h_dqdd);
	free(h_dqdd2);
	free(h_Minv);
	free(h_x);
	free(h_u);
	free(h_I);
	free(h_T);
	gpuErrchk(cudaFree(d_qdd));
	gpuErrchk(cudaFree(d_dqdd));
	gpuErrchk(cudaFree(d_Minv));
	gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_u));
	gpuErrchk(cudaFree(d_I));
	gpuErrchk(cudaFree(d_T));
	for(int i=0; i<NUM_STREAMS; i++){gpuErrchk(cudaStreamDestroy(streams[i]));} free(streams);
}

int main(void){
	test<float,TEST_ITERS_GLOBAL,1>();
	#if !TEST_FOR_EQUIVALENCE
		test<float,TEST_ITERS_GLOBAL,10>();
		test<float,TEST_ITERS_GLOBAL,16>();
		test<float,TEST_ITERS_GLOBAL,32>();
		test<float,TEST_ITERS_GLOBAL,64>();
		test<float,TEST_ITERS_GLOBAL,128>();
		test<float,TEST_ITERS_GLOBAL,256>();
	#endif
}
