/***
g++ -std=c++11 -o CPU_timing.exe time_CPU.cpp -lpthread -O3 -march=native -mavx
clang++-10 -std=c++11 time_CPU.cpp -o CPU_clang_timing.exe -pthread -O3 -march=native -mavx
***/

#include "utils/experiment_helpers.h"
#include "helpers_CPU/dynamicsGradient.h"

template <typename T, int TEST_ITERS, bool QDD_MINV_PASSED_IN = false, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void single_full(T *dqdd, knot<T> *currPoint, T *d_T, T *d_I, T *s_fext = nullptr){
	for (int i = 0; i < TEST_ITERS; i++){
		forwardDynamicsGradient<T,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(dqdd,currPoint,d_T,d_I);
	}
}

template <typename T, int TEST_ITERS, bool QDD_MINV_PASSED_IN = false, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void single_vaf_dcdu(T *dcdu, knot<T> *currPoint, T *d_T, T *d_I, T *s_fext = nullptr){
	for (int i = 0; i < TEST_ITERS; i++){
		forwardDynamicsGradient_vaf_dcdu<T,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(dcdu,currPoint,d_T,d_I);
	}
}

template <typename T, int TEST_ITERS, bool QDD_MINV_PASSED_IN = false, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void single_vaf(T *vaf, knot<T> *currPoint, T *d_T, T *d_I, T *s_fext = nullptr){
	for (int i = 0; i < TEST_ITERS; i++){
		forwardDynamicsGradient_vaf<T,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(vaf,currPoint,d_T,d_I);
	}
}

template <typename T, int THREADS, int KNOT_POINTS, bool QDD_MINV_PASSED_IN = false, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void dynamicsGradientThreaded_inner(T *dqdd, traj<T,KNOT_POINTS> *currTraj, T *Ts, T *Is, int tid){
	int kStart = KNOT_POINTS/THREADS*tid; int kMax = KNOT_POINTS/THREADS*(tid+1); if(tid == THREADS-1){kMax = KNOT_POINTS;} int tidInd = 36*NUM_POS*tid;
   	for (int k=kStart; k < kMax; k++){
		forwardDynamicsGradientThreaded<T,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(&dqdd[k*2*NUM_POS*NUM_POS],&(currTraj->knots[k]),&Ts[tidInd],&Is[tidInd]);
   	}
}

template <typename T, int THREADS, int KNOT_POINTS, bool QDD_MINV_PASSED_IN = false, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void dynamicsGradientThreaded(T *dqdd, traj<T,KNOT_POINTS> *currTraj, T *Ts, T *Is, std::thread cpuThreads[THREADS]){
	void (*f)(T*,traj<T,KNOT_POINTS>*,T*,T*,int) = &dynamicsGradientThreaded_inner<T,THREADS,KNOT_POINTS,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>;
    for (int tid = 0; tid < THREADS; tid++){
    	cpuThreads[tid] = std::thread(f, std::ref(dqdd), std::ref(currTraj), std::ref(Ts), std::ref(Is), tid);
    }
    for (int tid = 0; tid < THREADS; tid++){cpuThreads[tid].join();}
}

template<typename T, int THREADS, int KNOT_POINTS, bool QDD_MINV_PASSED_IN = false, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void dynamicsGradientReusableThreaded(T *dqdd, traj<T,KNOT_POINTS> *currTraj, T *Ts, T *Is, ReusableThreads<THREADS> *threads){
	void (*f)(T*,traj<T,KNOT_POINTS>*,T*,T*,int) = &dynamicsGradientThreaded_inner<T,THREADS,KNOT_POINTS,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>;
	for (int tid = 0; tid < THREADS; tid++){
			threads->addTask(tid, f, std::ref(dqdd), std::ref(currTraj), std::ref(Ts), std::ref(Is), tid);
	}
	threads->sync();
}

template<typename T, int TEST_ITERS, int NUM_THREADS_TEST, int NUM_TIME_STEPS_TEST, bool QDD_MINV_PASSED_IN = false, bool MPC_MODE = false, bool VEL_DAMPING = true, bool EXTERNAL_TI = false>
void test(){
	// allocate and load on CPU
	T *dqdd = (T *)malloc(2*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T));
	T *dcdu = (T *)malloc(2*NUM_POS*NUM_POS*NUM_TIME_STEPS_TEST*sizeof(T));
	T *vaf = (T *)malloc(6*NUM_POS*3*NUM_TIME_STEPS_TEST*sizeof(T));
	T *Ts = (T *)malloc(36*NUM_POS*NUM_THREADS_TEST*sizeof(T));
	T *Is = (T *)malloc(36*NUM_POS*NUM_THREADS_TEST*sizeof(T));
	for (int tid = 0; tid < NUM_THREADS_TEST; tid++){initTransforms(&Ts[36*NUM_POS*tid]); initInertiaTensors(&Is[36*NUM_POS*tid]);}
	traj<T,NUM_TIME_STEPS_TEST> *testTraj = new traj<T,NUM_TIME_STEPS_TEST>;
	for (int k = 0; k < NUM_TIME_STEPS_TEST; k++){
		for(int i = 0; i < NUM_POS; i++){
			testTraj->knots[k].q[i] = getRand<T>(); 
			testTraj->knots[k].qd[i] = getRand<T>();
			testTraj->knots[k].u[i] = getRand<T>();
		}
		forwardDynamics<T,MPC_MODE,VEL_DAMPING>(&(testTraj->knots[k]));
	}
	struct timespec start, end;
	
	// time kernel for single call (if needed)
	if(NUM_TIME_STEPS_TEST == 1){
		#if TEST_FOR_EQUIVALENCE
			#define SINGLE_TEST_ITERS (TEST_ITERS*1)
		#else
			#define SINGLE_TEST_ITERS (TEST_ITERS*10)
		#endif
		clock_gettime(CLOCK_MONOTONIC,&start);
		single_full<T,SINGLE_TEST_ITERS,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(dqdd,&(testTraj->knots[0]),Ts,Is);
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("Single Call vaf+dc/du+dqdd/du %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));

		#if TEST_FOR_EQUIVALENCE
			printf("q,qd,qdd,u\n");
			printMat<T,1,NUM_POS>(testTraj->knots[0].q,1);
			printMat<T,1,NUM_POS>(testTraj->knots[0].qd,1);
			printMat<T,1,NUM_POS>(testTraj->knots[0].qdd,1);
			printMat<T,1,NUM_POS>(testTraj->knots[0].u,1);
			printf("Minv\n");
			printMat<T,NUM_POS,NUM_POS>(testTraj->knots[0].Minv,NUM_POS);
			printf("dqdd/dq\n");
			printMat<T,NUM_POS,NUM_POS>(dqdd,NUM_POS);
			printf("dqdd/dqd\n");
			printMat<T,NUM_POS,NUM_POS>(&dqdd[NUM_POS*NUM_POS],NUM_POS);
		#endif

		clock_gettime(CLOCK_MONOTONIC,&start);
		single_vaf_dcdu<T,SINGLE_TEST_ITERS,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(dcdu,&(testTraj->knots[0]),Ts,Is);
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("Single Call vaf+dc/du %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
		
		clock_gettime(CLOCK_MONOTONIC,&start);
		single_vaf<T,SINGLE_TEST_ITERS,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(vaf,&(testTraj->knots[0]),Ts,Is);
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("Single Call vaf %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
		printf("----------------------------------------\n");
	}
	else{
		// time end to end for the given knot points
		std::thread cpuThreads[NUM_THREADS_TEST];
		std::vector<double> times = {};
		for(int iter = 0; iter < TEST_ITERS; iter++){
			clock_gettime(CLOCK_MONOTONIC,&start);
			dynamicsGradientThreaded<T,NUM_THREADS_TEST,NUM_TIME_STEPS_TEST,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(dqdd,testTraj,Ts,Is,cpuThreads);
			clock_gettime(CLOCK_MONOTONIC,&end);
			times.push_back(time_delta_us_timespec(start,end));
		}
		printf("[N:%d]: CPU Std::Threaded: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times);

		ReusableThreads<NUM_THREADS_TEST> threads;
		std::vector<double> times2 = {};
		for(int iter = 0; iter < TEST_ITERS; iter++){
			clock_gettime(CLOCK_MONOTONIC,&start);
			dynamicsGradientReusableThreaded<T,NUM_THREADS_TEST,NUM_TIME_STEPS_TEST,QDD_MINV_PASSED_IN,MPC_MODE,VEL_DAMPING,EXTERNAL_TI>(dqdd,testTraj,Ts,Is,&threads);
			clock_gettime(CLOCK_MONOTONIC,&end);
			times2.push_back(time_delta_us_timespec(start,end));
		}
		clock_gettime(CLOCK_MONOTONIC,&end);
		printf("[N:%d]: CPU ReusableThreaded: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times2);
		printf("----------------------------------------\n");
	}

    // free all
	free(dqdd); free(dcdu); free(vaf); delete testTraj;
}

int main(void){
	test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,1,QDD_MINV_PASSED_IN_GLOBAL,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL,EXTERNAL_TI_GLOBAL>();
	#if !TEST_FOR_EQUIVALENCE
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,10,QDD_MINV_PASSED_IN_GLOBAL,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL,EXTERNAL_TI_GLOBAL>();
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,16,QDD_MINV_PASSED_IN_GLOBAL,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL,EXTERNAL_TI_GLOBAL>();
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,32,QDD_MINV_PASSED_IN_GLOBAL,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL,EXTERNAL_TI_GLOBAL>();
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,64,QDD_MINV_PASSED_IN_GLOBAL,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL,EXTERNAL_TI_GLOBAL>();
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,128,QDD_MINV_PASSED_IN_GLOBAL,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL,EXTERNAL_TI_GLOBAL>();
		test<float,TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,256,QDD_MINV_PASSED_IN_GLOBAL,MPC_MODE_GLOBAL,VEL_DAMPING_GLOBAL,EXTERNAL_TI_GLOBAL>();
	#endif
}
