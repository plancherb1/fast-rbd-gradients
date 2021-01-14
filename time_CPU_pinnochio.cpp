/***
clang++-10 -std=c++11 -o pinnochio_timing.exe time_CPU_pinnochio.cpp -O3 -DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR -DPINOCCHIO_WITH_URDFDOM -lboost_system -L/opt/openrobots/lib -lpinocchio -lurdfdom_model -lpthread
***/
#include "utils/experiment_helpers.h" // include constants and other experiment consistency helpers

#define BOOST_BIND_GLOBAL_PLACEHOLDERS // fixes a warning I get

#include "pinocchio/spatial/fwd.hpp"
#include "pinocchio/spatial/se3.hpp"
#include "pinocchio/multibody/visitor.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/container/aligned-vector.hpp"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Dense>
#include <Eigen/StdVector>
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::VectorXd)

using namespace Eigen;
using namespace pinocchio;

void dynamicsGradientThreaded_inner(Model *model, Data *datas, MatrixXd *dqdd_dqs, MatrixXd *dqdd_dqds, \
                                    VectorXd *qs, VectorXd *qds, VectorXd *qdds, MatrixXd *Minvs, int tid, int kStart, int kMax){
   MatrixXd drnea_dq = MatrixXd::Zero(model->nq,model->nq);
   MatrixXd drnea_dv = MatrixXd::Zero(model->nv,model->nv);
   MatrixXd drnea_da = MatrixXd::Zero(model->nv,model->nv);
   for(int k = kStart; k < kMax; k++){
      computeRNEADerivatives(*model,datas[tid],qs[k],qds[k],qdds[k],drnea_dq,drnea_dv,drnea_da);
      dqdd_dqs[k] = -Minvs[k]*drnea_dq;
      dqdd_dqds[k] = -Minvs[k]*drnea_dv;
   }
}

template<int NUM_THREADS, int NUM_TIME_STEPS>
void dynamicsGradientReusableThreaded(Model *model, Data *datas, MatrixXd *dqdd_dqs, MatrixXd *dqdd_dqds, \
                                    VectorXd *qs, VectorXd *qds, VectorXd *qdds, MatrixXd *Minvs, ReusableThreads<NUM_THREADS> *threads){
      for (int tid = 0; tid < NUM_THREADS; tid++){
         int kStart = NUM_TIME_STEPS/NUM_THREADS*tid; int kMax = NUM_TIME_STEPS/NUM_THREADS*(tid+1); 
         if(tid == NUM_THREADS-1){kMax = NUM_TIME_STEPS;} 
         threads->addTask(tid, dynamicsGradientThreaded_inner, std::ref(model), std::ref(datas), std::ref(dqdd_dqs), std::ref(dqdd_dqds), 
                                                   std::ref(qs), std::ref(qds), std::ref(qdds), std::ref(Minvs), tid, kStart, kMax);
      }
      threads->sync();
}

template<int TEST_ITERS, int NUM_THREADS_TEST, int NUM_TIME_STEPS_TEST>
void test(){
   // Setup timer
   struct timespec start, end;

   // Import URDF model and prepare pinnochio
   Model model;
   std::string urdf_filename = "utils/iiwa_14.urdf";
   pinocchio::urdf::buildModel(urdf_filename,model);
   // model.gravity.setZero();
   model.gravity.linear(Eigen::Vector3d(0,0,-9.81));
   Data datas[NUM_THREADS_TEST]; 
   for(int i = 0; i < NUM_THREADS_TEST; i++){datas[i] = Data(model);}

   // allocate and load on CPU
   VectorXd qs[NUM_TIME_STEPS_TEST];
   VectorXd qds[NUM_TIME_STEPS_TEST];
   VectorXd qdds[NUM_TIME_STEPS_TEST];
   VectorXd us[NUM_TIME_STEPS_TEST];
   MatrixXd Minvs[NUM_TIME_STEPS_TEST];
   MatrixXd dqdd_dqs[NUM_TIME_STEPS_TEST];
   MatrixXd dqdd_dqds[NUM_TIME_STEPS_TEST];
   for(int i = 0; i < NUM_TIME_STEPS_TEST; i++){
      qs[i] = VectorXd::Zero(model.nq);
      qds[i] = VectorXd::Zero(model.nv);
      qdds[i] = VectorXd::Zero(model.nv);
      us[i] = VectorXd::Zero(model.nv);
      Minvs[i] = MatrixXd::Zero(model.nq,model.nq);
      dqdd_dqs[i] = MatrixXd::Zero(model.nv,model.nq);
      dqdd_dqds[i] = MatrixXd::Zero(model.nv,model.nv);
      for(int j = 0; j < model.nq; j++){qs[i][j] = getRand<double>(); qds[i][j] = getRand<double>(); us[i][j] = getRand<double>();}
      aba(model,datas[0],qs[i],qds[i],us[i]);
      computeMinverse(model,datas[0],qs[i]); 
      datas[0].Minv.template triangularView<Eigen::StrictlyLower>() = datas[0].Minv.transpose().template triangularView<Eigen::StrictlyLower>();
      qdds[i] = datas[0].ddq;
      Minvs[i] = datas[0].Minv;
   }

   // Single call
   if(NUM_TIME_STEPS_TEST == 1){
      #if TEST_FOR_EQUIVALENCE
         int SINGLE_TEST_ITERS = TEST_ITERS*1;
      #else
         int SINGLE_TEST_ITERS = TEST_ITERS*10;
      #endif
      MatrixXd drnea_dq = MatrixXd::Zero(model.nq,model.nq);
      MatrixXd drnea_dv = MatrixXd::Zero(model.nv,model.nv);
      MatrixXd drnea_da = MatrixXd::Zero(model.nv,model.nv);
      
      clock_gettime(CLOCK_MONOTONIC,&start);
      for(int i = 0; i < SINGLE_TEST_ITERS; i++){int ind = 0;
         computeRNEADerivatives(model,datas[0],qs[ind],qds[ind],qdds[ind],drnea_dq,drnea_dv,drnea_da);
         dqdd_dqs[ind] = -Minvs[ind]*drnea_dq;
         dqdd_dqds[ind] = -Minvs[ind]*drnea_dv;

         #if TEST_FOR_EQUIVALENCE
            std::cout << "q,qd,qdd,u" << std::endl;
            std::cout << qs[0].transpose() << std::endl;
            std::cout << qds[0].transpose() << std::endl;
            std::cout << qdds[0].transpose() << std::endl;
            std::cout << us[0].transpose() << std::endl;
            std::cout << "Minv" << std::endl << Minvs[0] << std::endl;
            std::cout << "dRNEA_dq" << std::endl << drnea_dq << std::endl;
            std::cout << "dRNEA_dqd" << std::endl << drnea_dv << std::endl;
            std::cout << "dqdd_dq" << std::endl << dqdd_dqs[0] << std::endl;
            std::cout << "dqdd_dqd" << std::endl << dqdd_dqds[0] << std::endl;
         #endif
      }
      clock_gettime(CLOCK_MONOTONIC,&end);
      printf("Single Call vaf+dc/du+dqdd/du %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
      
      clock_gettime(CLOCK_MONOTONIC,&start);
      for(int i = 0; i < SINGLE_TEST_ITERS; i++){int ind = 0;
         computeRNEADerivatives(model,datas[0],qs[ind],qds[ind],qdds[ind],drnea_dq,drnea_dv,drnea_da);
      }
      clock_gettime(CLOCK_MONOTONIC,&end);
      printf("Single Call vaf+dc/du %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
      
      clock_gettime(CLOCK_MONOTONIC,&start);
      for(int i = 0; i < SINGLE_TEST_ITERS; i++){int ind = 0;
         rnea(model,datas[0],qs[ind],qds[ind],qdds[ind]);
      }
      clock_gettime(CLOCK_MONOTONIC,&end);
      printf("Single Call vaf %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
      
      clock_gettime(CLOCK_MONOTONIC,&start);
      for(int i = 0; i < SINGLE_TEST_ITERS; i++){int ind = 0;
         dqdd_dqs[ind] = -Minvs[ind]*drnea_dq;
         dqdd_dqds[ind] = -Minvs[ind]*drnea_dv;
      }
      clock_gettime(CLOCK_MONOTONIC,&end);
      printf("Single Call dqdd/du %fus\n",time_delta_us_timespec(start,end)/static_cast<double>(SINGLE_TEST_ITERS));
      printf("----------------------------------------\n");
   }
   // multi call with threadPools
   else{
      ReusableThreads<NUM_THREADS_TEST> threads;
      std::vector<double> times = {};
      for(int iter = 0; iter < TEST_ITERS; iter++){
         clock_gettime(CLOCK_MONOTONIC,&start);
         dynamicsGradientReusableThreaded<NUM_THREADS_TEST,NUM_TIME_STEPS_TEST>(&model,datas,dqdd_dqs,dqdd_dqds,qs,qds,qdds,Minvs,&threads);
         clock_gettime(CLOCK_MONOTONIC,&end);
         times.push_back(time_delta_us_timespec(start,end));
      }
      printf("[N:%d]: CPU ReusableThreaded: ",NUM_TIME_STEPS_TEST); printStats<PRINT_DISTRIBUTIONS_GLOBAL>(&times);
      printf("----------------------------------------\n");
   }
}

int main(int argc, const char ** argv)
{
   test<TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,1>();
   #if !TEST_FOR_EQUIVALENCE
      test<TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,10>();
      test<TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,16>();
      test<TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,32>();
      test<TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,64>();
      test<TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,128>();
      test<TEST_ITERS_GLOBAL,CPU_THREADS_GLOBAL,256>();
   #endif
   return 0;
}
