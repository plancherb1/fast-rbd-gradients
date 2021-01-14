/*
clang++-10 -std=c++11 -o testMinv2.exe testMinv2.cpp -O3 -DPINOCCHIO_URDFDOM_TYPEDEF_SHARED_PTR -DPINOCCHIO_WITH_URDFDOM -lboost_system -L/opt/openrobots/lib -lpinocchio -lurdfdom_model -lpthread
*/

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

#define TEST_FOR_EQUIVALENCE 1
#include "../utils/experiment_helpers.h"

int main(void){
	// compare CPU Minvs
	Model model;
	std::string urdf_filename = "../utils/iiwa_14.urdf";
	pinocchio::urdf::buildModel(urdf_filename,model);
	// model.gravity.setZero();
	model.gravity.linear(Eigen::Vector3d(0,0,-9.81));
	Data data = Data(model);
	VectorXd q = VectorXd::Zero(model.nq);
	VectorXd qd = VectorXd::Zero(model.nq);
	VectorXd u = VectorXd::Zero(model.nq);
	MatrixXd Minv = MatrixXd::Zero(model.nq,model.nq);
	for(int j = 0; j < model.nq; j++){
		q[j] = static_cast<double>(getRand<float>()); 
		qd[j] = static_cast<double>(getRand<float>());
		u[j] = static_cast<double>(getRand<float>());
	}
    computeMinverse(model,data,q); 
    data.Minv.template triangularView<Eigen::StrictlyLower>() = data.Minv.transpose().template triangularView<Eigen::StrictlyLower>();
    for(int r = 0; r < model.nq; r++){
    	for(int c = 0; c < model.nq; c++){
			printf("%f ",data.Minv(r,c));
		}
		printf("\n");
	}
}