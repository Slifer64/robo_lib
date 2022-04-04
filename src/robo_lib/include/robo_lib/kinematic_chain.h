#ifndef ROBO_LIB_KINEMATIC_CHAIN_H
#define ROBO_LIB_KINEMATIC_CHAIN_H

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <memory>
#include <string>

#include <urdf/model.h>
#include <kdl/chain.hpp> // KDL::Chain
#include <kdl_parser/kdl_parser.hpp> // KDL::Tree
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainiksolver.hpp>

#include <armadillo>
#include <Eigen/Dense>

namespace as64_
{

namespace robo_
{

enum IK_POS_SOLVER
{
  NR,    // inverse position kinematics algorithm based on Newton-Raphson
  NR_JL, // inverse position kinematics algorithm based on Newton-Raphson that takes joint limits into account
  LMA    // inverse position kinematics that uses Levenberg-Marquardt
};

enum IK_VEL_SOLVER
{
  PINV,        // inverse velocity kinematics algorithm based on the generalized pseudo inverse
  PINV_GIVENS,
  PINV_NSO,  
  WDLS        // inverse velocity kinematics algorithm based on the weighted pseudo inverse with damped least-square
};

class KinematicChain
{
public:
  KinematicChain(urdf::Model &urdf_model, const std::string &base_link, const std::string &tool_link);
  KinematicChain(const std::string &robot_desc_param, const std::string &base_link, const std::string &tool_link);
  ~KinematicChain();

  void setInvKinematicsSolver(IK_POS_SOLVER ik_pos_solver_, IK_VEL_SOLVER ik_vel_solver_=IK_VEL_SOLVER::PINV, int max_iter=100, double err_tol=1e-5);

  int getNumOfJoints() const;

  arma::vec getJointPosLowerLim() const { return arma::vec(joint_pos_lower_lim); }
  arma::vec getJointPosUpperLim() const { return arma::vec(joint_pos_upper_lim); }

  // =========== Armadillo interface ===========

  arma::vec getJointPositions(const arma::mat &pose, const arma::vec &q0, bool *found_solution=NULL) const;
  arma::vec getPosition(const arma::vec &j_pos, const std::string &link_name="") const;
  arma::vec getQuat(const arma::vec &j_pos, const std::string &link_name="") const;
  arma::mat getRotm(const arma::vec &j_pos, const std::string &link_name="") const;
  arma::mat getPose(const arma::vec &j_pos, const std::string &link_name="") const;
  arma::mat getJacobian(const arma::vec j_pos) const;

  // =========== Eigen interface ===========

  Eigen::VectorXd getJointPositions(const Eigen::MatrixXd &pose, const Eigen::VectorXd &q0, bool *found_solution=NULL) const;
  Eigen::Vector3d getPosition(const Eigen::VectorXd &j_pos, const std::string &link_name="") const;
  Eigen::Vector4d getQuat(const Eigen::VectorXd &j_pos, const std::string &link_name="") const;
  Eigen::Matrix3d getRotm(const Eigen::VectorXd &j_pos, const std::string &link_name="") const;
  Eigen::Matrix4d getPose(const Eigen::VectorXd &j_pos, const std::string &link_name="") const;
  Eigen::MatrixXd getJacobian(const Eigen::VectorXd j_pos) const;

  std::vector<std::string> joint_names;
  std::vector<std::string> link_names;
  std::vector<double> joint_pos_lower_lim;
  std::vector<double> joint_pos_upper_lim;
  std::vector<double> joint_vel_lim;
  std::vector<double> effort_lim;

  const KDL::Tree getKdlTree() const { return tree; }

private:

  template<typename T>
  static KDL::JntArray to_JntArray(const T &a)
  {
    KDL::JntArray q(a.size());
    for (int i=0; i<a.size(); i++) q(i) = a[i];
    return q;
  }

  void init();

  int N_JOINTS;

  urdf::Model urdf_model;
  std::shared_ptr<KDL::ChainFkSolverPos_recursive> fk_solver;
  std::shared_ptr<KDL::ChainIkSolverVel> ik_vel_solver;//Inverse velocity solver
  std::shared_ptr<KDL::ChainIkSolverPos> ik_solver;
  std::shared_ptr<KDL::ChainJntToJacSolver> jac_solver;

  KDL::Chain chain;
  KDL::Tree tree;

  std::map<std::string, int> link_id_map;

  std::string base_link_name;
  std::string tool_link_name;
};

}; // namespace robo_

}; // namespace as64_

#endif // ROBO_LIB_KINEMATIC_CHAIN_H
