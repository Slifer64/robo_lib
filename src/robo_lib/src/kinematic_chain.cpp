#include <robo_lib/kinematic_chain.h>

#include <stdexcept>
#include <sstream>
#include <cmath>
#include <stack>

#include <urdf/model.h>

#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolverpos_lma.hpp>

#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolvervel_pinv_givens.hpp>
#include <kdl/chainiksolvervel_pinv_nso.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>

// #include <Eigen/Dense>

namespace as64_
{

namespace robo_
{

#define KinematicChain_fun_ std::string("[KinematicChain::") + __func__ + "]: "

arma::vec rotm2quat(const arma::mat &R)
{
  // Eigen::Map<const Eigen::Matrix3d> rotm_wrapper(rotm.memptr());
  // Eigen::Quaterniond quat(rotm_wrapper);
  // return arma::vec({quat.w(), quat.x(), quat.y(), quat.z()});

  arma::vec q = {1.0, 0, 0, 0};

  double tr = R(0, 0) + R(1, 1) + R(2, 2);
  double s;

  if (tr > 0)
  {
    s = std::sqrt(tr + 1.0) * 2;  // S=4*qwh
    q(0) = 0.25 * s;
    q(1) = (R(2, 1) - R(1, 2)) / s;
    q(2) = (R(0, 2) - R(2, 0)) / s;
    q(3) = (R(1, 0) - R(0, 1)) / s;
  }

  else if (R(0, 0) > R(1, 1)  &&  R(0, 0) > R(2, 2))
  {
    s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2;  // S=4*qx
    q(0) = (R(2, 1) - R(1, 2)) / s;
    q(1) = 0.25 * s;
    q(2) = (R(0, 1) + R(1, 0)) / s;
    q(3) = (R(0, 2) + R(2, 0)) / s;
  }

  else if (R(1, 1) > R(2, 2))
  {
    s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2;  // S=4*qy
    q(0) = (R(0, 2) - R(2, 0)) / s;
    q(1) = (R(0, 1) + R(1, 0)) / s;
    q(2) = 0.25 * s;
    q(3) = (R(1, 2) + R(2, 1)) / s;
  }

  else
  {
    s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2;  // S=4*qz
    q(0) = (R(1, 0) - R(0, 1)) / s;
    q(1) = (R(0, 2) + R(2, 0)) / s;
    q(2) = (R(1, 2) + R(2, 1)) / s;
    q(3) = 0.25 * s;
  }

  return q / arma::norm(q);
}

KinematicChain::KinematicChain(): N_JOINTS(0) {}

KinematicChain::~KinematicChain() {}

void KinematicChain::initFromFile(const std::string &urdf_filename, const std::string &base_link, const std::string &tip_link)
{
  urdf::Model urdf_model;
  if (!urdf_model.initFile(urdf_filename))
    throw std::ios_base::failure("Couldn't load urdf model from file \"" + urdf_filename + "\"...\n");

  this->base_link_name = base_link;
  this->tip_link_name = tip_link;

  init(reinterpret_cast<void *>(&urdf_model));
}

void KinematicChain::initFromParam(const std::string &robot_description_param, const std::string &base_link, const std::string &tip_link)
{
  urdf::Model urdf_model;
  if (!urdf_model.initParam(robot_description_param))
    throw std::ios_base::failure("Couldn't load urdf model from param \"" + robot_description_param + "\"...\n");

  this->base_link_name = base_link;
  this->tip_link_name = tip_link;

  init(reinterpret_cast<void *>(&urdf_model));
}

void KinematicChain::initFromXmlString(const std::string &robot_description_xml_str, const std::string &base_link, const std::string &tip_link)
{
  urdf::Model urdf_model;
  if (!urdf_model.initString(robot_description_xml_str))
    throw std::ios_base::failure("Couldn't load urdf model from xml string \"" + robot_description_xml_str + "\"...\n");

  this->base_link_name = base_link;
  this->tip_link_name = tip_link;

  init(reinterpret_cast<void *>(&urdf_model));
}

void KinematicChain::init(void *urdf_model_ptr)
{
  urdf::Model *urdf_model = reinterpret_cast<urdf::Model *>(urdf_model_ptr);

  auto link_type = urdf_model->getRoot();
  
  // find base_link and tip_link
  bool found_base_link = false;
  bool found_tip_link = false;
  decltype(link_type) base_link;
  decltype(link_type) tip_link;
  std::stack<decltype(link_type)> link_stack;
  link_stack.push(urdf_model->getRoot());
  while (!link_stack.empty())
  {
    auto link = link_stack.top();
    link_stack.pop();

    if (base_link_name.compare(link->name) == 0)
    {
      base_link = link;
      found_base_link = true;
    }

    if (tip_link_name.compare(link->name) == 0)
    {
      tip_link = link;
      found_tip_link = true;
    }

    for (int i=0;i<link->child_links.size();i++) link_stack.push(link->child_links[i]);
    // for (int i=0;i<link->child_joints.size();i++) _joints.push_back(link->child_joints[i]);
  }

  if (!found_base_link)
    throw std::runtime_error("Couldn't find specified base link \"" + base_link_name + "\" in the robot urdf model...\n");

  if (!found_tip_link)
    throw std::runtime_error("Couldn't find specified tool link \"" + tip_link_name + "\" in the robot urdf model...\n");

  // find all links in the chain from tip_link to base_link
  std::vector<decltype(link_type)> chain_links;
  auto link = tip_link;
  while (link->name.compare(base_link->name))
  {
    chain_links.push_back(link);
    link = link->getParent();
    if (!link) throw std::runtime_error("[RobotUrdf Error]: The tool link \"" + tip_link_name + "\" does not belong in the branch of the base link \"" + base_link_name + "\"...");
  }
  chain_links.push_back(base_link);

  // parse all joints for each link starting from base_link
  for (int i=chain_links.size()-1; i>0; i--)
  {
    link = chain_links[i];
    auto next_link = chain_links[i-1];

    for (int i=0;i<link->child_joints.size();i++)
    {
      auto joint = link->child_joints[i];
      auto jtype = joint->type;

      if (jtype==urdf::Joint::FIXED || jtype==urdf::Joint::FLOATING) continue;

      if (joint->mimic) continue;

      if (joint->child_link_name.compare(next_link->name)) continue;

      joint_names.push_back(joint->name);

      if (jtype==urdf::Joint::CONTINUOUS)
      {
        joint_pos_lower_lim.push_back(-M_PI);
        joint_pos_upper_lim.push_back(M_PI);
      }
      else
      {
        joint_pos_lower_lim.push_back(joint->limits->lower);
        joint_pos_upper_lim.push_back(joint->limits->upper);
      }

      effort_lim.push_back(joint->limits->effort);
      joint_vel_lim.push_back(joint->limits->velocity);
    }
  }

  *const_cast<int *>(&N_JOINTS) = joint_pos_lower_lim.size();

  // create KDL::Chain and forward/inverse kinematics and Jacobian solvers
  //KDL::Tree tree;
  kdl_parser::treeFromUrdfModel(*urdf_model, tree);

  if (!tree.getChain(base_link_name, tip_link_name, chain))
    throw std::runtime_error("Failed to create kdl chain from " + base_link_name + " to " + tip_link_name + " ...\n");

  std::vector<KDL::Segment> links(chain.segments);
  for (int i=0; i<links.size(); i++)
  {
    link_names.push_back(links[i].getName());
    link_id_map[links[i].getName()] = i;
  }

  fk_solver.reset(new KDL::ChainFkSolverPos_recursive(chain));
  jac_solver.reset(new KDL::ChainJntToJacSolver(chain));

  setInvKinematicsSolver(IK_POS_SOLVER::NR_JL, IK_VEL_SOLVER::PINV_GIVENS, 100, 1e-5);
}

void KinematicChain::setInvKinematicsSolver(IK_POS_SOLVER ik_pos_solver_, int max_iter, double err_tol)
{
  setInvKinematicsSolver(ik_pos_solver_, IK_VEL_SOLVER::PINV, max_iter, err_tol);
}

void KinematicChain::setInvKinematicsSolver(IK_POS_SOLVER ik_pos_solver_, IK_VEL_SOLVER ik_vel_solver_, int max_iter, double err_tol)
{
  switch (ik_pos_solver_)
  {
  case IK_VEL_SOLVER::PINV:
    ik_vel_solver.reset(new KDL::ChainIkSolverVel_pinv(chain));
    break;
  case IK_VEL_SOLVER::PINV_GIVENS:
    ik_vel_solver.reset(new KDL::ChainIkSolverVel_pinv_givens(chain));
    break;
  case IK_VEL_SOLVER::PINV_NSO:
    ik_vel_solver.reset(new KDL::ChainIkSolverVel_pinv_nso(chain));
    break;
  case IK_VEL_SOLVER::WDLS:
    ik_vel_solver.reset(new KDL::ChainIkSolverVel_wdls(chain));
    break;
  default:
    throw std::runtime_error(KinematicChain_fun_ + "Unsupported ik velocity solver...");
  }

  switch (ik_pos_solver_)
  {
  case IK_POS_SOLVER::NR:
    ik_solver.reset(new KDL::ChainIkSolverPos_NR(chain, *fk_solver, *ik_vel_solver, max_iter, err_tol));
    break;
  case IK_POS_SOLVER::NR_JL:
    ik_solver.reset(new KDL::ChainIkSolverPos_NR_JL(chain, to_JntArray(joint_pos_lower_lim), to_JntArray(joint_pos_upper_lim), *fk_solver, *ik_vel_solver, max_iter, err_tol));
    break;
  case IK_POS_SOLVER::LMA:
    ik_solver.reset(new KDL::ChainIkSolverPos_LMA(chain, err_tol, max_iter));
    break;
  default:
    throw std::runtime_error(KinematicChain_fun_ + "Unsupported ik position solver...");
  }


}

int KinematicChain::getNumOfJoints() const
{
  return N_JOINTS;
}

// ==================================
// ==========  ARMADILLO  ===========
// ==================================

arma::vec KinematicChain::getJointPositions(const arma::mat &pose, const arma::vec &q0, bool *found_solution) const
{
  KDL::JntArray jnt(N_JOINTS);
  KDL::JntArray jnt0 = to_JntArray(q0);

  KDL::Frame kdl_pose;
  for (int i=0;i<3;i++)
  {
    kdl_pose.p[i] = pose(i,3);
    for (int j=0;j<3;j++) kdl_pose.M(i,j) = pose(i,j);
  }

  int ret = ik_solver->CartToJnt(jnt0,kdl_pose,jnt);

  if (found_solution) *found_solution = ret >= 0;

  arma::vec q = arma::vec().zeros(N_JOINTS);

  if (ret>=0)
  {
    for (int i=0;i<N_JOINTS;i++) q(i) = jnt(i);
  }

  return q;
}

arma::mat KinematicChain::getPose(const arma::vec &j_pos, const std::string &link_name) const
{
  int link_ind = -1;
  if (!link_name.empty())
  {
    auto it = link_id_map.find(link_name);
    if (it == link_id_map.end()) throw std::runtime_error(KinematicChain_fun_ + "link '" + link_name + "' is not present in the chain...");
    link_ind = it->second;
  }

  arma::mat task_pose(4,4);

  KDL::JntArray jnt = to_JntArray(j_pos);

  KDL::Frame fk;
  int ret = fk_solver->JntToCart(jnt, fk, link_ind);
  if (ret < 0) throw std::runtime_error(KinematicChain_fun_ + "forward kinematic solver failed...");
  for (int i=0;i<3;i++)
  {
    for (int j=0;j<4;j++) task_pose(i,j) = fk(i,j);
  }
  task_pose.row(3) = arma::rowvec({0,0,0,1});

  return task_pose;
}

arma::vec KinematicChain::getPosition(const arma::vec &j_pos, const std::string &link_name) const
{
  return getPose(j_pos, link_name).submat(0, 3, 2, 3);
}

arma::vec KinematicChain::getQuat(const arma::vec &j_pos, const std::string &link_name) const
{
  return rotm2quat(getRotm(j_pos));
}

arma::mat KinematicChain::getRotm(const arma::vec &j_pos, const std::string &link_name) const
{
  return getPose(j_pos, link_name).submat(0, 0, 2, 2);
}

arma::mat KinematicChain::getJacobian(const arma::vec j_pos) const
{
  KDL::JntArray jnt = to_JntArray(j_pos);

  KDL::Jacobian J(N_JOINTS);
  int ret = jac_solver->JntToJac(jnt, J);
  if (ret < 0) throw std::runtime_error(KinematicChain_fun_ + "Jacobian solver failed...");
  arma::mat Jac(6, N_JOINTS);
  for (int i=0;i<Jac.n_rows;i++)
  {
    for (int j=0;j<Jac.n_cols;j++) Jac(i,j) = J(i,j);
  }

  return Jac;
}

// ===============================
// ==========  EIGEN  ============
// ===============================

#define eigen_vec_as_arma(eig_vec) arma::vec(const_cast<double *>(eig_vec.data()), eig_vec.size(), false)
#define eigen_mat_as_arma(eig_mat) arma::mat(const_cast<double *>(eig_mat.data()), eig_mat.rows(), eig_mat.cols(), false)

Eigen::VectorXd KinematicChain::getJointPositions(const Eigen::MatrixXd &pose, const Eigen::VectorXd &q0, bool *found_solution) const
{
  arma::vec result = getJointPositions(eigen_mat_as_arma(pose), eigen_vec_as_arma(q0), found_solution);
  return Eigen::Map<Eigen::VectorXd>(result.memptr(), result.size());
}

Eigen::Vector3d KinematicChain::getPosition(const Eigen::VectorXd &j_pos, const std::string &link_name) const
{
  arma::vec result = getPosition(eigen_vec_as_arma(j_pos), link_name);
  return Eigen::Map<Eigen::Vector3d>(result.memptr());
}

Eigen::Vector4d KinematicChain::getQuat(const Eigen::VectorXd &j_pos, const std::string &link_name) const
{
  arma::vec result = getQuat(eigen_vec_as_arma(j_pos), link_name);
  return Eigen::Map<Eigen::Vector4d>(result.memptr());
}

Eigen::Matrix3d KinematicChain::getRotm(const Eigen::VectorXd &j_pos, const std::string &link_name) const
{
  arma::mat result = getRotm(eigen_vec_as_arma(j_pos), link_name);
  return Eigen::Map<Eigen::Matrix3d>(result.memptr());
}

Eigen::Matrix4d KinematicChain::getPose(const Eigen::VectorXd &j_pos, const std::string &link_name) const
{
  arma::mat result = getPose(eigen_vec_as_arma(j_pos), link_name);
  return Eigen::Map<Eigen::Matrix4d>(result.memptr());
}

Eigen::MatrixXd KinematicChain::getJacobian(const Eigen::VectorXd j_pos) const
{
  arma::mat result = getJacobian(eigen_vec_as_arma(j_pos));
  return Eigen::Map<Eigen::MatrixXd>(result.memptr(), result.n_rows, result.n_cols);
}


}; // namespace robo_

}; // namespace as64_
