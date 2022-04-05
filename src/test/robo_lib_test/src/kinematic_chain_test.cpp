#include <iostream>
#include <cstdlib>
#include <map>
#include <sstream>
#include <fstream>
#include <streambuf>
#include <exception>

#include <ros/ros.h>
#include <ros/package.h>

#include <robo_lib/kinematic_chain.h>

using namespace as64_;

void printTitle(const std::string &title, char s='*', const std::string &color="blue");

// ******************************
// ****  Test ARMADILLO api  ****
// ******************************

void test_armadillo(const robo_::KinematicChain &chain)
{
	arma::vec j_pos = {-0.22, -0.7, 1.63, -4, -1.1, 0.5};

	printTitle("Forward kinematics");
	{
		arma::mat pose = chain.getPose(j_pos);
		arma::vec pos = chain.getPosition(j_pos);
		arma::mat R = chain.getRotm(j_pos);
		arma::vec quat = chain.getQuat(j_pos);
		arma::mat jacob = chain.getJacobian(j_pos);

		std::cout << "pose:\n" <<  pose << "\n";
		std::cout << "pos:\n" <<  pos.t() << "\n";
		std::cout << "R:\n" <<  R << "\n";
		std::cout << "quat:\n" <<  quat.t() << "\n";
		std::cout << "jacob:\n" <<  jacob << "\n";
	}

	printTitle("Inverse kinematics");
	{
		arma::vec j_pos = {-0.22, -0.7, 1.63, -4, -1.1, 0.5};
		arma::mat pose = chain.getPose(j_pos);

		arma::vec j_pos_0 = j_pos + 0.6 * arma::vec().randu(chain.getNumOfJoints());
		bool found_sol;
		arma::vec j_pos_hat = chain.getJointPositions(pose, j_pos_0, &found_sol);
		arma::mat pose_hat = chain.getPose(j_pos_hat);

		std::cout << "found solution = " << found_sol << "\n";
		std::cout << "pose:\n" << pose << "\n";
		std::cout << "pose_hat:\n" << pose_hat << "\n";
		std::cout << "j_pos:\n" << j_pos.t() << "\n";
		std::cout << "j_pos_hat:\n" << j_pos_hat.t() << "\n";
	}
	
	printTitle("Forward kinematics for another link in the chain");
	{
		std::string link_name = "robot_left_arm_wrist_2_link";
    arma::mat pose = chain.getPose(j_pos, link_name);
    arma::vec pos = chain.getPosition(j_pos, link_name);
    arma::mat R = chain.getRotm(j_pos, link_name);
    arma::vec quat = chain.getQuat(j_pos, link_name);
    std::cout << "link: " << link_name << "\n";
    std::cout << "pos:\n" << pos << "\n";
    std::cout << "R:\n" << R << "\n";
    std::cout << "quat:\n" << quat.t() << "\n";
	}
}

// **************************
// ****  Test Eigen api  ****
// **************************

void test_eigen(const robo_::KinematicChain &chain)
{
	Eigen::VectorXd j_pos(chain.getNumOfJoints());
	j_pos << -0.22, -0.7, 1.63, -4, -1.1, 0.5;

	printTitle("Forward kinematics");
	{
		Eigen::Matrix4d pose = chain.getPose(j_pos);
		Eigen::Vector3d pos = chain.getPosition(j_pos);
		Eigen::Matrix3d R = chain.getRotm(j_pos);
		Eigen::Vector4d quat = chain.getQuat(j_pos);
		Eigen::MatrixXd jacob = chain.getJacobian(j_pos);

		std::cout << "pose:\n" <<  pose << "\n";
		std::cout << "pos:\n" <<  pos.transpose() << "\n";
		std::cout << "R:\n" <<  R << "\n";
		std::cout << "quat:\n" <<  quat.transpose() << "\n";
		std::cout << "jacob:\n" <<  jacob << "\n";
	}

	printTitle("Inverse kinematics");
	{
		Eigen::VectorXd j_pos(chain.getNumOfJoints());
		j_pos << -0.22, -0.7, 1.63, -4, -1.1, 0.5;
		Eigen::Matrix4d pose = chain.getPose(j_pos);

		Eigen::VectorXd j_pos_0 = j_pos + 0.6 * Eigen::VectorXd().Random(chain.getNumOfJoints());
		bool found_sol;
		Eigen::VectorXd j_pos_hat = chain.getJointPositions(pose, j_pos_0, &found_sol);
		Eigen::MatrixXd pose_hat = chain.getPose(j_pos_hat);

		std::cout << "found solution = " << found_sol << "\n";
		std::cout << "pose:\n" << pose << "\n";
		std::cout << "pose_hat:\n" << pose_hat << "\n";
		std::cout << "j_pos:\n" << j_pos.transpose() << "\n";
		std::cout << "j_pos_hat:\n" << j_pos_hat.transpose() << "\n";
	}
	
	printTitle("Forward kinematics for another link in the chain");
	{
		std::string link_name = "robot_left_arm_wrist_2_link";
    Eigen::Matrix4d pose = chain.getPose(j_pos, link_name);
    Eigen::Vector3d pos = chain.getPosition(j_pos, link_name);
    Eigen::Matrix3d R = chain.getRotm(j_pos, link_name);
    Eigen::Vector4d quat = chain.getQuat(j_pos, link_name);
    std::cout << "link: " << link_name << "\n";
    std::cout << "pos:\n" << pos << "\n";
    std::cout << "R:\n" << R << "\n";
    std::cout << "quat:\n" << quat.transpose() << "\n";
	}
}

// ******************
// ****  params  ****
// ******************
std::string init_from;
robo_::IK_POS_SOLVER ik_pos_solver;
robo_::IK_VEL_SOLVER ik_vel_solver;
int max_iter;
double err_tol;

void loadParams();

// ******************
// ****   MAIN   ****
// ******************

int main(int argc, char **argv)
{
	ros::init(argc, argv, "kinematic_chain_test_node");

	// ========= Load params ============
	loadParams();

	// ========== Initialize chain =============
	robo_::KinematicChain chain;

	std::string urdf_filename = ros::package::getPath("robo_lib_test") + "/urdf/robot_description.urdf";
	
	if (init_from.compare("file") == 0)
		chain.initFromFile(urdf_filename, "robot_left_arm_base_link", "robot_left_arm_zed_link");
	else if (init_from.compare("param") == 0) 
		chain.initFromParam("robot_description", "robot_left_arm_base_link", "robot_left_arm_zed_link");
	else if (init_from.compare("string") == 0)
	{
		std::ifstream ifs(urdf_filename); // open file
		std::string urdf_xml_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>()); // read all file contents
		chain.initFromXmlString(urdf_xml_str, "robot_left_arm_base_link", "robot_left_arm_zed_link");
	}
		
	// ========  Set inverse kinematic solver options ==========
	chain.setInvKinematicsSolver(ik_pos_solver, ik_vel_solver, max_iter, err_tol);

	int n_joints = chain.getNumOfJoints();

	printTitle("Links");
	for (auto name : chain.link_names) std::cout << name << "\n";

	printTitle("Joints");
	for (auto name : chain.joint_names) std::cout << name << "\n";

	printTitle("Joint Limits");
	std::cout << "Lower limits: " <<  chain.getJointPosLowerLim().t();
	std::cout << "Upper limits: " <<  chain.getJointPosUpperLim().t();
	
	printTitle("ARMADILLO", '#', "green");
	test_armadillo(chain);

	printTitle("EIGEN", '#', "green");
	test_eigen(chain); 

	return 0;
}

// ***************************
// ***  Utility functions  ***
// ***************************

void printTitle(const std::string &title, char s, const std::string &color)
{
	std::string color_;
	if (color.compare("blue") == 0) color_ = "\033[1m\033[34m";
	else if (color.compare("green") == 0) color_ = "\033[1m\033[32m";
	else throw std::runtime_error("Unsupported color: " + color + "...\n");

	std::ostringstream oss;
	oss << std::string(4, s) << "  " << title << "  " << std::string(4, s);
	std::string title_stars = oss.str();
	std::string stars_str(title_stars.size(), s);

	std::cout << color_
						<< stars_str << "\n"
						<< title_stars << "\n"
						<< stars_str << "\n"
						<< "\033[0m" << std::flush;

}

void loadParams()
{
	ros::NodeHandle nh("~");
	
	if (!nh.getParam("init_from", init_from)) throw std::runtime_error("Failed to load param 'init_from'...");
	if (init_from.compare("file") && init_from.compare("param") && init_from.compare("string"))
		throw std::runtime_error("Unsupported init_from: '" + init_from + "'...\n");

	std::string ik_pos_solver_str; // "LMA", "NR", "NR_JL"
	std::string ik_vel_solver_str; // "PINV", "PINV_GIVENS", "PINV_NSO", "WDLS"
	int max_iter;
	double err_tol;
	if (!nh.getParam("ik_pos_solver", ik_pos_solver_str)) throw std::runtime_error("Failed to load param 'ik_pos_solver'...");
	if (!nh.getParam("ik_vel_solver", ik_vel_solver_str)) throw std::runtime_error("Failed to load param 'ik_vel_solver'...");
	if (!nh.getParam("max_iter", max_iter)) throw std::runtime_error("Failed to load param 'max_iter'...");
	if (!nh.getParam("err_tol", err_tol)) throw std::runtime_error("Failed to load param 'err_tol'...");

	
	if (ik_pos_solver_str.compare("LMA") == 0) ik_pos_solver = robo_::IK_POS_SOLVER::LMA;
	else if (ik_pos_solver_str.compare("NR") == 0) ik_pos_solver = robo_::IK_POS_SOLVER::NR;
	else if (ik_pos_solver_str.compare("NR_JL") == 0) ik_pos_solver = robo_::IK_POS_SOLVER::NR_JL;
	else throw std::runtime_error("Unsupported ik_pos_solver: '" + ik_pos_solver_str + "'...\n");

	if (ik_vel_solver_str.compare("PINV") == 0) ik_vel_solver = robo_::IK_VEL_SOLVER::PINV;
	else if (ik_vel_solver_str.compare("PINV_GIVENS") == 0) ik_vel_solver = robo_::IK_VEL_SOLVER::PINV_GIVENS;
	else if (ik_vel_solver_str.compare("PINV_NSO") == 0) ik_vel_solver = robo_::IK_VEL_SOLVER::PINV_NSO;
	else if (ik_vel_solver_str.compare("WDLS") == 0) ik_vel_solver = robo_::IK_VEL_SOLVER::WDLS;
	else throw std::runtime_error("Unsupported ik_vel_solver: '" + ik_vel_solver_str + "'...\n");
}