#include <iostream>
#include <cstdlib>
#include <sstream>
#include <exception>

#include <ros/ros.h>

#include <robo_lib/kinematic_chain.h>

using namespace as64_;

void printTitle(const std::string &title, char s='*', const std::string &color="blue")
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


int main(int argc, char **argv)
{
	ros::init(argc, argv, "kinematic_chain_test_node");

	robo_::KinematicChain chain("robot_description", "robot_left_arm_base_link", "robot_left_arm_zed_link");

	chain.setInvKinematicsSolver(robo_::LMA);

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