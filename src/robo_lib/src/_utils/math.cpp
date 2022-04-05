
#include <robo_lib/_utils/math.h>

namespace as64_
{
    
namespace robo_
{

arma::vec quatProd(const arma::vec &quat1, const arma::vec &quat2)
{
  arma::vec quat12(4);

  double n1 = quat1(0);
  arma::vec e1 = quat1.subvec(1,3);

  double n2 = quat2(0);
  arma::vec e2 = quat2.subvec(1,3);

  quat12(0) = n1*n2 - arma::dot(e1,e2);
  quat12.subvec(1,3) = n1*e2 + n2*e1 + arma::cross(e1,e2);

  return quat12;
}

arma::vec quatInv(const arma::vec &quat)
{
  arma::vec quatI(4);

  quatI(0) = quat(0);
  quatI.subvec(1,3) = - quat.subvec(1,3);

  return quatI;
}

arma::vec quatExp(const arma::vec &v_rot, double zero_tol)
{
  arma::vec quat(4);
  double norm_v_rot = arma::norm(v_rot);
  double theta = norm_v_rot;

  if (norm_v_rot > zero_tol)
  {
    quat(0) = std::cos(theta/2);
    quat.subvec(1,3) = std::sin(theta/2)*v_rot/norm_v_rot;
  }
  else{
    quat << 1 << 0 << 0 << 0;
  }

  return quat;
}

arma::vec quatLog(const arma::vec &quat, double zero_tol)
{
  arma::vec v = quat.subvec(1,3);
  double u = quat(0);

  arma::vec omega(3);
  double v_norm = arma::norm(v);

  if (v_norm > zero_tol) omega = 2*std::atan2(v_norm,u)*v/v_norm;
  else omega = arma::vec().zeros(3);

  return omega;
}

arma::vec logDot_to_rotVel(const arma::vec &logQ_dot, const arma::vec &quat)
{   
    arma::vec rot_vel;

    double w = quat[0];
    if ((1 - std::fabs(w)) <= 1e-8) rot_vel = logQ_dot;
    else
    {
        arma::vec v = quat.subvec(1,3);
        double norm_v = arma::norm(v);
        arma::vec k = v / norm_v;  // axis of rotation
        double s_th = norm_v;  // sin(theta/2)
        double c_th = w;  // cos(theta/2)
        double th = std::atan2(s_th, c_th);  // theta/2
        arma::vec Pk_qdot = k * arma::dot(k, logQ_dot);  // projection of logQ_dot on k
        rot_vel = Pk_qdot + (logQ_dot - Pk_qdot) * s_th * c_th / th + (std::pow(s_th,2) / th) * arma::cross(k, logQ_dot);
    }
        
    return rot_vel;
}

arma::vec logDDot_to_rotAccel(const arma::vec &logQ_ddot, const arma::vec &rotVel, const arma::vec &quat)
{
    arma::vec rot_accel;

    double w = quat[0];
    if ((1-std::fabs(w)) <= 1e-8) rot_accel = logQ_ddot;
    else
    {
        arma::vec v = quat.subvec(1,3);
        double norm_v = arma::norm(v);
        arma::vec k = v / norm_v; // axis of rotation
        double s_th = norm_v; // sin(theta/2)
        double c_th = w;     // cos(theta/2)
        double th = std::atan2(s_th, c_th); // theta/2

        arma::vec Pk_rotVel = k*arma::dot(k, rotVel); // projection of rotVel on k
        arma::vec qdot = Pk_rotVel + (rotVel - Pk_rotVel)*th*c_th/s_th - th*arma::cross(k, rotVel);

        arma::vec qdot_bot = qdot - arma::dot(k, qdot)*k; // projection of qdot on plane normal to k
        arma::vec k_dot = 0.5*qdot_bot/th;
        double th2_dot = 0.5*arma::dot(k, qdot);

        double sc_over_th = (s_th * c_th) / th;

        arma::vec JnDot_qdot = (1 - sc_over_th)*(arma::dot(k, qdot)*k_dot + arma::dot(k_dot, qdot)*k) + 
                (std::pow(s_th,2)/th)*arma::cross(k_dot, qdot) + 
                ( (1 - 2*std::pow(s_th,2))/th - sc_over_th/th )*th2_dot*qdot_bot + 
                (2*sc_over_th - std::pow(s_th/th, 2))*th2_dot*arma::cross(k, qdot);

        arma::vec Pk_qddot = arma::dot(k, logQ_ddot)*k; // projection of qddot on k
        arma::vec Jn_qddot = Pk_qddot + (logQ_ddot - Pk_qddot)*sc_over_th + (std::pow(s_th,2)/th)*arma::cross(k, logQ_ddot);

        rot_accel = Jn_qddot + JnDot_qdot;
    }

    return rot_accel;
}


} // robo_

} // namespace as64_
