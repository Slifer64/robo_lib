#include <robo_lib/trajectory.h>
#include <robo_lib/_utils/math.h>

namespace as64_
{

namespace robo_
{

TrajPoint get5thOrderTraj(double t, const arma::vec &p0, const arma::vec &pf, double total_time)
{
    int n_dof = p0.size();

    arma::vec pos = arma::vec().zeros(n_dof);
    arma::vec vel = arma::vec().zeros(n_dof);
    arma::vec accel = arma::vec().zeros(n_dof);

    if (t < 0) pos = p0;
    else if (t > total_time) pos = pf;
    else
    {
        pos = p0 + (pf - p0) * (10 *  std::pow(t / total_time, 3) -
                                15 *  std::pow(t / total_time, 4) + 6 *  std::pow(t / total_time, 5));
        vel = (pf - p0) * (30 *  std::pow(t, 2) /  std::pow(total_time, 3) -
                           60 *  std::pow(t, 3) /  std::pow(total_time, 4) + 30 *  std::pow(t, 4) /  std::pow(total_time, 5));
        accel = (pf - p0) * (60 * t /  std::pow(total_time, 3) -
                             180 *  std::pow(t, 2) /  std::pow(total_time, 4) + 120 *  std::pow(t, 3) /  std::pow(total_time, 5));
    }
        
    return TrajPoint(pos, vel, accel);
}
    
QuatTrajPoint get5thOrderQuatTraj(double t, const arma::vec &Q0, const arma::vec &Qf, double total_time)
{
    arma::vec Qf_ = Qf;
    if (arma::dot(Q0, Qf_) < 0) Qf_ = -Qf_;

    arma::vec qLog_0 = {0, 0, 0};
    arma::vec qLog_f = quatLog(quatProd(Qf_, quatInv(Q0)));

    TrajPoint temp = get5thOrderTraj(t, qLog_0, qLog_f, total_time);
    arma::vec logQ1 = temp.pos;
    arma::vec logQ1_dot = temp.vel;
    arma::vec logQ1_ddot = temp.accel;

    arma::vec Q1 = quatExp(logQ1);
    arma::vec Q = quatProd(Q1, Q0);
    arma::vec rot_vel = logDot_to_rotVel(logQ1_dot, Q1);
    arma::vec rot_accel = logDDot_to_rotAccel(logQ1_ddot, rot_vel, Q1);

    return QuatTrajPoint(Q, rot_vel, rot_accel);
}

} // namespace robo_

} // namespace as64_