#include <iostream>
#include <cmath>
#include <armadillo>

namespace as64_
{

namespace robo_
{

struct TrajPoint
{
    TrajPoint(const arma::vec &p, const arma::vec &v, const arma::vec &a):
    pos(p), vel(v), accel(a) {}

    arma::vec pos;
    arma::vec vel;
    arma::vec accel;
};

struct QuatTrajPoint
{
    QuatTrajPoint(const arma::vec &q, const arma::vec &v_rot, const arma::vec &a_rot):
    quat(q), rot_vel(v_rot), rot_accel(a_rot) {}

    arma::vec quat;
    arma::vec rot_vel;
    arma::vec rot_accel;
};

TrajPoint get5thOrderTraj(double t, const arma::vec &p0, const arma::vec &pf, double total_time);
    
QuatTrajPoint get5thOrderQuatTraj(double t, const arma::vec &Q0, const arma::vec &Qf, double total_time);

} // namespace robo_

} // namespace as64_