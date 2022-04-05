#include <armadillo>

namespace as64_
{
    
namespace robo_
{

arma::vec quatProd(const arma::vec &quat1, const arma::vec &quat2);

arma::vec quatInv(const arma::vec &quat);

arma::vec quatExp(const arma::vec &v_rot, double zero_tol=1e-16);

arma::vec quatLog(const arma::vec &quat, double zero_tol=1e-16);

arma::vec logDot_to_rotVel(const arma::vec &logQ_dot, const arma::vec &quat);

arma::vec logDDot_to_rotAccel(const arma::vec &logQ_ddot, const arma::vec &rotVel, const arma::vec &quat);

} // robo_

} // namespace as64_