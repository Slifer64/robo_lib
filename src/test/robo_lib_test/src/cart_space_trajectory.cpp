#include <iostream>
#include <cmath>
#include <armadillo>

#include <robo_lib/trajectory.h>
#include <plot_lib/qt_plot.h>

using namespace as64_;

int main(int argc, char **argv)
{
    arma::vec p0 = {0, 0, 0};
    arma::vec Q0 = {0.5, 0.6, -0.7, 0.6};
    Q0 = Q0 / arma::norm(Q0);

    arma::vec pf = {1.2, 3.4, 2.7};
    arma::vec Qf = {0.1, -0.4, 0.3, 0.9};
    Qf = Qf / arma::norm(Qf);
    Qf = -Qf;

    double Tf = 5.2; // total duration

    double dt = 0.002;

    double t = 0;

    arma::rowvec Time;
    arma::mat pos_data;
    arma::mat vel_data;

    arma::mat Quat_data;
    arma::mat rotVel_data;

    while (t < Tf)
    {
        // std::cerr << "t = " << t << "\n";

        Time = arma::join_horiz(Time, arma::vec({t}));

        robo_::TrajPoint p = robo_::get5thOrderTraj(t, p0, pf, Tf);
        pos_data = arma::join_horiz(pos_data, p.pos);
        vel_data = arma::join_horiz(vel_data, p.vel);

        robo_::QuatTrajPoint q = robo_::get5thOrderQuatTraj(t, Q0, Qf, Tf);
        Quat_data = arma::join_horiz(Quat_data, q.quat);
        rotVel_data = arma::join_horiz(rotVel_data, q.rot_vel);

        arma::vec twist = arma::join_vert(p.vel, q.rot_vel);
        // robot->setTaskVelocity(twist);

        t += dt;
    }

    pl_::QtPlot::init();

    // ============ Position ================

    pl_::Figure *fig_ = pl_::figure("Position", {500, 600});
    int n = pos_data.n_rows;
    fig_->setAxes(n,1);
    for (int i=0; i<n; i++)
    {
        pl_::Axes *ax = fig_->getAxes(i);
        ax->plot(Time, pos_data.row(i), pl_::LineWidth_,2.0, pl_::Color_,pl_::BLUE);
        ax->plot({Tf}, arma::rowvec({pf(i)}), pl_::LineWidth_,3.0, pl_::LineStyle_,pl_::NoLine, pl_::Color_, pl_::RED, pl_::MarkerStyle_, pl_::ssCross, pl_::MarkerSize_, 12);
        ax->plot({0}, arma::rowvec({p0(i)}), pl_::LineWidth_,3.0, pl_::LineStyle_,pl_::NoLine, pl_::Color_, pl_::GREEN, pl_::MarkerStyle_, pl_::ssCircle, pl_::MarkerSize_, 12);
        if (i == n-1) ax->xlabel("Time [s]", pl_::FontSize_,14);
        ax->drawnow();
    }


    fig_ = pl_::figure("Velocity", {500, 600});
    n = vel_data.n_rows;
    fig_->setAxes(n,1);
    for (int i=0; i<n; i++)
    {
        pl_::Axes *ax = fig_->getAxes(i);
        ax->plot(Time, vel_data.row(i), pl_::LineWidth_,2.0, pl_::Color_,pl_::RED);
        if (i == n-1) ax->xlabel("Time [s]", pl_::FontSize_,14);
        ax->drawnow();
    }


    // ============ Quaternion ================

    fig_ = pl_::figure("Quaternion", {500, 600});
    n = Quat_data.n_rows;
    fig_->setAxes(n,1);
    for (int i=0; i<n; i++)
    {
        pl_::Axes *ax = fig_->getAxes(i);
        ax->plot(Time, Quat_data.row(i), pl_::LineWidth_,2.0, pl_::Color_,pl_::BLUE);
        ax->plot({Tf}, arma::rowvec({Qf(i)}), pl_::LineWidth_,3.0, pl_::LineStyle_,pl_::NoLine, pl_::Color_, pl_::RED, pl_::MarkerStyle_, pl_::ssCross, pl_::MarkerSize_, 12);
        ax->plot({Tf}, arma::rowvec({-Qf(i)}), pl_::LineWidth_,3.0, pl_::LineStyle_,pl_::NoLine, pl_::Color_, pl_::RED, pl_::MarkerStyle_, pl_::ssStar, pl_::MarkerSize_, 12);
        ax->plot({0}, arma::rowvec({Q0(i)}), pl_::LineWidth_,3.0, pl_::LineStyle_,pl_::NoLine, pl_::Color_, pl_::GREEN, pl_::MarkerStyle_, pl_::ssCircle, pl_::MarkerSize_, 12);
        if (i == n-1) ax->xlabel("Time [s]", pl_::FontSize_,14);
        ax->drawnow();
    }


    fig_ = pl_::figure("Rotational Velocity", {500, 600});
    n = rotVel_data.n_rows;
    fig_->setAxes(n,1);
    for (int i=0; i<n; i++)
    {
        pl_::Axes *ax = fig_->getAxes(i);
        ax->plot(Time, rotVel_data.row(i), pl_::LineWidth_,2.0, pl_::Color_,pl_::RED);
        if (i == n-1) ax->xlabel("Time [s]", pl_::FontSize_,14);
        ax->drawnow();
    }

    std::cout << "Press [enter] to continue...\n";
    std::string dummy;
    std::getline(std::cin, dummy, '\n');

    return 0;

}