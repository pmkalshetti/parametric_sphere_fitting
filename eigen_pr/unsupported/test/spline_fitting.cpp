

// Spline fit with unknown parameter positions.


#include <iostream>

#include "main.h"
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>

#include <Eigen/Eigen>
#include "unsupported/Eigen/src/SparseExtra/BlockSparseQR.h"
#include "unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h"

// This disables some useless Warnings on MSVC.
// It is intended to be done for this test only.
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

#define NUM_SAMPLE_POINTS 100

struct SplineParameter {
  int segment;
  double t;
};

template <typename _Scalar>
struct SplineFitting : LevenbergMarquardtFunctor<_Scalar>
{
  typedef LevenbergMarquardtFunctor<Scalar> Base;

  typedef int Index;
  typedef Matrix<Scalar, Dynamic, 1> InputType;
  typedef Matrix<Scalar, Dynamic, 1> ValueType;
  typedef Matrix<Scalar, Dynamic, 1> StepType;

  typedef SparseMatrix<Scalar, ColMajor, Index> JacobianType;

  typedef SparseQR<SparseMatrix<Scalar>, COLAMDOrdering<int> > BlockSolver;
  typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > DenseSolver;
  typedef BlockDiagonalSparseQR<JacobianType, DenseSolver> LeftSuperBlockSolver;
  typedef BlockSparseQR<JacobianType, LeftSuperBlockSolver, DenseSolver> QRSolver;

  const Eigen::Matrix<double, 3, Eigen::Dynamic> SplinePoints;

  static const int nParamsModel = 5;

  SplineFitting(Eigen::Matrix<double, 3, Eigen::Dynamic>& points) :
    Base(nParamsModel + points.cols(), points.cols() * 2),
    SplinePoints(points)
  {
  }

  // Functor functions
  int operator()(const InputType& uv, ValueType& fvec) {
    int npoints = SplinePoints.cols();
    auto params = uv.tail(nParamsModel);
    double a = params[0];
    double b = params[1];
    double x0 = params[2];
    double y0 = params[3];
    double r = params[4];
    for (int i = 0; i < npoints; i++) {
      double t = uv[i];
      double x = a*cos(t)*cos(r) - b*sin(t)*sin(r) + x0;
      double y = a*cos(t)*sin(r) + b*sin(t)*cos(r) + y0;
      fvec(2 * i + 0) = SplinePoints(0, i) - x;
      fvec(2 * i + 1) = SplinePoints(1, i) - y;
    }

    return 0;
  }

  int df(const InputType& uv, JacobianType& fjac) {
    // X_i - (a*cos(t_i) + x0)
    // Y_i - (b*sin(t_i) + y0)
    int npoints = SplinePoints.cols();
    auto params = uv.tail(nParamsModel);
    double a = params[0];
    double b = params[1];
    double r = params[4];
    for (int i = 0; i<npoints; i++) {
      double t = uv(i);
      fjac.coeffRef(2 * i, npoints + 0) = -cos(t)*cos(r);
      fjac.coeffRef(2 * i, npoints + 1) = +sin(t)*sin(r);
      fjac.coeffRef(2 * i, npoints + 2) = -1;
      fjac.coeffRef(2 * i, npoints + 4) = +a*cos(t)*sin(r) + b*sin(t)*cos(r);
      fjac.coeffRef(2 * i, i) = +a*cos(r)*sin(t) + b*sin(r)*cos(t);

      fjac.coeffRef(2 * i + 1, npoints + 0) = -cos(t)*sin(r);
      fjac.coeffRef(2 * i + 1, npoints + 1) = -sin(t)*cos(r);
      fjac.coeffRef(2 * i + 1, npoints + 3) = -1;
      fjac.coeffRef(2 * i + 1, npoints + 4) = -a*cos(t)*cos(r) + b*sin(t)*sin(r);
      fjac.coeffRef(2 * i + 1, i) = +a*sin(r)*sin(t) - b*cos(r)*cos(t);
    }

    fjac.makeCompressed();
    return 0;
  }


  void initQRSolver(QRSolver &qr) {
    // set block size
    qr.getLeftSolver().setSparseBlockParams(2, 1);
    qr.setBlockParams(SplinePoints.cols());
  }

  void increment_in_place(InputType* x, StepType const& p)
  {
    *x += p;
  }

  double estimateNorm(InputType const& x, StepType const& diag)
  {
    return x.cwiseProduct(diag).stableNorm();
  }
};


void testSplineFitting() {

  // Spline PARAMETERS
  double a, b, x0, y0, r;
  a = 7.5;
  b = 2;
  x0 = 17.;
  y0 = 23.;
  r = 0.23;

  std::cout << "GROUND TRUTH   " << " ";
  std::cout << "a=" << a << "\t";
  std::cout << "b=" << b << "\t";
  std::cout << "x0=" << x0 << "\t";
  std::cout << "y0=" << y0 << "\t";
  std::cout << "r=" << r*180. / EIGEN_PI << "\t";
  std::cout << std::endl;

  // CREATE DATA SAMPLES

  int nDataPoints = NUM_SAMPLE_POINTS;
  Eigen::Matrix<double, 3, Eigen::Dynamic> SplinePoints;
  SplinePoints.resize(3, nDataPoints);
  double incr = 1.3*EIGEN_PI / double(nDataPoints);
  for (int i = 0; i<nDataPoints; i++) {
    double t = double(i)*incr;
    SplinePoints(0, i) = x0 + a*cos(t)*cos(r) - b*sin(t)*sin(r);
    SplinePoints(1, i) = y0 + a*cos(t)*sin(r) + b*sin(t)*cos(r);
    SplinePoints(2, i) = 1;
  }

  // INITIAL PARAMS
  SplineFitting<double>::InputType lm_params;
  auto& params = lm_params;
  params.resize(SplineFitting<double>::nParamsModel + nDataPoints);
  double minX, minY, maxX, maxY;
  minX = maxX = SplinePoints(0, 0);
  minY = maxY = SplinePoints(1, 0);
  for (int i = 0; i<SplinePoints.cols(); i++) {
    minX = (std::min)(minX, SplinePoints(0, i));
    maxX = (std::max)(maxX, SplinePoints(0, i));
    minY = (std::min)(minY, SplinePoints(1, i));
    maxY = (std::max)(maxY, SplinePoints(1, i));
  }
  params(SplinePoints.cols()) = 0.5*(maxX - minX);
  params(SplinePoints.cols() + 1) = 0.5*(maxY - minY);
  params(SplinePoints.cols() + 2) = 0.5*(maxX + minX);
  params(SplinePoints.cols() + 3) = 0.5*(maxY + minY);
  params(SplinePoints.cols() + 4) = 0;
  for (int i = 0; i<SplinePoints.cols(); i++) {
    params(i) = double(i)*incr;
  }

  std::cout << "INITIALIZATION" << " ";
  std::cout << "a=" << params(SplinePoints.cols()) << "\t";
  std::cout << "b=" << params(SplinePoints.cols() + 1) << "\t";
  std::cout << "x0=" << params(SplinePoints.cols() + 2) << "\t";
  std::cout << "y0=" << params(SplinePoints.cols() + 3) << "\t";
  std::cout << "r=" << params(SplinePoints.cols() + 4)*180. / EIGEN_PI << "\t";
  std::cout << std::endl << std::endl;

  SplineFitting<double> functor(SplinePoints);
  Eigen::LevenbergMarquardt< SplineFitting<double> > lm(functor);
  Eigen::LevenbergMarquardtSpace::Status info;

  info = lm.minimizeInit(lm_params);
  if (info == Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
    std::cerr << "Improper Input Parameters" << std::endl;
    return;
  }

  do {
    info = lm.minimizeOneStep(lm_params);
  } while (info == Eigen::LevenbergMarquardtSpace::Running);

  std::cout << "END" << " ";
  std::cout << "a=" << params(SplinePoints.cols()) << "\t";
  std::cout << "b=" << params(SplinePoints.cols() + 1) << "\t";
  std::cout << "x0=" << params(SplinePoints.cols() + 2) << "\t";
  std::cout << "y0=" << params(SplinePoints.cols() + 3) << "\t";
  std::cout << "r=" << params(SplinePoints.cols() + 4)*180. / EIGEN_PI << "\t";
  std::cout << std::endl << std::endl;

  // check parameters ambiguity before test result
  // a should be bigger than b
  if (fabs(params(SplinePoints.cols() + 1)) > fabs(params(SplinePoints.cols()))) {
    std::swap(params(SplinePoints.cols()), params(SplinePoints.cols() + 1));
    params(SplinePoints.cols() + 4) -= 0.5*EIGEN_PI;
  }
  // a and b should be positive
  if (params(SplinePoints.cols())<0) {
    params(SplinePoints.cols()) *= -1.;
    params(SplinePoints.cols() + 1) *= -1.;
    params(SplinePoints.cols() + 4) += EIGEN_PI;
  }
  // fix rotation angle range
  while (params(SplinePoints.cols() + 4) < 0) params(SplinePoints.cols() + 4) += 2.*EIGEN_PI;
  while (params(SplinePoints.cols() + 4) > EIGEN_PI) params(SplinePoints.cols() + 4) -= EIGEN_PI;


  eigen_assert(fabs(a - params(SplinePoints.cols())) < 0.00001);
  eigen_assert(fabs(b - params(SplinePoints.cols() + 1)) < 0.00001);
  eigen_assert(fabs(x0 - params(SplinePoints.cols() + 2)) < 0.00001);
  eigen_assert(fabs(y0 - params(SplinePoints.cols() + 3)) < 0.00001);
  eigen_assert(fabs(r - params(SplinePoints.cols() + 4)) < 0.00001);



}


void test_spline_fitting()
{
  CALL_SUBTEST(testSplineFitting());
}
