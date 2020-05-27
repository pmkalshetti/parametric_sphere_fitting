
#include <stdio.h>
#include <iostream>

#ifdef NDEBUG
#define EIGEN_NO_DEBUG
#endif

#include "main.h"
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/SparseExtra>

#include <Eigen/Eigen>
#include "unsupported/Eigen/src/SparseExtra/BlockSparseQR.h"
#include "unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h"

// This disables some useless Warnings on MSVC.
// It is intended to be done for this test only.
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

const size_t NUM_SAMPLE_POINTS =
#ifdef NDEBUG
500000;
#else
50000;
#endif

template <typename _Scalar>
struct EllipseFitting : SparseFunctor<_Scalar, int>
{
    // Class data: 2xN matrix with each column a 2D point
    Matrix2Xd ellipsePoints;

    // Number of parameters in the model, to which will be added
    // one latent variable per point.
    static const int nParamsModel = 5;

    // Constructor initializes points, and tells the base class how many parameters there are in total
    EllipseFitting(const Matrix2Xd& points ):
      SparseFunctor<_Scalar, int>(nParamsModel + points.cols(), points.cols()*2),
      ellipsePoints(points) 
    {
    }

    // Functor functions
    int operator()(const InputType& uv, ValueType& fvec) const {
      // Ellipse parameters are the last 5 entries
      auto params = uv.tail(nParamsModel);
      double a = params[0];
      double b = params[1];
      double x0 = params[2];
      double y0 = params[3];
      double r = params[4];

      // Correspondences (t values) are the first N
      for (int i = 0; i < ellipsePoints.cols(); i++) {
        double t = uv(i);
        double x = a*cos(t)*cos(r) - b*sin(t)*sin(r) + x0;
        double y = a*cos(t)*sin(r) + b*sin(t)*cos(r) + y0;
        fvec(2 * i + 0) = ellipsePoints(0, i) - x;
        fvec(2 * i + 1) = ellipsePoints(1, i) - y;
      }

      return 0;
    }

    // Functor jacobian
    int df(const InputType& uv, JacobianType& fjac) {
        // X_i - (a*cos(t_i) + x0)
        // Y_i - (b*sin(t_i) + y0)
        int npoints = ellipsePoints.cols();
        auto params = uv.tail(nParamsModel);
        double a = params[0];
        double b = params[1];
        double r = params[4];

        TripletArray<JacobianType::Scalar> triplets(npoints * 2 * 5); // npoints * rows_per_point * nonzeros_per_row
        for(int i=0; i<npoints; i++) {
            double t = uv(i);
            triplets.add(2 * i, i,             +a*cos(r)*sin(t) + b*sin(r)*cos(t));
            triplets.add(2 * i, npoints + 0,   -cos(t)*cos(r));
            triplets.add(2 * i, npoints + 1,   +sin(t)*sin(r));
            triplets.add(2 * i, npoints + 2,   -1);
            triplets.add(2 * i, npoints + 4,   +a*cos(t)*sin(r) + b*sin(t)*cos(r));

            triplets.add(2 * i + 1, i,             +a*sin(r)*sin(t) - b*cos(r)*cos(t));
            triplets.add(2 * i + 1, npoints + 0,   -cos(t)*sin(r));
            triplets.add(2 * i + 1, npoints + 1,   -sin(t)*cos(r));
            triplets.add(2 * i + 1, npoints + 3,   -1);
            triplets.add(2 * i + 1, npoints + 4,   -a*cos(t)*cos(r) + b*sin(t)*sin(r));
        }

        fjac.setFromTriplets(triplets.begin(), triplets.end());
        return 0;
    }

    // For generic Jacobian, one might use this Dense QR solver.
    typedef SparseQR<JacobianType, COLAMDOrdering<int> > GeneralQRSolver;

    // But for optimal performance, declare QRSolver that understands the sparsity structure.
    // Here it's block-diagonal LHS with dense RHS
    //
    // J1 = [J11   0   0 ... 0
    //         0 J12   0 ... 0
    //                   ...
    //         0   0   0 ... J1N];
    // And 
    // J = [J1 J2];

    // QR for J1 subblocks is 2x1
    typedef ColPivHouseholderQR<Matrix<Scalar, 2, 1> > DenseQRSolver2x1;

    // QR for J1 is block diagonal
    typedef BlockDiagonalSparseQR<JacobianType, DenseQRSolver2x1> LeftSuperBlockSolver;
    
    // QR for J1'J2 is general dense (faster than general sparse by about 1.5x for n=500K)
    typedef ColPivHouseholderQR<Matrix<Scalar, Dynamic, Dynamic> > RightSuperBlockSolver;

    // QR for J is concatenation of the above.
    typedef BlockSparseQR<JacobianType, LeftSuperBlockSolver, RightSuperBlockSolver> SchurlikeQRSolver;  
    
    typedef SchurlikeQRSolver QRSolver;

    // And tell the algorithm how to set the QR parameters.
    void initQRSolver(GeneralQRSolver &qr) {}

    void initQRSolver(SchurlikeQRSolver &qr) {
        // set block size
        qr.getLeftSolver().setSparseBlockParams(2, 1);
        qr.setBlockParams(ellipsePoints.cols());
    }
};


void ellipseFitting()
{

  //eigen_assert(false);

  // _CrtSetDbgFlag(_CRTDBG_CHECK_ALWAYS_DF);

  if (1) {
    // Check fast QR
    Matrix<double, 5, 2> A;
    A << 12, 3, -5, 17, -7, 132, 1.0, 1.1, -3.1, 4.7;
    ColPivHouseholderQR<Matrix<double, 5, 2> > qr(A);

    MatrixXd R = qr.matrixR().template triangularView<Upper>();

    Matrix<double, 5, 5> Q = qr.matrixQ();
    std::cout << "A=\n" << A << std::endl;
    std::cout << "AP=\n" << A * qr.colsPermutation() << std::endl;
    //std::cout << "QR=\n" << qr.matrixQ() * qr.matrixR().template triangularView<Upper>() << std::endl;
    std::cout << "Q=\n" << Q << std::endl;
    std::cout << "R=\n" << R << std::endl;
    std::cout << "QR=\n" << Q * R << std::endl;
    VERIFY_IS_APPROX(Q * R, A * qr.colsPermutation());
  }


    // ELLIPSE PARAMETERS
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
    std::cout << "r=" << r*180./EIGEN_PI << "\t";
    std::cout << std::endl;

    // CREATE DATA SAMPLES
    
    int nDataPoints = NUM_SAMPLE_POINTS;
    Matrix2Xd ellipsePoints;
    ellipsePoints.resize(2, nDataPoints);
    double incr = 1.3*EIGEN_PI / double(nDataPoints);
    for(int i=0; i<nDataPoints; i++) {
        double t = double(i)*incr;
        ellipsePoints(0, i) = x0 + a*cos(t)*cos(r) - b*sin(t)*sin(r);
        ellipsePoints(1, i) = y0 + a*cos(t)*sin(r) + b*sin(t)*cos(r);
    }

    // INITIAL PARAMS
    EllipseFitting<double>::InputType params;
    params.resize(EllipseFitting<double>::nParamsModel+nDataPoints);
    double minX, minY, maxX, maxY;
    minX = maxX = ellipsePoints(0,0);
    minY = maxY = ellipsePoints(1,0);
    for(int i=0; i<ellipsePoints.cols(); i++) {
        minX = (std::min)(minX, ellipsePoints(0,i));
        maxX = (std::max)(maxX, ellipsePoints(0,i));
        minY = (std::min)(minY, ellipsePoints(1,i));
        maxY = (std::max)(maxY, ellipsePoints(1,i));
    }
    params(ellipsePoints.cols()) = 0.5*(maxX - minX);
    params(ellipsePoints.cols()+1) = 0.5*(maxY - minY);
    params(ellipsePoints.cols()+2) = 0.5*(maxX + minX);
    params(ellipsePoints.cols()+3) = 0.5*(maxY + minY);
    params(ellipsePoints.cols()+4) = 0;
    for(int i=0; i<ellipsePoints.cols(); i++) {
        params(i) = double(i)*incr;
    }

    std::cout << "INITIALIZATION" << " ";
    std::cout << "a=" << params(ellipsePoints.cols()) << "\t";
    std::cout << "b=" << params(ellipsePoints.cols() + 1) << "\t";
    std::cout << "x0=" << params(ellipsePoints.cols() + 2) << "\t";
    std::cout << "y0=" << params(ellipsePoints.cols() + 3) << "\t";
    std::cout << "r=" << params(ellipsePoints.cols() + 4)*180. / EIGEN_PI << "\t";
    std::cout << std::endl << std::endl;

    typedef EllipseFitting<double> Functor;
    Functor functor(ellipsePoints);
    Eigen::LevenbergMarquardt< Functor > lm(functor);
    lm.setVerbose(true);

    Eigen::LevenbergMarquardtSpace::Status info = lm.minimize(params);

    std::cout << "END[" << info << "]";
    std::cout << "a=" << params(ellipsePoints.cols()) << "\t";
    std::cout << "b=" << params(ellipsePoints.cols()+1) << "\t";
    std::cout << "x0=" << params(ellipsePoints.cols()+2) << "\t";
    std::cout << "y0=" << params(ellipsePoints.cols()+3) << "\t";
    std::cout << "r=" << params(ellipsePoints.cols()+4)*180./EIGEN_PI << "\t";
    std::cout << std::endl << std::endl;

    // check parameters ambiguity before test result
    // a should be bigger than b
    if( fabs(params(ellipsePoints.cols()+1)) > fabs(params(ellipsePoints.cols())) ) {
       std::swap( params(ellipsePoints.cols()), params(ellipsePoints.cols()+1) );
       params(ellipsePoints.cols()+4) -= 0.5*EIGEN_PI;
    }
    // a and b should be positive
    if(params(ellipsePoints.cols())<0 ) {
       params(ellipsePoints.cols()) *= -1.;
       params(ellipsePoints.cols()+1) *= -1.;
       params(ellipsePoints.cols()+4) += EIGEN_PI;
    }
    // fix rotation angle range
    while( params(ellipsePoints.cols()+4) < 0 ) params(ellipsePoints.cols()+4) += 2.*EIGEN_PI;
    while( params(ellipsePoints.cols()+4) > EIGEN_PI ) params(ellipsePoints.cols()+4) -= EIGEN_PI;


    eigen_assert( fabs(a - params(ellipsePoints.cols())) < 0.00001 );
    eigen_assert( fabs(b - params(ellipsePoints.cols()+1)) < 0.00001 );
    eigen_assert( fabs(x0 - params(ellipsePoints.cols()+2)) < 0.00001 );
    eigen_assert( fabs(y0 - params(ellipsePoints.cols()+3)) < 0.00001 );
    eigen_assert( fabs(r - params(ellipsePoints.cols()+4)) < 0.00001 );



}


void test_ellipse_fitting()
{
    CALL_SUBTEST(ellipseFitting());
}
