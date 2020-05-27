// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <Eigen/SparseExtra>
#include "unsupported/Eigen/src/SparseExtra/BlockSparseQR.h"

template<typename Scalar, typename BlockSolverLeft, typename BlockSolverRight>
void block_sparse_qr(int nRows, int nCols, int blockCols) 
{
    std::cout << "block_sparse_qr< " << eigen_test_nice_typename<BlockSolverLeft>() << ", " << eigen_test_nice_typename<BlockSolverRight>() << ">[" << nRows << "x" << blockCols << " | " << nRows << "x" << (nCols - blockCols) << "]\n";
    typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;

    // Generate random sparse matrix
    SparseMatrix<Scalar> mat;
    mat.resize(nRows, nCols);
    std::vector< Eigen::Triplet<Scalar> > triplets;
    const double occupancy = 0.5;
    for(int i=0; i<nRows; i++) {
        for(int j=0; j<nCols; j++) {
            if( Eigen::internal::random<double>(0., 1.) > occupancy )
                triplets.push_back( Eigen::Triplet<Scalar>(i, j, Eigen::internal::random<Scalar>() ) );
        }
    }
    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat.makeCompressed();

    // solve using BlockSparseQR, using provided Left and Right solvers
    BlockSparseQR<SparseMatrix<Scalar>, BlockSolverLeft, BlockSolverRight> solver;
    solver.setBlockParams(blockCols);
    solver.compute(mat);

    // check result
    auto Q = solver.matrixQ();
    auto R = solver.matrixR();

    DenseMatrix I = DenseMatrix::Identity(nRows, nRows);

    DenseMatrix sQ = Q * I;

    // check A*P = Q*R
    DenseMatrix  Q_dot_R = Q*R;
    DenseMatrix AP = mat.toDense();
    solver.colsPermutation().applyThisOnTheRight(AP);
    VERIFY_IS_APPROX( Q_dot_R, AP );

    // check Qt*Q = I
    DenseMatrix  QtQ = Q.transpose()*sQ;
    VERIFY_IS_APPROX( QtQ, I );

    // check R = upper triangular
    for(int i=0; i<R.rows(); i++) 
        for(int j=0; j<R.cols() && j<i; j++) 
            eigen_assert( fabs(R.coeff(i,j)) < 0.00001  );

}



void test_block_sparse_qr()
{
  for(int i = 0; i < g_repeat; i++) {

    typedef double Scalar;
    typedef SparseQR<SparseMatrix<Scalar>, COLAMDOrdering<int> > BlockSparseSolver;
    typedef ColPivHouseholderQR<Matrix<Scalar,Dynamic,Dynamic> > BlockDenseSolver;

    CALL_SUBTEST((block_sparse_qr<Scalar, BlockSparseSolver, BlockDenseSolver>(20, 31, 13)));
    CALL_SUBTEST((block_sparse_qr<Scalar, BlockSparseSolver, BlockSparseSolver>(20, 31, 13)));
    CALL_SUBTEST((block_sparse_qr<Scalar, BlockSparseSolver, BlockDenseSolver>(10, 6, 2)));
    CALL_SUBTEST((block_sparse_qr<Scalar, BlockSparseSolver, BlockDenseSolver>(6, 10, 2)));
    CALL_SUBTEST(( block_sparse_qr<Scalar, BlockSparseSolver, BlockDenseSolver>( 100, 50, 20 ) ));
    CALL_SUBTEST(( block_sparse_qr<Scalar, BlockSparseSolver, BlockDenseSolver>( 9, 8, 2 ) ));

  }
}
