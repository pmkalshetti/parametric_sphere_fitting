// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2010 Gael Guennebaud <g.gael@free.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


// import basic and product tests for deprectaed DynamicSparseMatrix
#define EIGEN_NO_DEPRECATED_WARNING

#include "main.h"
#include <Eigen/SparseExtra>
#include "unsupported/Eigen/src/SparseExtra/BlockDiagonalSparseQR.h"



template<typename Scalar, typename BlockSparseSolver>
void block_diagonal_sparse_qr(int nBlocks, int blockRows, int blockCols) 
{
    typedef Eigen::Matrix<Scalar, Dynamic, Dynamic> DenseMatrix;

    std::cout << "block_diagonal_sparse_qr< " << eigen_test_nice_typename<BlockSparseSolver>() << "> " << nBlocks << " blocks of size " << blockRows << "x" << blockCols << "\n";

    // Generate random block diagonal matrix
    SparseMatrix<Scalar> mat;
    mat.resize(blockRows*nBlocks, blockCols*nBlocks);
    std::vector< Eigen::Triplet<Scalar> > triplets;
    triplets.reserve(blockRows*blockCols*nBlocks);
    for(int i=0; i<nBlocks; i++) 
        for(int r=0; r<blockRows; r++) 
            for(int c=0; c<blockCols; c++) 
                triplets.push_back( Eigen::Triplet<Scalar>( i*blockRows+r, i*blockCols+c, Eigen::internal::random<Scalar>() ) );

    mat.setFromTriplets(triplets.begin(), triplets.end());

    // solve using BlockDiagonalSparseQR
    BlockDiagonalSparseQR<SparseMatrix<Scalar, ColMajor, Index>, BlockSparseSolver > solver;
    solver.setSparseBlockParams(blockRows, blockCols);
    solver.compute(mat);

    // check result
    SparseMatrix<Scalar> Q = solver.matrixQ();
    SparseMatrix<Scalar> R = solver.matrixR();

    // check A*P = Q*R
    DenseMatrix  Q_dot_R = Q.toDense()*R.toDense();
    DenseMatrix AP = mat.toDense();
    solver.colsPermutation().applyThisOnTheRight(AP);
    VERIFY_IS_APPROX( Q_dot_R, AP );

    // check Q*Qt = I
    DenseMatrix QQt = (Q*Q.transpose()).toDense();
    DenseMatrix I = Matrix<Scalar, Dynamic, Dynamic>::Identity(Q.rows(), Q.cols());
    VERIFY_IS_APPROX( QQt, I );

    // check R = upper triangular
    for(int i=0; i<R.rows(); i++)
        for(int j=0; j<R.cols() && j<i; j++)
            eigen_assert( fabs(R.coeff(i,j)) < 0.00001 );
        
}


void block_diagonal_sparse_qr_check_invalid_structure(int nBlocks, int blockRows, int blockCols) {

    // Generate random matrix with incorrect size
    SparseMatrix<double> mat;
    mat.resize(blockRows*nBlocks, blockCols*nBlocks+Eigen::internal::random<int>(1,10));

    // try to solve using BlockDiagonalSparseQR
    BlockDiagonalSparseQR<SparseMatrix<double, ColMajor, Index>, SparseQR<SparseMatrix<double>, COLAMDOrdering<int> > > solver;
    solver.setSparseBlockParams(blockRows, blockCols);
    VERIFY_RAISES_ASSERT( solver.compute(mat) );

    // again
    mat.resize(blockRows*nBlocks+Eigen::internal::random<int>(1,10), blockCols*nBlocks);
    VERIFY_RAISES_ASSERT( solver.compute(mat) );
}


void block_diagonal_sparse_qr_check_values_outside_blocks(int nBlocks, int blockRows, int blockCols) {

    // Generate full random matrix
    SparseMatrix<double> mat;
    mat.resize(blockRows*nBlocks, blockCols*nBlocks);
    std::vector< Eigen::Triplet<double> > triplets;
    for(int i=0; i<mat.rows(); i++) {
        for(int j=0; j<mat.cols(); j++) {
            triplets.push_back( Eigen::Triplet<double>( i, j, Eigen::internal::random<double>() ) );
        }
    }
    mat.setFromTriplets(triplets.begin(), triplets.end());

    // try to solve using BlockDiagonalSparseQR
    BlockDiagonalSparseQR<SparseMatrix<double, ColMajor, Index>, SparseQR<SparseMatrix<double>, COLAMDOrdering<int> > > solver;
    solver.setSparseBlockParams(blockRows, blockCols);
    VERIFY_RAISES_ASSERT( solver.compute(mat) );

}


void test_block_diagonal_sparse_qr()
{
  for(int i = 0; i < g_repeat; i++) {

    typedef double Scalar;

    // Fixed-size blocks
    CALL_SUBTEST((block_diagonal_sparse_qr<Scalar, ColPivHouseholderQR<Matrix<Scalar, 3, 2>>>(3, 3, 2)));


    // check BlockDiagonalSparseQR with sparse solver (for internal blocks)
    typedef SparseQR<SparseMatrix<Scalar>, COLAMDOrdering<int> > BlockSparseSolver;
    CALL_SUBTEST((block_diagonal_sparse_qr<Scalar, BlockSparseSolver>(2, 3, 5)));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 1,1,1 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 1,2,2 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 1,3,5 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 1,5,3 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 2,1,1 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 2,2,2 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 2,5,3 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 11,1,1 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 11,2,2 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 11,3,5 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockSparseSolver>( 11,5,3 ) ));

    // check BlockDiagonalSparseQR with dense solver (for internal blocks)
    typedef ColPivHouseholderQR<Matrix<Scalar,Dynamic,Dynamic> > BlockDenseSolver;
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 1,1,1 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 1,2,2 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 1,3,5 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 1,5,3 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 2,1,1 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 2,2,2 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 2,3,5 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 2,5,3 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 11,1,1 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 11,2,2 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 11,3,5 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr<Scalar, BlockDenseSolver>( 11,5,3 ) ));

    // check BlockDiagonalSparseQR fails with an invalid block structure
    CALL_SUBTEST(( block_diagonal_sparse_qr_check_invalid_structure( 1,1,1 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr_check_invalid_structure( 1,2,2 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr_check_invalid_structure( 2,3,5 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr_check_invalid_structure( 11,5,3 ) ));

    // check BlockDiagonalSparseQR fails with data outside of the blocks
    CALL_SUBTEST(( block_diagonal_sparse_qr_check_values_outside_blocks( 3,1,1 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr_check_values_outside_blocks( 5,2,2 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr_check_values_outside_blocks( 7,3,5 ) ));
    CALL_SUBTEST(( block_diagonal_sparse_qr_check_values_outside_blocks( 11,5,3 ) ));

  }
}


//triplets.push_back( Eigen::Triplet<Scalar>(0,blockCols*nBlocks-1,4));
