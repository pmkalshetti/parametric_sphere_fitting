// -*- coding: utf-8
// vim: set fileencoding=utf-8

// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_NUMERICAL_DIFF_H
#define EIGEN_NUMERICAL_DIFF_H

#include <Eigen/SparseCore>

namespace Eigen { 

enum NumericalDiffMode {
    Forward,
    Central
};


/**
  * This class allows you to add a method df() to your functor, which will 
  * use numerical differentiation to compute an approximate of the
  * derivative for the functor. Of course, if you have an analytical form
  * for the derivative, you should rather implement df() by yourself.
  *
  * More information on
  * http://en.wikipedia.org/wiki/Numerical_differentiation
  *
  * Currently only "Forward" and "Central" scheme are implemented.
  */
template<typename _Functor, NumericalDiffMode mode=Forward>
class NumericalDiff : public _Functor
{
public:
    typedef _Functor Functor;
    typedef typename Functor::Scalar Scalar;
    typedef typename Functor::InputType InputType;
    typedef typename Functor::StepType  StepType;
    typedef typename Functor::ValueType ValueType;
    typedef typename Functor::JacobianType JacobianType;

    NumericalDiff(Scalar _epsfcn=0.) : Functor(), epsfcn(_epsfcn) {}
    NumericalDiff(const Functor& f, Scalar _epsfcn=0.) : Functor(f), epsfcn(_epsfcn) {}

    // forward constructors
    template<typename T0>
        NumericalDiff(const T0& a0) : Functor(a0), epsfcn(0) {}
    template<typename T0, typename T1>
        NumericalDiff(const T0& a0, const T1& a1) : Functor(a0, a1), epsfcn(0) {}
    template<typename T0, typename T1, typename T2>
        NumericalDiff(const T0& a0, const T1& a1, const T2& a2) : Functor(a0, a1, a2), epsfcn(0) {}

    enum {
        InputsAtCompileTime = Functor::InputsAtCompileTime,
        ValuesAtCompileTime = Functor::ValuesAtCompileTime
    };

    /**
      * return the number of evaluation of functor
      */
    int df(const InputType& _x, JacobianType &jac) 
    {
      return gradient(_x, jac);
    }

    // Gradient calculation for sparse Jacobian uses Triplets    
    template <class Derived>
    int gradient(InputType const& _x, SparseMatrixBase<Derived> &jac)
    {
        using std::sqrt;
        using std::abs;
        /* Local variables */
        int nfev=0;

        const Index n = Functor::inputs();
        const Index m = Functor::values();
        const Scalar eps = sqrt(((std::max)(epsfcn,NumTraits<Scalar>::epsilon() )));
        ValueType val1, val2;
        InputType x = _x;

        StepType delta(n);
        delta.setZero();

        // TODO : we should do this only if the size is not already known
        val1.resize(m);
        val2.resize(m);
        
        // initialization
        switch(mode) {
            case Forward:
                // compute f(x)
                Functor::operator()(x, val1); nfev++;
                break;
            case Central:
                // do nothing
                break;
            default:
                eigen_assert(false);
        };

        // Resize to an estimate of the nnz per row.
        const int NUM_NONZEROS_PER_ROW = 1;
        TripletArray<Scalar> triplets(m * NUM_NONZEROS_PER_ROW);

        // Function Body
        
        for (int j = 0; j < n; ++j) {
          Scalar h = eps; // awf: it's rarely useful to scale by x[j].  Consider the common case where derivatives are computed at x = 0.  *abs(x[j]);
          if (h == 0.) {
            h = eps;
          }
          ValueType tmp;
          x = _x;
          switch (mode) {
          case Forward:
            delta[j] = h;
            Functor::increment_in_place(&x, delta);
            Functor::operator()(x, val2);
            nfev++;
            delta[j] = 0;
            break;
          case Central:
            delta[j] = h;
            Functor::increment_in_place(&x, delta);
            Functor::operator()(x, val2); nfev++;
            delta[j] = -2 * h;
            Functor::increment_in_place(&x, delta);
            Functor::operator()(x, val1); nfev++;
            h = 2 * h;
            break;
          default:
            eigen_assert(false);
          };
          for (int i = 0; i < m; ++i) {
            Scalar v = (val2[i] - val1[i]) / h;
            if (v != 0.)
              triplets.add(i, j, v);
          }
        }

        jac.derived().resize(m, n);
        jac.derived().setFromTriplets(triplets.begin(), triplets.end());
        return nfev;
    }

    // Gradient calculation for dense Jacobian uses column assignment
    template<typename Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    int gradient(InputType const& _x, Matrix<Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &jac) const
    {
      using std::sqrt;
      using std::abs;
      /* Local variables */
      int nfev = 0;
      const typename InputType::Index n = _x.size();
      const typename InputType::Index m = Functor::values();
      const Scalar eps = sqrt(((std::max)(epsfcn, NumTraits<Scalar>::epsilon())));
      ValueType val1, val2;
      InputType x = _x;

      StepType delta(n);  // added entire line by pmkalshetti
      delta.setZero();    // added entire line by pmkalshetti

      // TODO : we should do this only if the size is not already known
      val1.resize(m);
      val2.resize(m);

      // initialization
      switch (mode) {
      case Forward:
        // compute f(x)
        Functor::operator()(x, val1); nfev++;
        break;
      case Central:
        // do nothing
        break;
      default:
        eigen_assert(false);
      };

      // Function Body
      for (int j = 0; j < n; ++j) {
        Scalar h = eps; // awf: it's rarely useful to scale by x[j].  Consider the common case where derivatives are computed at x = 0.  *abs(x[j]);
        if (h == 0.) {
          h = eps;
        }
        ValueType tmp;
        x = _x;
        switch (mode) {
        case Forward:
          delta[j] = h;
          Functor::increment_in_place(&x, delta);
          Functor::operator()(x, val2);
          nfev++;
          delta[j] = 0;
          break;
        case Central:
          delta[j] = h;
          Functor::increment_in_place(&x, delta);
          Functor::operator()(x, val2); nfev++;
          delta[j] = -2 * h;
          Functor::increment_in_place(&x, delta);
          Functor::operator()(x, val1); nfev++;
          h = 2 * h;
          break;
        default:
          eigen_assert(false);
        };
        jac.col(j) = (val2 - val1) / h;
      }
      return nfev;
    }

private:
    Scalar epsfcn;

    NumericalDiff& operator=(const NumericalDiff&);
};

} // end namespace Eigen

//vim: ai ts=4 sts=4 et sw=4
#endif // EIGEN_NUMERICAL_DIFF_H

