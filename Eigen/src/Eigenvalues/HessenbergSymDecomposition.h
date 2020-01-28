// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_HessenbergSymDecomposition_H
#define EIGEN_HessenbergSymDecomposition_H

#include <iostream>

namespace Eigen { 

namespace internal {
  
template<typename MatrixType> struct HessenbergSymDecompositionMatrixHReturnType;
template<typename MatrixType>
struct traits<HessenbergSymDecompositionMatrixHReturnType<MatrixType> >
{
  typedef MatrixType ReturnType;
};

}

/** \eigenvalues_module \ingroup Eigenvalues_Module
  *
  *
  * \class HessenbergSymDecomposition
  *
  * \brief Reduces a square symmetric matrix to Hessenberg form by an orthogonal similarity transformation
  *
  * \tparam _MatrixType the type of the matrix of which we are computing the Hessenberg decomposition
  *
  * This class performs an Hessenberg decomposition of a matrix \f$ A \f$. In
  * the real case, the Hessenberg decomposition consists of an orthogonal
  * matrix \f$ Q \f$ and a Hessenberg matrix \f$ H \f$ such that \f$ A = Q H
  * Q^T \f$. An orthogonal matrix is a matrix whose inverse equals its
  * transpose (\f$ Q^{-1} = Q^T \f$). A Hessenberg matrix has zeros below the
  * subdiagonal, so it is almost upper triangular. The Hessenberg decomposition
  * of a complex matrix is \f$ A = Q H Q^* \f$ with \f$ Q \f$ unitary (that is,
  * \f$ Q^{-1} = Q^* \f$).
  *
  * Call the function compute() to compute the Hessenberg decomposition of a
  * given matrix. Alternatively, you can use the
  * HessenbergSymDecomposition(const MatrixType&) constructor which computes the
  * Hessenberg decomposition at construction time. Once the decomposition is
  * computed, you can use the matrixH() and matrixQ() functions to construct
  * the matrices H and Q in the decomposition.
  *
  * The documentation for matrixH() contains an example of the typical use of
  * this class.
  *
  * \sa class ComplexSchur, class Tridiagonalization, \ref QR_Module "QR Module"
  */
template<typename _MatrixType> class HessenbergSymDecomposition
{
  public:

    /** \brief Synonym for the template parameter \p _MatrixType. */
    typedef _MatrixType MatrixType;

    enum {
      Size = MatrixType::RowsAtCompileTime,
      SizeMinusOne = Size == Dynamic ? Dynamic : Size - 1,
      Options = MatrixType::Options,
      MaxSize = MatrixType::MaxRowsAtCompileTime,
      MaxSizeMinusOne = MaxSize == Dynamic ? Dynamic : MaxSize - 1
    };

    /** \brief Scalar type for matrices of type #MatrixType. */
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;

    /** \brief Type for vector of Householder coefficients.
      *
      * This is column vector with entries of type #Scalar. The length of the
      * vector is one less than the size of #MatrixType, if it is a fixed-side
      * type.
      */
    typedef Matrix<Scalar, SizeMinusOne, 1, Options & ~RowMajor, MaxSizeMinusOne, 1> CoeffVectorType;

    /** \brief Return type of matrixQ() */
    typedef HouseholderSequence<MatrixType,typename internal::remove_all<typename CoeffVectorType::ConjugateReturnType>::type> HouseholderSequenceType;
    
    typedef internal::HessenbergSymDecompositionMatrixHReturnType<MatrixType> MatrixHReturnType;

    /** \brief Default constructor; the decomposition will be computed later.
      *
      * \param [in] size  The size of the matrix whose Hessenberg decomposition will be computed.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via compute().  The \p size parameter is only
      * used as a hint. It is not an error to give a wrong \p size, but it may
      * impair performance.
      *
      * \sa compute() for an example.
      */
    HessenbergSymDecomposition(Index size = Size==Dynamic ? 2 : Size)
      : m_matrix(size,size),
        m_temp(size),
        m_isInitialized(false)
    {
      if(size>1)
        m_hCoeffs.resize(size-1);
    }

    /** \brief Constructor; computes Hessenberg decomposition of given matrix.
      *
      * \param[in]  matrix  Square matrix whose Hessenberg decomposition is to be computed.
      *
      * This constructor calls compute() to compute the Hessenberg
      * decomposition.
      *
      * \sa matrixH() for an example.
      */
    HessenbergSymDecomposition(const MatrixType& matrix)
      : m_matrix(matrix),
        m_temp(matrix.rows()),
        m_isInitialized(false)
    {
      if(matrix.rows()<2)
      {
        m_isInitialized = true;
        return;
      }
      m_hCoeffs.resize(matrix.rows()-1,1);
      _compute(m_matrix, m_hCoeffs, m_temp);
      m_isInitialized = true;
    }

    /** \brief Computes Hessenberg decomposition of given matrix.
      *
      * \param[in]  matrix  Square matrix whose Hessenberg decomposition is to be computed.
      * \returns    Reference to \c *this
      *
      * The Hessenberg decomposition is computed by bringing the columns of the
      * matrix successively in the required form using Householder reflections
      * (see, e.g., Algorithm 7.4.2 in Golub \& Van Loan, <i>%Matrix
      * Computations</i>). The cost is \f$ 10n^3/3 \f$ flops, where \f$ n \f$
      * denotes the size of the given matrix.
      *
      * This method reuses of the allocated data in the HessenbergSymDecomposition
      * object.
      *
      * Example: \include HessenbergSymDecomposition_compute.cpp
      * Output: \verbinclude HessenbergSymDecomposition_compute.out
      */
    HessenbergSymDecomposition& compute(const MatrixType& matrix)
    {
      m_matrix = matrix;
      if(matrix.rows()<2)
      {
        m_isInitialized = true;
        return *this;
      }
      m_hCoeffs.resize(matrix.rows()-1,1);
      _compute(m_matrix, m_hCoeffs, m_temp);
      m_isInitialized = true;
      return *this;
    }

    /** \brief Returns the Householder coefficients.
      *
      * \returns a const reference to the vector of Householder coefficients
      *
      * \pre Either the constructor HessenbergSymDecomposition(const MatrixType&)
      * or the member function compute(const MatrixType&) has been called
      * before to compute the Hessenberg decomposition of a matrix.
      *
      * The Householder coefficients allow the reconstruction of the matrix
      * \f$ Q \f$ in the Hessenberg decomposition from the packed data.
      *
      * \sa packedMatrix(), \ref Householder_Module "Householder module"
      */
    const CoeffVectorType& householderCoefficients() const
    {
      eigen_assert(m_isInitialized && "HessenbergSymDecomposition is not initialized.");
      return m_hCoeffs;
    }

    /** \brief Returns the internal representation of the decomposition
      *
      *	\returns a const reference to a matrix with the internal representation
      *	         of the decomposition.
      *
      * \pre Either the constructor HessenbergSymDecomposition(const MatrixType&)
      * or the member function compute(const MatrixType&) has been called
      * before to compute the Hessenberg decomposition of a matrix.
      *
      * The returned matrix contains the following information:
      *  - the upper part and lower sub-diagonal represent the Hessenberg matrix H
      *  - the rest of the lower part contains the Householder vectors that, combined with
      *    Householder coefficients returned by householderCoefficients(),
      *    allows to reconstruct the matrix Q as
      *       \f$ Q = H_{N-1} \ldots H_1 H_0 \f$.
      *    Here, the matrices \f$ H_i \f$ are the Householder transformations
      *       \f$ H_i = (I - h_i v_i v_i^T) \f$
      *    where \f$ h_i \f$ is the \f$ i \f$th Householder coefficient and
      *    \f$ v_i \f$ is the Householder vector defined by
      *       \f$ v_i = [ 0, \ldots, 0, 1, M(i+2,i), \ldots, M(N-1,i) ]^T \f$
      *    with M the matrix returned by this function.
      *
      * See LAPACK for further details on this packed storage.
      *
      * Example: \include HessenbergSymDecomposition_packedMatrix.cpp
      * Output: \verbinclude HessenbergSymDecomposition_packedMatrix.out
      *
      * \sa householderCoefficients()
      */
    const MatrixType& packedMatrix() const
    {
      eigen_assert(m_isInitialized && "HessenbergSymDecomposition is not initialized.");
      return m_matrix;
    }

    /** \brief Reconstructs the orthogonal matrix Q in the decomposition
      *
      * \returns object representing the matrix Q
      *
      * \pre Either the constructor HessenbergSymDecomposition(const MatrixType&)
      * or the member function compute(const MatrixType&) has been called
      * before to compute the Hessenberg decomposition of a matrix.
      *
      * This function returns a light-weight object of template class
      * HouseholderSequence. You can either apply it directly to a matrix or
      * you can convert it to a matrix of type #MatrixType.
      *
      * \sa matrixH() for an example, class HouseholderSequence
      */
    HouseholderSequenceType matrixQ() const
    {
      eigen_assert(m_isInitialized && "HessenbergSymDecomposition is not initialized.");
      return HouseholderSequenceType(m_matrix, m_hCoeffs.conjugate())
             .setLength(m_matrix.rows() - 1)
             .setShift(1);
    }

    /** \brief Constructs the Hessenberg matrix H in the decomposition
      *
      * \returns expression object representing the matrix H
      *
      * \pre Either the constructor HessenbergSymDecomposition(const MatrixType&)
      * or the member function compute(const MatrixType&) has been called
      * before to compute the Hessenberg decomposition of a matrix.
      *
      * The object returned by this function constructs the Hessenberg matrix H
      * when it is assigned to a matrix or otherwise evaluated. The matrix H is
      * constructed from the packed matrix as returned by packedMatrix(): The
      * upper part (including the subdiagonal) of the packed matrix contains
      * the matrix H. It may sometimes be better to directly use the packed
      * matrix instead of constructing the matrix H.
      *
      * Example: \include HessenbergSymDecomposition_matrixH.cpp
      * Output: \verbinclude HessenbergSymDecomposition_matrixH.out
      *
      * \sa matrixQ(), packedMatrix()
      */
    MatrixHReturnType matrixH() const
    {
      eigen_assert(m_isInitialized && "HessenbergSymDecomposition is not initialized.");
      return MatrixHReturnType(*this);
    }

  private:

    typedef Matrix<Scalar, 1, Size, Options | RowMajor, 1, MaxSize> VectorType;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    static void _compute(MatrixType& matA, CoeffVectorType& hCoeffs, VectorType& temp);

  protected:
    MatrixType m_matrix;
    CoeffVectorType m_hCoeffs;
    VectorType m_temp;
    bool m_isInitialized;
};

/** \internal
  * Performs a tridiagonal decomposition of \a matA in place.
  *
  * \param matA the input selfadjoint matrix
  * \param hCoeffs returned Householder coefficients
  *
  * The result is written in the lower triangular part of \a matA.
  *
  * Implemented from Golub's "%Matrix Computations", algorithm 8.3.1.
  *
  * \sa packedMatrix()
  */
template<typename MatrixType>
void HessenbergSymDecomposition<MatrixType>::_compute(MatrixType& matA, CoeffVectorType& hCoeffs, VectorType& temp)
{
  // Apply similarity transformation to remaining columns,
  // i.e., compute A = H A H'
  // option
  // 0: original
  // 1: original, but the multiplication from the right leverages symmetry
  // 2: symmetry usage by golub, symmetric upper Hessenberg
  // 2: symmetry usage by golub, symmetric upper Hessenberg, efficient computation, n^2
  // 4: use symmetry of A = A - tau * vvT*A - tau * A*vvT + tau*tau * vvT*A*vvT
  // 5: use same symmetry but computationally efficient, 2n^2
  // by implementing A = A - tau * vpT * A - tau * A * pvT + tau*tau*vTp * vvT, with p=Av
  int option = 3;
  eigen_assert(matA.rows()==matA.cols());
  Index n = matA.rows();
  temp.resize(n);
  for (Index i = 0; i<n-1; ++i)
  {
    // let's consider the vector v = i-th column starting at position i+1
    Index remainingSize = n-i-1;
    RealScalar beta;
    Scalar h;
    matA.col(i).tail(remainingSize).makeHouseholderInPlace(h, beta);
    matA.col(i).coeffRef(i+1) = beta;
    hCoeffs.coeffRef(i) = h;

    if (option == 0)
    {
        // A = H A
        matA.bottomRightCorner(remainingSize, remainingSize)
            .applyHouseholderOnTheLeft(matA.col(i).tail(remainingSize-1), h, &temp.coeffRef(0));

        // A = A H'
        matA.rightCols(remainingSize)
            .applyHouseholderOnTheRight(matA.col(i).tail(remainingSize-1).conjugate(), numext::conj(h), &temp.coeffRef(0));
    }
    if (option == 1)
    {

        // A = H A
        // A is not symmetric anymore
        matA.bottomRightCorner(remainingSize, remainingSize)
            .applyHouseholderOnTheLeft(matA.col(i).tail(remainingSize-1), h, &temp.coeffRef(0));
        // A = A H'
        // This run is symmetric; However, the full matrix is required for the next iteration
        const Eigen::VectorXd& essential = matA.col(i).tail(remainingSize-1);
        // loop in triangular shape
        double tmp = matA.row(i+1).tail(remainingSize-1) * essential;
        tmp += matA.coeffRef(i+1,i+1);
        matA.coeffRef(i+1,i+1) -= h * tmp;
        matA.row(i+1).tail(remainingSize-1) -= h * tmp * essential.tail(remainingSize-1).transpose();
        Index ctr = 0;
        for (Index j = i+2; j<n; j++) // row
        {
            tmp = matA.row(j).tail(remainingSize-1) * essential;
            tmp += matA.coeffRef(j,i+1);
            matA.row(j).tail(remainingSize-1-ctr) -= h * tmp * essential.tail(remainingSize-1-ctr).transpose();
            ctr++;
        }
        // mirror upper triangular view transposed to lower triangular view
        matA.row(i).coeffRef(i+1) = beta;
        matA.row(i).tail(remainingSize-1).setZero();
        ctr = 0;
        for (Index j = i+1; j<n; j++) // row
        {
            matA.col(j).tail(remainingSize-1-ctr) = (matA.row(j).tail(remainingSize-1-ctr)).transpose().eval();
            if (j>i+1)
                ctr++;
        }
    }
    else if (option == 2) // golub
    {
        Eigen::VectorXd essential = Eigen::VectorXd::Zero(remainingSize);
        essential[0] = 1;
        essential.tail(remainingSize-1) = matA.col(i).tail(remainingSize-1);
        Eigen::VectorXd p = h * matA.bottomRightCorner(remainingSize,remainingSize) * essential;
        double fac = (double)(p.transpose()*essential);
        Eigen::VectorXd w = p - 0.5*h*fac * essential;
        matA(i,i+1) = matA(i+1,i);
        matA.row(i).tail(remainingSize-1).setZero();
        matA.bottomRightCorner(remainingSize,remainingSize).noalias()
          -= (essential*w.transpose() + w*essential.transpose());
    }
    else if (option == 3) // golub, optimized
    {
        const Eigen::VectorXd& essential = matA.col(i).tail(remainingSize-1);
        Eigen::Block<Eigen::MatrixXd> A = matA.bottomRightCorner(remainingSize,remainingSize-1);
        Eigen::Block<Eigen::MatrixXd> Al = matA.bottomRightCorner(remainingSize-1,remainingSize-1);
        // p = A * [1 essential]
        Eigen::VectorXd tmp = Eigen::VectorXd(remainingSize); // for some reason input temp is a horizontal vector
        tmp[0] = matA.col(i+1).tail(remainingSize-1).transpose() * essential;
        tmp.tail(remainingSize-1) = Al.selfadjointView<Eigen::Lower>() * essential;
        tmp.noalias() += matA.col(i+1).tail(remainingSize);
        // vTp
        Scalar vTp = tmp.coeffRef(0) + (essential.transpose() * tmp.tail(remainingSize-1))[0];
        // w = p - 0.5*h*vTp * essential
        tmp[0] = tmp[0] - 0.5*h*vTp;
        tmp.tail(remainingSize-1) = tmp.tail(remainingSize-1) - 0.5*h*vTp * essential;
        // go lower triangular, the n^2 part
        /*/ rank 2 update, part 1 */
        matA.coeffRef(i+1,i+1) -= h * tmp.coeffRef(0); // [1\\e]wT, the 1 part
        matA.col(i+1).tail(remainingSize-1) -= h * tmp.coeffRef(0) * essential; // [1\\e]wT, the ep[0] part
        /*/ rank 2 update, part 2 */
        matA.col(i+1).tail(remainingSize) -= h * tmp; // w*[1 e] = [w w*e], the w part
        // w(2:end)*e, ewT(2:end)
        for (Index j = 1; j<remainingSize; j++)
        {
            A.row(j).head(j) -= h * tmp.coeffRef(j) * essential.head(j).transpose();
            A.row(j).head(j) -= h * essential.coeffRef(j-1) * tmp.segment(1,j).transpose();
        }
    }
    else if (option == 4)
    {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(remainingSize);
        v[0] = 1;
        v.tail(remainingSize-1) = matA.col(i).tail(remainingSize-1);
        Eigen::Block<Eigen::MatrixXd> A = matA.bottomRightCorner(remainingSize,remainingSize);
        Eigen::MatrixXd B = v * v.transpose();
        A = A - h * B * A - h * A * B + h*h * B * A * B;
        matA.row(i).coeffRef(i+1) = beta;
        matA.row(i).tail(remainingSize-1).setZero();
    }
    else if (option == 5)
    {
        const Eigen::VectorXd& essential = matA.col(i).tail(remainingSize-1);
        Eigen::Block<Eigen::MatrixXd> A = matA.bottomRightCorner(remainingSize,remainingSize-1);
        Eigen::Block<Eigen::MatrixXd> Al = matA.bottomRightCorner(remainingSize-1,remainingSize-1);
        // p
        Eigen::VectorXd tmp = Eigen::VectorXd(remainingSize); // for some reason input temp is a horizontal vector
        // A * [1 essential]
        tmp[0] = matA.col(i+1).tail(remainingSize-1).transpose() * essential;
        tmp.tail(remainingSize-1) = Al.selfadjointView<Eigen::Lower>() * essential;
        tmp.noalias() += matA.col(i+1).tail(remainingSize);
        // vTp
        Scalar vTp = tmp.coeffRef(0) + (essential.transpose() * tmp.tail(remainingSize-1))[0];
        // go lower triangular, the 2n^2 part
        /*/ rank 1 update, part 1 */
        matA.coeffRef(i+1,i+1) -= h * tmp.coeffRef(0); // [1\\e]pT, the 1 part
        matA.coeffRef(i+1,i+1) += std::pow(h,2) * vTp; // [1\\e][1 e] = [1 e\\e eeT], the 1 part
        matA.col(i+1).tail(remainingSize-1) += std::pow(h,2) * vTp * essential; // [1\\e][1 e] = [1 e\\e eeT], the e part
        /*/ rank 1 update, part 2 */
        matA.col(i+1).tail(remainingSize) -= h * tmp; // p*[1 e] = [p p*e], the p part
        matA.col(i+1).tail(remainingSize-1) -= h * tmp.coeffRef(0) * essential; // [1\\e]pT, the ep[0] part
        // eeT, p(2:end)*e, epT(2:end)
        for (Index j = 1; j<remainingSize; j++)
        {
            A.row(j).head(j) += std::pow(h,2) * vTp * essential.coeffRef(j-1) * essential.head(j).transpose();
            A.row(j).head(j) -= h * tmp.coeffRef(j) * essential.head(j).transpose();
            A.row(j).head(j) -= h * essential.coeffRef(j-1) * tmp.segment(1,j).transpose();
        }
    }
  }
  // set upper triangular of matA to zero, only lower half (diagonal and lower subdiagonal) is required due to symmetry
  // only needs to be done now since previously we only worked on the lower triangular matrix
  if (option == 3 || option == 5)
  {
    for (Index i = 0; i<n-1; i++)
    {
        matA.row(i).tail(n-i-1).setZero();
    }
  }
}

namespace internal {

/** \eigenvalues_module \ingroup Eigenvalues_Module
  *
  *
  * \brief Expression type for return value of HessenbergSymDecomposition::matrixH()
  *
  * \tparam MatrixType type of matrix in the Hessenberg decomposition
  *
  * Objects of this type represent the Hessenberg matrix in the Hessenberg
  * decomposition of some matrix. The object holds a reference to the
  * HessenbergSymDecomposition class until the it is assigned or evaluated for
  * some other reason (the reference should remain valid during the life time
  * of this object). This class is the return type of
  * HessenbergSymDecomposition::matrixH(); there is probably no other use for this
  * class.
  */
template<typename MatrixType> struct HessenbergSymDecompositionMatrixHReturnType
: public ReturnByValue<HessenbergSymDecompositionMatrixHReturnType<MatrixType> >
{
    typedef typename MatrixType::Index Index;
  public:
    /** \brief Constructor.
      *
      * \param[in] hess  Hessenberg decomposition
      */
    HessenbergSymDecompositionMatrixHReturnType(const HessenbergSymDecomposition<MatrixType>& hess) : m_hess(hess) { }

    /** \brief Hessenberg matrix in decomposition.
      *
      * \param[out] result  Hessenberg matrix in decomposition \p hess which
      *                     was passed to the constructor
      */
    template <typename ResultType>
    inline void evalTo(ResultType& result) const
    {
      result = m_hess.packedMatrix();
      Index n = result.rows();
      if (n>2)
        result.bottomLeftCorner(n-2, n-2).template triangularView<Lower>().setZero();
    }

    Index rows() const { return m_hess.packedMatrix().rows(); }
    Index cols() const { return m_hess.packedMatrix().cols(); }

  protected:
    const HessenbergSymDecomposition<MatrixType>& m_hess;
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_HessenbergSymDecomposition_H
