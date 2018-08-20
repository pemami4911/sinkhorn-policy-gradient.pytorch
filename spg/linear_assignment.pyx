# distutils: language=c++
# distutils: sources = spg/hungarian/hungarian.cpp

import numpy as np
cimport numpy as np

# Declarations
cdef extern from "hungarian/hungarian.hpp":
    cdef cppclass hungarian_problem_t:
        int num_rows
        int num_cols
        double** cost

    cdef double hungarian_init (hungarian_problem_t* p, 
            double** cost_matrix,
            int rows,
            int cols,
            int mode)

    cdef void hungarian_free (hungarian_problem_t* p)

    cdef void hungarian_solve (hungarian_problem_t* p, double** assignment)
    
    cdef void hungarian_print_costmatrix (hungarian_problem_t* p)

    cdef void hungarian_print_status (hungarian_problem_t* p)

    cdef double** array_to_matrix (double* m, int rows, int cols)


def linear_assignment(np.ndarray[double, ndim=2] cost_matrix not None):
    """
    Instantiates a hungarian problem instance, solves it, and returns the 
    assignment matrix. Finds the minimimum cost assignment.
    
    Any padding (adding rows/columns of zeros, etc) should be performed 
    a priori so that cost_matrix.dim[0] == cost_matrix.dim[1]

    Args:
        cost_matrix is a numpy ndarray of size [r,c] of np.float64
    """
    cdef int r, c
    r = cost_matrix.shape[0]
    c = cost_matrix.shape[1]
    assert r == c

    # http://cython-users.narkive.com/y5HhpJZm/pointer-to-pointer-double-pointer-issue
    cdef np.intp_t[:] tmp = np.zeros(r, dtype=np.intp)
    cdef double** cpp_cost_matrix = <double**> (<void*> &tmp[0])
    cdef double[:,::1] assignment = np.zeros((r,c), dtype=np.float64)
    cdef np.intp_t[:] tmp2 = np.zeros(r, dtype=np.intp)
    cdef double** cpp_assignment = <double**> (<void*> &tmp2[0])
    cdef int i
    for i in range(r):
        cpp_cost_matrix[i] = &cost_matrix[i,0]
        cpp_assignment[i] = &assignment[i,0]

    cdef hungarian_problem_t p
    # HUNGARIAN_MODE_MINIMIZE_COST 0
    # HUNGARIAN_MODE_MAXIMIZE_UTIL 1
    hungarian_init(&p, cpp_cost_matrix, r, c, 0)
    hungarian_solve(&p, cpp_assignment)
    # Free memory allocated to problem
    hungarian_free(&p)

    return assignment
