/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_cyclops_utils_hpp
#define _aer_cyclops_utils_hpp

#include <cstdlib>

#include <mpi.h>

namespace AER {
namespace Cyclops {

namespace MPIUtil {

bool is_initialized();
int rank();
int num_processes();
void initialize();
void finalize();

inline bool is_initialized() {
  int flag;
  MPI_Initialized(&flag);
  return flag;
}

inline int rank() {
  static int rank = -1;
  initialize();
  if (rank == -1) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

inline int num_processes() {
  static int n = -1;
  initialize();
  if (n == -1) MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

inline void initialize() {
  if (!is_initialized()) {
    std::atexit(finalize);
    MPI_Init(NULL, NULL);
  }
}

inline void finalize() {
  MPI_Finalize();
}

//-------------------------------------------------------------------------
} // end namespace MPIUtil
//-------------------------------------------------------------------------
} // end namespace Cyclops
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
