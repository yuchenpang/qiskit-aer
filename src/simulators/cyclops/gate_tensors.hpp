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

#ifndef _aer_cyclops_gate_tensors_hpp
#define _aer_cyclops_gate_tensors_hpp

#include <vector>
#define _USE_MATH_DEFINE
#include <cmath>

#include "types.hpp"
#include "mpi_util.hpp"

namespace AER {
namespace Cyclops {
namespace GateTensors {

// ----------------------------------------------------------------------------
// Constant gates
// ----------------------------------------------------------------------------
TensorPtr x();
TensorPtr y();
TensorPtr z();
TensorPtr h();
TensorPtr s();
TensorPtr sdg();
TensorPtr t();
TensorPtr tdg();
TensorPtr cx();
TensorPtr ccx();
TensorPtr swap();

// ----------------------------------------------------------------------------
// Nonconstant gates
// ----------------------------------------------------------------------------
TensorPtr u1(real_t lambda);


//=============================================================================
// Implementation: Utilities
//=============================================================================
inline TensorPtr make_gate_tensor(uint_t num_qubits) {
  return make_tensor(num_qubits*2, std::vector<int>(num_qubits*2, 2).data());
}

template <int npairs>
inline void write_tensor(TensorPtr tensor, Pair const (&pairs)[npairs]) {
  // TODO let each process write to its local data
  if (MPIUtil::rank() == 0) {
    tensor->write(npairs, pairs);
  } else {
    tensor->write(0, nullptr);
  }
}

template <std::size_t n_in, std::size_t n_out>
inline Pair entry(int const (&in_idx)[n_in], int const (&out_idx)[n_out], complex_t value) {
   int idx = 0;
   int base = 1;
   for (int n : in_idx) {
     idx += n * base;
     base *= 2;
   }
  for (int n : out_idx) {
    idx += n * base;
    base *= 2;
  }
  return Pair(idx, value);
}

//=============================================================================
// Implementation: Gate definitions
//=============================================================================

// ----------------------------------------------------------------------------
// Constant gates
// ----------------------------------------------------------------------------
inline TensorPtr x() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(1);
      write_tensor(tensor, {
        entry({0}, {1}, 1),
        entry({1}, {0}, 1),
    });
  }
  return tensor;
}

inline TensorPtr y() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(1);
    write_tensor(tensor, {
      entry({0}, {1}, complex_t(0, 1)),
      entry({1}, {0}, complex_t(0, -1)),
    });
  }
  return tensor;
}

inline TensorPtr z() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(1);
    write_tensor(tensor, {
      entry({0}, {0}, 1),
      entry({1}, {1}, -1),
    });
  }
  return tensor;
}

inline TensorPtr h() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(1);
    complex_t a = 1/std::sqrt(2);
    write_tensor(tensor, {
      entry({0}, {0}, a),
      entry({0}, {1}, a),
      entry({1}, {0}, a),
      entry({1}, {1}, -a),
    });
  }
  return tensor;
}

inline TensorPtr s() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(1);
    write_tensor(tensor, {
      entry({0}, {0}, 1),
      entry({1}, {1}, complex_t(0, 1)),
    });
  }
  return tensor;
}

inline TensorPtr sdg() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(1);
    write_tensor(tensor, {
      entry({0}, {0}, 1),
      entry({1}, {1}, complex_t(0, -1)),
    });
  }
  return tensor;
}

inline TensorPtr t() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(1);
    write_tensor(tensor, {
      entry({0}, {0}, 1),
      entry({1}, {1}, std::exp(complex_t(0, M_PI/4))),
    });
  }
  return tensor;
}

inline TensorPtr tdg() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(1);
    write_tensor(tensor, {
      entry({0}, {0}, 1),
      entry({1}, {1}, std::exp(complex_t(0, -M_PI/4))),
    });
  }
  return tensor;
}

inline TensorPtr cx() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(2);
    write_tensor(tensor, {
      entry({0,0}, {0,0}, 1),
      entry({0,1}, {0,1}, 1),
      entry({1,0}, {1,1}, 1),
      entry({1,1}, {1,0}, 1),
    });
  }
  return tensor;
}

inline TensorPtr ccx() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(3);
    write_tensor(tensor, {
      entry({0,0,0}, {0,0,0}, 1),
      entry({0,0,1}, {0,0,1}, 1),
      entry({0,1,0}, {0,1,0}, 1),
      entry({0,1,1}, {0,1,1}, 1),
      entry({1,0,0}, {1,0,0}, 1),
      entry({1,0,1}, {1,0,1}, 1),
      entry({1,1,0}, {1,1,1}, 1),
      entry({1,1,1}, {1,1,0}, 1),
    });
  }
  return tensor;
}

inline TensorPtr swap() {
  static TensorPtr tensor = nullptr;
  if (!tensor) {
    tensor = make_gate_tensor(2);
    write_tensor(tensor, {
      entry({0,0}, {0,0}, 1),
      entry({0,1}, {1,0}, 1),
      entry({1,0}, {0,1}, 1),
      entry({1,1}, {1,1}, 1),
    });
  }
  return tensor;
}

// ----------------------------------------------------------------------------
// Nonconstant gates
// ----------------------------------------------------------------------------
inline TensorPtr u1(real_t lambda) {
  TensorPtr tensor = make_gate_tensor(1);
  write_tensor(tensor, {
    entry({0}, {0}, complex_t(1)),
    entry({1}, {1}, std::exp(complex_t(0, lambda))),
  });
  return tensor;
}

} // end namespace GateTensors
} // end namespace Cyclops
} // end namespace AER
#endif
