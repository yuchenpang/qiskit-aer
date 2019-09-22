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

#ifndef _aer_cyclops_statevector_hpp
#define _aer_cyclops_statevector_hpp

#include <string>
#include <vector>

#include "types.hpp"
#include "mpi_util.hpp"
#include "gate_tensors.hpp"

namespace AER {
namespace Cyclops {

class StateVector {
public:
  StateVector();

  void initialize(uint_t num_qubits);

  uint_t num_qubits() const;
  bool empty() const;

  void apply_x(uint_t qubit);
  void apply_y(uint_t qubit);
  void apply_z(uint_t qubit);
  void apply_h(uint_t qubit);
  void apply_s(uint_t qubit);
  void apply_sdg(uint_t qubit);
  void apply_t(uint_t qubit);
  void apply_tdg(uint_t qubit);
  void apply_u1(uint_t qubit, real_t lambda);
  void apply_cx(uint_t ctrl, uint_t qubit);
  void apply_ccx(uint_t ctrl1, uint_t ctrl2, uint_t qubit);
  void apply_swap(uint_t qubit1, uint_t qubit2);

private:
  template<int gate_size>
  void apply_gate(TensorPtr gate, uint_t const (&qubits)[gate_size]);

private:
  Tensor state_;
};

//=========================================================================
// Implementation
//=========================================================================

inline StateVector::StateVector() {
  MPIUtil::initialize();
}

inline void StateVector::initialize(uint_t num_qubits) {
  state_ = Tensor(num_qubits, std::vector<int>(num_qubits, 2).data());
  // TODO let each process write to its local data
	if (state_.wrld->rank == 0) {
    Pair data[] = { Pair(0, 1) };
	  state_.write(1, data);
	} else {
	  state_.write(0, nullptr);
	}
}

inline uint_t StateVector::num_qubits() const {
  return state_.order == -1 ? 0 : state_.order;
}

inline bool StateVector::empty() const {
  return num_qubits() == 0;
}

inline void StateVector::apply_x(uint_t qubit) {
  apply_gate(GateTensors::x(), {qubit});
}

inline void StateVector::apply_y(uint_t qubit) {
  apply_gate(GateTensors::y(), {qubit});
}

inline void StateVector::apply_z(uint_t qubit) {
  apply_gate(GateTensors::z(), {qubit});
}

inline void StateVector::apply_h(uint_t qubit) {
  apply_gate(GateTensors::h(), {qubit});
}

inline void StateVector::apply_s(uint_t qubit) {
  apply_gate(GateTensors::s(), {qubit});
}

inline void StateVector::apply_sdg(uint_t qubit) {
  apply_gate(GateTensors::sdg(), {qubit});
}

inline void StateVector::apply_t(uint_t qubit) {
  apply_gate(GateTensors::t(), {qubit});
}

inline void StateVector::apply_tdg(uint_t qubit) {
  apply_gate(GateTensors::tdg(), {qubit});
}

inline void StateVector::apply_cx(uint_t ctrl, uint_t qubit) {
  apply_gate(GateTensors::cx(), {ctrl, qubit});
}

inline void StateVector::apply_ccx(uint_t ctrl1, uint_t ctrl2, uint_t qubit) {
  apply_gate(GateTensors::ccx(), {ctrl1, ctrl2, qubit});
}

inline void StateVector::apply_swap(uint_t qubit1, uint_t qubit2) {
  apply_gate(GateTensors::swap(), {qubit1, qubit2});
}

inline void StateVector::apply_u1(uint_t qubit, real_t lambda) {
  apply_gate(GateTensors::u1(lambda), {qubit});
}

template<int gate_size>
inline void StateVector::apply_gate(TensorPtr gate, uint_t const (&qubits)[gate_size]) {
  std::string in_idx(state_.order, 0);
	char idx = 1;
	for (int i = 0; i < state_.order; i++) {
	  in_idx[i] = idx++;
	}
	std::string out_idx(in_idx);
	std::string gate_idx(gate_size, 0);
	for (int i = 0; i < gate_size; i++) {
	  out_idx[qubits[i]] = idx++;
	  gate_idx[i] = in_idx[qubits[i]];
	  gate_idx[gate_size+i] = out_idx[qubits[i]];
	}
	state_[out_idx.c_str()] = state_[in_idx.c_str()] * (*gate)[gate_idx.c_str()];
}


} // end namespace Cyclops
} // end namespace AER
#endif
