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

#ifdef WITH_CTF
#ifndef _aer_cyclops_state_hpp
#define _aer_cyclops_state_hpp

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include "framework/utils.hpp"
#include "framework/json.hpp"
#include "base/state.hpp"
#include "statevector.hpp"
#include "mpi_util.hpp"

namespace AER {
namespace Cyclops {

// Allowed gates enum class
enum class Gates {
  id, x, y, z, h, s, sdg, t, tdg, u1,
  swap, cx, ccx
};

// Allowed snapshots enum class
enum class Snapshots {
  cmemory, cregister
  // statevector,
  // probs, probs_var,
  // expval_pauli, expval_pauli_var, expval_pauli_shot,
  // expval_matrix, expval_matrix_var, expval_matrix_shot
};

// Enum class for different types of expectation values
enum class SnapshotDataType {average, average_var, single_shot};


class State : public Base::State<StateVector> {
public:
  using BaseState = Base::State<StateVector>;

  State() = default;
  virtual ~State() = default;

  //-----------------------------------------------------------------------
  // Base class overrides
  //-----------------------------------------------------------------------

  // Return the string name of the State class
  virtual std::string name() const override {return "cyclops";}

  // Return the set of qobj instruction types supported by the State
  virtual Operations::OpSet::optypeset_t allowed_ops() const override {
    return Operations::OpSet::optypeset_t({
      Operations::OpType::gate,
      Operations::OpType::measure,
      Operations::OpType::reset,
      Operations::OpType::initialize,
      Operations::OpType::snapshot,
      Operations::OpType::barrier,
      Operations::OpType::bfunc,
      Operations::OpType::roerror
    });
  }

  // Return the set of qobj gate instruction names supported by the State
  virtual stringset_t allowed_gates() const override {
    return {"id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "u1",
            "swap", "cx", "ccx"};
  }

  // Return the set of qobj snapshot types supported by the State
  virtual stringset_t allowed_snapshots() const override {
    return {"memory", "register"};
  }

  // Apply a sequence of operations by looping over list
  // If the input is not in allowed_ops an exeption will be raised.
  virtual void apply_ops(const std::vector<Operations::Op> &ops,
                         ExperimentData &data,
                         RngEngine &rng) override;

  // Initializes an n-qubit state to the all |0> state
  virtual void initialize_qreg(uint_t num_qubits) override;

  // Initializes to a specific n-qubit state
  virtual void initialize_qreg(uint_t num_qubits,
                               const StateVector &state) override;

  // Returns the required memory for storing an n-qubit state in megabytes.
  // For this state the memory is indepdentent of the number of ops
  // and is approximately 16 * 1 << num_qubits bytes
  virtual size_t required_memory_mb(uint_t num_qubits,
                                    const std::vector<Operations::Op> &ops)
                                    const override;

  // Load the threshold for applying OpenMP parallelization
  // if the controller/engine allows threads for it
  virtual void set_config(const json_t &config) override;

  // Sample n-measurement outcomes without applying the measure operation
  // to the system state
  virtual std::vector<reg_t> sample_measure(const reg_t& qubits,
                                            uint_t shots,
                                            RngEngine &rng) override;


protected:

  //-----------------------------------------------------------------------
  // Apply instructions
  //-----------------------------------------------------------------------

  // Applies a sypported Gate operation to the state class.
  // If the input is not in allowed_gates an exeption will be raised.
  void apply_gate(const Operations::Op &op);

  // Measure qubits and return a list of outcomes [q0, q1, ...]
  // If a state subclass supports this function it then "measure"
  // should be contained in the set returned by the 'allowed_ops'
  // method.
  virtual void apply_measure(const reg_t &qubits,
                             const reg_t &cmemory,
                             const reg_t &cregister,
                             RngEngine &rng);

  // Reset the specified qubits to the |0> state by simulating
  // a measurement, applying a conditional x-gate if the outcome is 1, and
  // then discarding the outcome.
  void apply_reset(const reg_t &qubits, RngEngine &rng);

  // Initialize the specified qubits to a given state |psi>
  // by applying a reset to the these qubits and then
  // computing the tensor product with the new state |psi>
  // /psi> is given in params
  void apply_initialize(const reg_t &qubits, const cvector_t &params, RngEngine &rng);

  // Apply a supported snapshot instruction
  // If the input is not in allowed_snapshots an exeption will be raised.
  virtual void apply_snapshot(const Operations::Op &op, ExperimentData &data);

  //-----------------------------------------------------------------------
  // Config Settings
  //-----------------------------------------------------------------------

  // Threshold for chopping small values to zero
  double zero_threshold_ = 1e-10;

  // Table of allowed gate names to gate enum class members
  const static stringmap_t<Gates> gateset_;

  // Table of allowed snapshot types to enum class members
  const static stringmap_t<Snapshots> snapshotset_;
};


//=========================================================================
// Implementation: Allowed ops and gateset
//=========================================================================

const stringmap_t<Gates> State::gateset_({
  {"id", Gates::id},      // Pauli-Identity gate
  {"x", Gates::x},        // Pauli-X gate
  {"y", Gates::y},        // Pauli-Y gate
  {"z", Gates::z},        // Pauli-Z gate
  {"h", Gates::h},        // Hadamard gate (X + Z / sqrt(2))
  {"s", Gates::s},        // Phase gate (aka sqrt(Z) gate)
  {"sdg", Gates::sdg},    // Conjugate-transpose of Phase gate
  {"t", Gates::t},        // T-gate (sqrt(S))
  {"tdg", Gates::tdg},    // Conjguate-transpose of T gate
  {"u1", Gates::u1},      // zero-X90 pulse waltz gate
  {"cx", Gates::cx},      // Controlled-X gate (CNOT)
  {"swap", Gates::swap},  // SWAP gate
  {"ccx", Gates::ccx},    // Controlled-CX gate (Toffoli)
});

const stringmap_t<Snapshots> State::snapshotset_({
  {"memory", Snapshots::cmemory},
  {"register", Snapshots::cregister}
});

//=========================================================================
// Implementation: Base class method overrides
//=========================================================================

//-------------------------------------------------------------------------
// Initialization
//-------------------------------------------------------------------------

inline void State::initialize_qreg(uint_t num_qubits) {
  BaseState::qreg_.initialize(num_qubits);
}

inline void State::initialize_qreg(uint_t num_qubits, const StateVector &state) {
  // Check dimension of state
  if (state.num_qubits() != num_qubits) {
    throw std::invalid_argument("Cyclops::State::initialize: initial state does not match qubit number");
  }
  BaseState::qreg_ = state;
}

inline std::vector<reg_t> State::sample_measure(const reg_t& qubits,
                                         uint_t shots,
                                         RngEngine &rng) {
  // TODO implement sample_measure
  (void) qubits;
  (void) shots;
  (void) rng;
  return {};
}

//-------------------------------------------------------------------------
// Utility
//-------------------------------------------------------------------------

inline size_t State::required_memory_mb(uint_t num_qubits,
                                 const std::vector<Operations::Op> &ops)
                                 const {
  (void)ops; // avoid unused variable compiler warning
  return (1UL << num_qubits) * sizeof(complex_t) / size_t(MPIUtil::num_processes());
}

void State::set_config(const json_t &config) {
  // Set threshold for truncating snapshots
  JSON::get_value(zero_threshold_, "zero_threshold", config);
}

//=========================================================================
// Implementation: apply operations
//=========================================================================

inline void State::apply_ops(const std::vector<Operations::Op> &ops,
                             ExperimentData &data,
                             RngEngine &rng) {
  // Simple loop over vector of input operations
  for (const auto & op: ops) {
    if(BaseState::creg_.check_conditional(op)) {
      switch (op.type) {
        case Operations::OpType::barrier:
          break;
        case Operations::OpType::bfunc:
          BaseState::creg_.apply_bfunc(op);
          break;
        case Operations::OpType::roerror:
          BaseState::creg_.apply_roerror(op, rng);
          break;
        case Operations::OpType::gate:
          apply_gate(op);
          break;
        case Operations::OpType::snapshot:
          apply_snapshot(op, data);
          break;
        default:
          throw std::invalid_argument("Cyclops::State::invalid instruction \'" +
                                      op.name + "\'.");
      }
    }
  }
}

//=========================================================================
// Implementation: Snapshots
//=========================================================================

inline void State::apply_snapshot(const Operations::Op &op, ExperimentData &data) {
  // Look for snapshot type in snapshotset
  auto it = snapshotset_.find(op.name);
  if (it == snapshotset_.end())
    throw std::invalid_argument("Cyclops::State::invalid snapshot instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
    case Snapshots::cmemory:
      BaseState::snapshot_creg_memory(op, data);
      break;
    case Snapshots::cregister:
      BaseState::snapshot_creg_register(op, data);
      break;
    default:
      // We shouldn't get here unless there is a bug in the snapshotset
      throw std::invalid_argument("Cyclops::State::invalid snapshot instruction \'" +
                                  op.name + "\'.");
  }
}

//=========================================================================
// Implementation: Gate application
//=========================================================================

inline void State::apply_gate(const Operations::Op &op) {
  // Look for gate name in gateset
  auto it = gateset_.find(op.name);
  if (it == gateset_.end())
    throw std::invalid_argument("Cyclops::State::invalid gate instruction \'" + 
                                op.name + "\'.");
  switch (it -> second) {
    case Gates::id:
      break;
    case Gates::x:
      BaseState::qreg_.apply_x(op.qubits[0]);
      break;
    case Gates::y:
      BaseState::qreg_.apply_y(op.qubits[0]);
      break;
    case Gates::z:
      BaseState::qreg_.apply_z(op.qubits[0]);
      break;
    case Gates::h:
      BaseState::qreg_.apply_h(op.qubits[0]);
      break;
    case Gates::s:
      BaseState::qreg_.apply_s(op.qubits[0]);
      break;
    case Gates::sdg:
      BaseState::qreg_.apply_sdg(op.qubits[0]);
      break;
    case Gates::t:
      BaseState::qreg_.apply_t(op.qubits[0]);
      break;
    case Gates::tdg:
      BaseState::qreg_.apply_tdg(op.qubits[0]);
      break;
    case Gates::u1:
      BaseState::qreg_.apply_u1(op.qubits[0], std::real(op.params[0]));
      break;
    case Gates::cx:
      BaseState::qreg_.apply_cx(op.qubits[0], op.qubits[1]);
      break;
    case Gates::ccx:
      BaseState::qreg_.apply_ccx(op.qubits[0], op.qubits[1], op.qubits[2]);
      break;
    case Gates::swap:
      BaseState::qreg_.apply_swap(op.qubits[0], op.qubits[1]);
      break;
    default:
      // We shouldn't reach here unless there is a bug in gateset
      throw std::invalid_argument("Cyclops::State::invalid gate instruction \'" +
                                  op.name + "\'.");
  }
}

//=========================================================================
// Implementation: Reset, Initialize and Measurement Sampling
//=========================================================================

inline void State::apply_measure(const reg_t &qubits, const reg_t &cmemory,
                          const reg_t &cregister, RngEngine &rng) {
  // TODO implement measure
  (void) qubits;
  (void) cmemory;
  (void) cregister;
  (void) rng;
}

inline void State::apply_reset(const reg_t &qubits, RngEngine &rng) {
  // TODO implement reset
  (void) qubits;
  (void) rng;
}

inline void State::apply_initialize(const reg_t &qubits, const cvector_t &params,
                             RngEngine &rng) {
  // TODO implement initialize
  (void) qubits;
  (void) params;
  (void) rng;
}

//-------------------------------------------------------------------------
} // end namespace Cyclops
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
#endif
