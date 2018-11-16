/**
 * Copyright 2018, IBM.
 *
 * This source code is licensed under the Apache License, Version 2.0 found in
 * the LICENSE.txt file in the root directory of this source tree.
 */

#ifndef _aer_qasm_controller_hpp_
#define _aer_qasm_controller_hpp_

#include "base/controller.hpp"
#include "simulators/qubitvector/qv_state.hpp"

namespace AER {
namespace Simulator {

//=========================================================================
// QasmController class
//=========================================================================

class QasmController : public Base::Controller {
private:

  //-----------------------------------------------------------------------
  // Base class abstract method override
  //-----------------------------------------------------------------------

  // Abstract method for executing a circuit.
  // This method must initialize a state and return output data for
  // the required number of shots.
  virtual OutputData run_circuit(const Circuit &circ,
                                 uint_t shots,
                                 uint_t rng_seed,
                                 int num_threads_state) const override;

  //----------------------------------------------------------------
  // Run circuit without optimization
  //----------------------------------------------------------------

  // Execute n-shots of a circuit on the input state
  template <class State_t>
  void run_circuit_default(const Circuit &circ,
                           uint_t shots,
                           State_t &state,
                           OutputData &data,
                           RngEngine &rng) const;

  //----------------------------------------------------------------
  // Measure sampling optimizations
  //----------------------------------------------------------------
  
  // Execute n-shots of a circuit performing measurement sampling
  // if the input circuit supports it
  template <class State_t>
  void run_circuit_measure_sampler(const Circuit &circ,
                                   uint_t shots,
                                   State_t &state,
                                   OutputData &data,
                                   RngEngine &rng) const;


  // Sample measurement outcomes for the input measure ops from the
  // current state of the input State_t
  template <class State_t>
  void measure_sampler(const std::vector<Operations::Op> &meas_ops,
                       uint_t shots,
                       State_t &state,
                       OutputData &data,
                       RngEngine &rng) const;

  // Check if measure sampling optimization if valid for the input circuit
  // if so return a pair {true, pos} where pos is the position of the
  // first measurement operation in the input circuit
  std::pair<bool, size_t> check_measure_sampling_opt(const Circuit &circ) const;

};

//=========================================================================
// Implementations
//=========================================================================

//-------------------------------------------------------------------------
// Base class override
//-------------------------------------------------------------------------

OutputData QasmController::run_circuit(const Circuit &circ,
                                      uint_t shots,
                                      uint_t rng_seed,
                                      int num_threads_state) const {  
  // Check if circuit can run on a statevector simulator
  // TODO: Should we make validate circuit a static method of the class?
  bool statevec_valid = QubitVector::State<>().validate_circuit(circ);  
  // throw exception listing the invalid instructions
  if (statevec_valid == false) {
    QubitVector::State<>().validate_circuit_except(circ);
  }

  // Initialize statevector
  QubitVector::State<> state;
  state.set_config(Base::Controller::state_config_);
  state.set_available_threads(num_threads_state);
  
  // Rng engine
  RngEngine rng;
  rng.set_seed(rng_seed);

  // Output data container
  OutputData data;
  data.set_config(Base::Controller::data_config_);
  
  // Check if there is noise for the implementation
  if (noise_model_.ideal()) {
    // Implement without noise
    run_circuit_measure_sampler(circ, shots, state, data, rng);
  } else {
    // Sample noise for each shot
    while (shots-- > 0) {
      Circuit noise_circ = noise_model_.sample_noise(circ, rng);
      run_circuit_default(noise_circ, 1, state, data, rng);
    }
  }
  return data;
}


//-------------------------------------------------------------------------
// Run circuit helpers
//-------------------------------------------------------------------------

template <class State_t>
void QasmController::run_circuit_default(const Circuit &circ,
                                         uint_t shots,
                                         State_t &state,
                                         OutputData &data,
                                         RngEngine &rng) const {  
  while (shots-- > 0) {
    state.initialize_qreg(circ.num_qubits);
    state.initialize_creg(circ.num_memory, circ.num_registers);
    state.apply_ops(circ.ops, data, rng);
    state.add_creg_to_data(data);
  }
}


template <class State_t>
void QasmController::run_circuit_measure_sampler(const Circuit &circ,
                                                 uint_t shots,
                                                 State_t &state,
                                                 OutputData &data,
                                                 RngEngine &rng) const {
  // Check if optimization is valid
  auto check = check_measure_sampling_opt(circ);
  // Perform standard execution if we cannot apply the optimization
  // or the execution is only for a single shot
  if (shots == 1 || check.first == false) {
    run_circuit_default(circ, shots, state, data, rng);
    return;
  } 
  auto pos = check.second; // Position of first measurement op
  // Run circuit instructions before first measure
  std::vector<Operations::Op> ops(circ.ops.begin(), circ.ops.begin() + pos);
  state.initialize_qreg(circ.num_qubits);
  state.initialize_creg(circ.num_memory, circ.num_registers);
  state.apply_ops(ops, data, rng);

  // Get measurement operations and set of measured qubits
  ops = std::vector<Operations::Op>(circ.ops.begin() + pos, circ.ops.end());
  measure_sampler(ops, shots, state, data, rng);
}


//-------------------------------------------------------------------------
// Measure sampling optimization
//-------------------------------------------------------------------------

std::pair<bool, size_t> 
QasmController::check_measure_sampling_opt(const Circuit &circ) const {
  // Find first instance of a measurement and check there
  // are no reset operations before the measurement
  auto start = circ.ops.begin();
  while (start != circ.ops.end()) {
    const auto type = start->type;
    if (type == Operations::OpType::reset ||
        type == Operations::OpType::kraus ||
        type == Operations::OpType::roerror) {
      return std::make_pair(false, 0);
    }
    if (type == Operations::OpType::measure)
      break;
    ++start;
  }
  // Record position for if optimization passes
  auto start_meas = start;
  // Check all remaining operations are measurements
  while (start != circ.ops.end()) {
    if (start->type != Operations::OpType::measure) {
      return std::make_pair(false, 0);
    }
    ++start;
  }
  // If we made it this far we can apply the optimization
  size_t meas_pos = start_meas - circ.ops.begin();
  return std::make_pair(true, meas_pos);
}


template <class State_t>
void QasmController::measure_sampler(const std::vector<Operations::Op> &meas_ops,
                                     uint_t shots,
                                     State_t &state,
                                     OutputData &data,
                                     RngEngine &rng) const {                    
  // Check if meas_circ is empty, and if so return initial creg
  if (meas_ops.empty()) {
    while (shots-- > 0) {
      state.add_creg_to_data(data);
    }
    return;
  }
  // Get measured qubits from circuit sort and delete duplicates
  std::vector<uint_t> meas_qubits; // measured qubits
  for (const auto &op : meas_ops) {
    for (size_t j=0; j < op.qubits.size(); ++j)
      meas_qubits.push_back(op.qubits[j]);
  }
  sort(meas_qubits.begin(), meas_qubits.end());
  meas_qubits.erase(unique(meas_qubits.begin(), meas_qubits.end()), meas_qubits.end());
  // Generate the samples
  auto all_samples = state.sample_measure(meas_qubits, shots, rng);

  // Make qubit map of position in vector of measured qubits
  std::unordered_map<uint_t, uint_t> qubit_map;
  for (uint_t j=0; j < meas_qubits.size(); ++j) {
      qubit_map[meas_qubits[j]] = j;
  }

  // Maps of memory and register to qubit position
  std::unordered_map<uint_t, uint_t> memory_map;
  std::unordered_map<uint_t, uint_t> register_map;
  for (const auto &op : meas_ops) {
    for (size_t j=0; j < op.qubits.size(); ++j) {
      auto pos = qubit_map[op.qubits[j]];
      if (!op.memory.empty())
        memory_map[op.memory[j]] = pos;
      if (!op.registers.empty())
        register_map[op.registers[j]] = pos;
    }
  }

  // Process samples
  // Convert opts to circuit so we can get the needed creg sizes
  // NB: this function could probably be moved somewhere else like Utils or Ops
  Circuit meas_circ(meas_ops);
  ClassicalRegister creg;
  while (!all_samples.empty()) {
    auto sample = all_samples.back();
    creg.initialize(meas_circ.num_memory, meas_circ.num_registers);

    // process memory bit measurements
    for (const auto &pair : memory_map) {
      creg.store_measure(reg_t({sample[pair.second]}), reg_t({pair.first}), reg_t());
    }
    auto memory = creg.memory_hex();
    data.add_memory_count(memory);
    data.add_memory_singleshot(memory);

    // process register bit measurements
    for (const auto &pair : register_map) {
      creg.store_measure(reg_t({sample[pair.second]}), reg_t(), reg_t({pair.first}));
    }
    data.add_register_singleshot(creg.register_hex());

    // pop off processed sample
    all_samples.pop_back();
  }
}

//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif