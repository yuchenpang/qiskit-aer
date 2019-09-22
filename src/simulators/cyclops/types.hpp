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

#ifndef _aer_cyclops_types_hpp
#define _aer_cyclops_types_hpp

#include <cmath>
#include <string>
#include <memory>

#include <ctf.hpp>

namespace AER {
namespace Cyclops {

using real_t = double;
using complex_t = std::complex<real_t>;
using Tensor = CTF::Tensor<complex_t>;
using Pair = CTF::Pair<complex_t>;
using TensorPtr = std::shared_ptr<Tensor>;

template<typename... Args>
TensorPtr make_tensor(Args... args) {
    return std::make_shared<Tensor>(args...);
}

} // end namespace Cyclops
} // end namespace AER
#endif
