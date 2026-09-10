/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <vector>

namespace nsparse {

using idx_t = int32_t;
using offset_t = int64_t;  // CSR nnz offsets (indptr); distinct from idx_t so
                           // doc-ids / term-ids / counts stay 32-bit.
using term_t = uint16_t;
using weight_t = float;

// Whether a loaded index owns its buffers or borrows them from a file mapping.
// Lives here rather than in seismic_common.h because Index::read_csr takes it,
// and that header pulls in the SIMD distance kernels: including it from index.h
// puts those definitions in every translation unit, where they collide with
// distance.h.
enum class Residency : uint8_t {
    kInMemory,
    kMmap,
};

template <class T>
using pair_of_score_id_vector_t_t =
    std::pair<std::vector<float>, std::vector<T>>;
using pair_of_score_id_vector_t = pair_of_score_id_vector_t_t<idx_t>;
using pair_of_score_id_vectors_t =
    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<idx_t>>>;

namespace detail {
constexpr idx_t INVALID_IDX = -1;
}

}  // namespace nsparse

#endif  // TYPES_H