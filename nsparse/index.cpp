/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/index.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/csr_layout.h"

namespace nsparse {

Index::Index(int dim) : dimension_(dim) {}

void Index::build() { throw_not_implemented(); }

void Index::search(idx_t n, const offset_t* indptr, const term_t* indices,
                   const float* values, int k, float* distances, idx_t* labels,
                   SearchParameters* search_parameters) {
    throw_if_not_positive(n);
    throw_if_not_positive(k);
    throw_if_any_null(indptr, indices, values, labels, distances);

    auto [result_distances, result_labels] =
        search(n, indptr, indices, values, k, search_parameters);

    idx_t* dest_labels = labels;
    float* dest_distances = distances;
    for (size_t i = 0; i < result_labels.size(); ++i) {
        dest_distances =
            std::ranges::copy(result_distances[i], dest_distances).out;
        dest_labels = std::ranges::copy(result_labels[i], dest_labels).out;
    }
}

auto Index::search(idx_t n, const offset_t* indptr, const term_t* indices,
                   const float* values, int k,
                   SearchParameters* search_parameters)
    -> pair_of_score_id_vectors_t {
    throw_not_implemented("search not implementted in Index");
}

void Index::add_with_ids(idx_t n, const offset_t* indptr, const term_t* indices,
                         const float* values, const idx_t* ids) {
    throw_not_implemented("add_with_ids not implemented in Index");
}

void Index::read_csr(const char* file_path, Residency residency) {
    throw_if_null(file_path, "file_path must not be null");
    if (!std::filesystem::exists(file_path)) {
        throw std::invalid_argument(std::string("CSR file does not exist: ") +
                                    file_path);
    }

    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Cannot open CSR file: ") +
                                 file_path);
    }

    auto read_or_throw = [&file, file_path](void* dest, size_t bytes) {
        file.read(static_cast<char*>(dest),
                  static_cast<std::streamsize>(bytes));
        if (!file) {
            throw std::runtime_error(std::string("Truncated CSR file: ") +
                                     file_path);
        }
    };

    std::array<int64_t, 3> header{};
    read_or_throw(header.data(), sizeof(header));
    const int64_t num_rows = header[0];
    const int64_t num_cols = header[1];
    const int64_t nnz = header[2];

    if (num_rows <= 0 || num_cols <= 0 || nnz < 0) {
        throw std::invalid_argument(std::string("Invalid CSR header in: ") +
                                    file_path);
    }
    if (num_rows > std::numeric_limits<idx_t>::max()) {
        throw std::invalid_argument(
            std::string("CSR row count exceeds 32-bit doc-id range: ") +
            file_path);
    }
    if (num_cols > dimension_) {
        throw std::invalid_argument(
            std::string("CSR column count exceeds index dimension: ") +
            file_path);
    }

    const size_t indptr_size = static_cast<size_t>(num_rows) + 1;
    const auto nnz_size = static_cast<size_t>(nnz);
    // Interchange layout only; a native file is rejected rather than guessed at.
    if (std::filesystem::file_size(file_path) !=
        csr_layout::interchange_file_size(indptr_size, nnz_size)) {
        throw std::invalid_argument(
            std::string("CSR file is not in the interchange layout: ") +
            file_path);
    }

    std::vector<int64_t> file_indptr(indptr_size);
    read_or_throw(file_indptr.data(), file_indptr.size() * sizeof(int64_t));
    if (file_indptr.front() != 0 || file_indptr.back() != nnz) {
        throw std::invalid_argument(
            std::string("Inconsistent CSR indptr in: ") + file_path);
    }

    std::vector<int32_t> file_indices(nnz_size);
    read_or_throw(file_indices.data(), file_indices.size() * sizeof(int32_t));
    // Narrowing to term_t would wrap silently. Affordable here because every
    // element is being copied anyway; the mapped reader skips it.
    if (std::ranges::any_of(file_indices, [num_cols](int32_t term) {
            return term < 0 || term >= num_cols ||
                   term > std::numeric_limits<term_t>::max();
        })) {
        throw std::invalid_argument(std::string("CSR term out of range in: ") +
                                    file_path);
    }
    std::vector<term_t> indices(file_indices.begin(), file_indices.end());

    std::vector<float> values(nnz_size);
    read_or_throw(values.data(), values.size() * sizeof(float));

    add(static_cast<idx_t>(num_rows), file_indptr.data(), indices.data(),
        values.data());
}
}  // namespace nsparse
