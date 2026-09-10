/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/sparse_vectors.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nsparse/io/align.h"
#include "nsparse/io/io.h"
#include "nsparse/types.h"
#include "nsparse/utils/checks.h"
#include "nsparse/utils/mmap_file.h"

namespace nsparse {
SparseVectors::SparseVectors(SparseVectorsConfig config) : config_(config) {
    throw_if_not_positive(config_.dimension);
}

SparseVectors SparseVectors::map_vectors(SparseVectorsConfig config,
                                         const offset_t* indptr,
                                         size_t indptr_size,
                                         const term_t* indices,
                                         size_t indices_size,
                                         const uint8_t* values,
                                         size_t values_size) {
    throw_if_not_positive(config.dimension);
    if (config.element_size != U8 && config.element_size != U16 &&
        config.element_size != U32 && config.element_size != U64) {
        throw std::invalid_argument("Invalid element size");
    }

    SparseVectors vectors;
    vectors.config_ = config;
    if (indptr_size == 0) {
        return vectors;  // nothing mapped
    }
    throw_if_null(indptr, "Mapped indptr must not be null");
    // A matrix whose every row is empty has no indices and no values, and an
    // empty array has no address to hand out; only a non-empty one must be real.
    if (indices_size > 0) {
        throw_if_any_null(indices, values);
    }

    if (indices_size * config.element_size != values_size) {
        throw std::invalid_argument(
            "Indices and weights must have the same size");
    }
    // Reinterpreted in place, so a misaligned start is UB on x86 and faults on
    // ARM. Writers pad the array up to element_size to keep this satisfiable.
    if (reinterpret_cast<uintptr_t>(values) % config.element_size != 0) {
        throw std::invalid_argument(
            "Mapped values are misaligned for the element size");
    }

    // Search dereferences indptr and indices to reach a row, so validate once
    // here rather than per query.
    if (indptr[0] != 0) {
        throw std::invalid_argument("Mapped indptr must start at 0");
    }
    for (size_t i = 1; i < indptr_size; ++i) {
        if (indptr[i] < indptr[i - 1]) {
            throw std::invalid_argument("Mapped indptr is not monotonic");
        }
    }
    if (static_cast<size_t>(indptr[indptr_size - 1]) != indices_size) {
        throw std::invalid_argument(
            "Mapped indptr does not end at the index count");
    }

    vectors.indptr_ = Buf<offset_t>::borrow(indptr, indptr_size);
    vectors.indices_ = Buf<term_t>::borrow(indices, indices_size);
    vectors.values_ = Buf<uint8_t>::borrow(values, values_size);
    return vectors;
}

void SparseVectors::add_vectors(const std::vector<offset_t>& indptr,
                                const std::vector<term_t>& indices,
                                const std::vector<uint8_t>& weights) {
    add_vectors(indptr.data(), indptr.size(), indices.data(), indices.size(),
                weights.data(), weights.size());
}

void SparseVectors::add_vectors(const offset_t* indptr, size_t indptr_size,
                                const term_t* indices, size_t indices_size,
                                const uint8_t* weights, size_t weights_size) {
    if (indices_size * config_.element_size != weights_size) {
        throw std::invalid_argument(
            "Indices and weights must have the same size");
    }
    if (indptr_size < 2) {
        return;  // Nothing to add
    }
    // Always copies: the arguments are often a caller-local buffer (freshly
    // quantized codes, say), and the incoming offsets are rebased onto what is
    // already stored. Borrowing is map_vectors' job.
    std::vector<offset_t> indptr_vec = indptr_.take_vector();
    std::vector<term_t> indices_vec = indices_.take_vector();
    std::vector<uint8_t> values_vec = values_.take_vector();

    if (indptr_vec.empty()) {
        indptr_vec.push_back(0);
    }

    // Append new indices
    indices_vec.insert(indices_vec.end(), indices, indices + indices_size);

    // Append weights directly (already in uint8_t format)
    values_vec.insert(values_vec.end(), weights, weights + weights_size);
    offset_t offset = indptr_vec.back();
    for (size_t i = 1; i < indptr_size; ++i) {
        indptr_vec.push_back(indptr[i] + offset);
    }

    indptr_ = Buf<offset_t>::own(std::move(indptr_vec));
    indices_ = Buf<term_t>::own(std::move(indices_vec));
    values_ = Buf<uint8_t>::own(std::move(values_vec));
}

void SparseVectors::add_vector(const std::vector<term_t>& indices,
                               const std::vector<uint8_t>& weights) {
    add_vector(indices.data(), indices.size(), weights.data(), weights.size());
}

void SparseVectors::add_vector(const term_t* indices, size_t indices_size,
                               const uint8_t* weights, size_t weights_size) {
    // Copies for the same reason add_vectors does; see the note there.
    std::vector<offset_t> indptr_vec = indptr_.take_vector();
    std::vector<term_t> indices_vec = indices_.take_vector();
    std::vector<uint8_t> values_vec = values_.take_vector();

    // Get the current offset (where the new vector starts)
    offset_t offset = indptr_vec.empty() ? 0 : indptr_vec.back();

    // If this is the first vector, initialize indptr with 0
    if (indptr_vec.empty()) {
        indptr_vec.push_back(0);
    }

    indices_vec.insert(indices_vec.end(), indices, indices + indices_size);
    values_vec.insert(values_vec.end(), weights, weights + weights_size);
    indptr_vec.push_back(offset + static_cast<offset_t>(indices_size));

    indptr_ = Buf<offset_t>::own(std::move(indptr_vec));
    indices_ = Buf<term_t>::own(std::move(indices_vec));
    values_ = Buf<uint8_t>::own(std::move(values_vec));
}

std::vector<float> SparseVectors::get_dense_vector_float(
    idx_t vector_idx) const {
    if (vector_idx < 0 || vector_idx > static_cast<idx_t>(indptr_.size()) - 2) {
        throw std::out_of_range("Vector index out of range");
    }

    offset_t start = indptr_[vector_idx];
    offset_t end = indptr_[vector_idx + 1];
    std::vector<float> dense_vector(
        config_.dimension > 0 ? config_.dimension : indices_[end - 1] + 1,
        0.0F);
    for (offset_t i = start; i < end; ++i) {
        const uint8_t* value_ptr = values_.data() + (i * config_.element_size);
        if (config_.element_size == U32) {
            dense_vector[indices_[i]] =
                *reinterpret_cast<const float*>(value_ptr);
        } else if (config_.element_size == U16) {
            dense_vector[indices_[i]] = static_cast<float>(
                *reinterpret_cast<const uint16_t*>(value_ptr));
        } else {
            dense_vector[indices_[i]] = static_cast<float>(*value_ptr);
        }
    }
    return dense_vector;
}

std::vector<uint8_t> SparseVectors::get_dense_vector(idx_t vector_idx) const {
    if (vector_idx < 0 || vector_idx > static_cast<idx_t>(indptr_.size()) - 2) {
        throw std::out_of_range("Vector index out of range");
    }
    offset_t start = indptr_[vector_idx];
    offset_t end = indptr_[vector_idx + 1];
    size_t size = end - start;
    std::vector<uint8_t> dense_vector(config_.dimension * config_.element_size,
                                      0.0F);
    for (offset_t i = start; i < end; ++i) {
        for (idx_t j = 0; j < config_.element_size; ++j) {
            dense_vector[indices_[i] * config_.element_size + j] =
                values_[i * config_.element_size + j];
        }
    }
    return dense_vector;
}

size_t SparseVectors::num_vectors() const {
    if (indptr_.empty()) return 0;
    return indptr_.size() - 1;
}

void SparseVectors::serialize(IOWriter* io_writer) const {
    size_t vector_count = num_vectors();
    io_writer->write(&vector_count, sizeof(size_t), 1);
    if (vector_count > 0) {
        auto dimension = get_dimension();
        io_writer->write(&dimension, sizeof(size_t), 1);
        auto element_size = get_element_size();
        io_writer->write(&element_size, sizeof(size_t), 1);

        // Each array is padded to its own alignment so a mapped reader can
        // borrow it in place; see io/align.h. The header above is three size_t,
        // so indptr needs none in practice, but the layout should not depend on
        // that.
        size_t indptr_size = vector_count + 1;
        io_align::write_padded(io_writer, indptr_.data(), indptr_size);

        size_t indices_size = indptr_[vector_count];
        io_align::write_padded(io_writer, indices_.data(), indices_size);

        // Values are bytes on the wire but reinterpreted as element_size-wide
        // words on read, so they are padded to that width, not to 1.
        size_t value_size = indices_size * element_size;
        io_align::write_padded(io_writer, values_.data(), value_size,
                               element_size);
    }
}

void SparseVectors::deserialize(IOReader* io_reader) {
    size_t vector_count = 0;
    io_reader->read(&vector_count, sizeof(size_t), 1);
    if (vector_count > 0) {
        size_t dimension = 0;
        io_reader->read(&dimension, sizeof(size_t), 1);
        size_t element_size = 0;
        io_reader->read(&element_size, sizeof(size_t), 1);
        config_ = SparseVectorsConfig(element_size, dimension);

        // Skips the padding serialize() wrote before each array.
        size_t indptr_size = vector_count + 1;
        indptr_ = io_align::read_padded<offset_t>(io_reader, indptr_size);

        size_t indices_size = indptr_[vector_count];
        indices_ = io_align::read_padded<term_t>(io_reader, indices_size);

        size_t value_size = checked_mul(indices_size, element_size);
        values_ = io_align::read_padded<uint8_t>(io_reader, value_size, element_size);
    }
}

void SparseVectors::mmap_deserialize(MmapCursor* cursor) {
    throw_if_null(cursor, "cursor must not be null");
    // Same layout deserialize() walks, borrowed rather than copied. The scalars
    // are still read out: they are small, and copying them keeps the mapped
    // object independent of the file for everything but the arrays.
    const auto vector_count = cursor->read_scalar<size_t>();
    if (vector_count == 0) {
        *this = SparseVectors();
        return;
    }
    const auto dimension = cursor->read_scalar<size_t>();
    const auto element_size = cursor->read_scalar<size_t>();

    // SIZE_MAX vectors wrap this to an empty indptr, and its borrow is the null
    // pointer the row-count read below would dereference.
    const size_t indptr_size = vector_count + 1;
    if (indptr_size == 0) {
        throw std::runtime_error("mmap: implausible vector count in index file");
    }
    io_align::skip_padding(cursor, alignof(offset_t));
    const offset_t* indptr = cursor->read_array<offset_t>(indptr_size);

    const auto indices_size = static_cast<size_t>(indptr[vector_count]);
    io_align::skip_padding(cursor, alignof(term_t));
    const term_t* indices = cursor->read_array<term_t>(indices_size);

    const size_t value_size = checked_mul(indices_size, element_size);
    io_align::skip_padding(cursor, element_size);
    const uint8_t* values = cursor->read_array<uint8_t>(value_size);

    // map_vectors validates before borrowing, so a corrupt file throws here
    // rather than faulting mid-search.
    *this = map_vectors({.element_size = element_size, .dimension = dimension},
                        indptr, indptr_size, indices, indices_size, values,
                        value_size);
}
}  // namespace nsparse