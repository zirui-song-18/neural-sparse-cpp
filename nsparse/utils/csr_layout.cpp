/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/utils/csr_layout.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "nsparse/types.h"

namespace nsparse::csr_layout {
namespace {

// Bounds conversion memory regardless of file size.
constexpr size_t kChunkElements = 1 << 16;

void read_or_throw(std::ifstream& file, void* dest, size_t bytes,
                   const std::string& path) {
    file.read(static_cast<char*>(dest), static_cast<std::streamsize>(bytes));
    if (!file) {
        throw std::runtime_error("Truncated CSR file: " + path);
    }
}

void write_or_throw(std::ofstream& file, const void* data, size_t bytes,
                    const std::string& path) {
    file.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(bytes));
    if (!file) {
        throw std::runtime_error("Failed writing CSR file: " + path);
    }
}

// Narrows `Wide` to `Narrow`, rejecting values that would wrap.
template <class Narrow, class Wide>
std::vector<Narrow> narrow(const std::vector<Wide>& source, const char* what,
                           const std::string& path) {
    std::vector<Narrow> result(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        if (source[i] < 0 ||
            static_cast<uint64_t>(source[i]) >
                static_cast<uint64_t>(std::numeric_limits<Narrow>::max())) {
            throw std::invalid_argument(std::string(what) +
                                        " does not fit the native CSR width in: " +
                                        path);
        }
        result[i] = static_cast<Narrow>(source[i]);
    }
    return result;
}

}  // namespace

std::string native_path(const std::string& path) { return path + kNativeSuffix; }

void convert(const std::string& interchange_path,
             const std::string& native_path) {
    if (!std::filesystem::exists(interchange_path)) {
        throw std::invalid_argument("CSR file does not exist: " +
                                    interchange_path);
    }
    std::ifstream in(interchange_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open CSR file: " + interchange_path);
    }

    std::array<int64_t, 3> header{};
    read_or_throw(in, header.data(), sizeof(header), interchange_path);
    const int64_t num_rows = header[0];
    const int64_t num_cols = header[1];
    const int64_t nnz = header[2];
    if (num_rows <= 0 || num_cols <= 0 || nnz < 0) {
        throw std::invalid_argument("Invalid CSR header in: " +
                                    interchange_path);
    }

    const auto indptr_size = static_cast<size_t>(num_rows) + 1;
    const auto nnz_size = static_cast<size_t>(nnz);
    if (std::filesystem::file_size(interchange_path) !=
        interchange_file_size(indptr_size, nnz_size)) {
        throw std::invalid_argument(
            "CSR file is not in the interchange layout: " + interchange_path);
    }

    std::ofstream out(native_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open CSR file for writing: " +
                                 native_path);
    }
    write_or_throw(out, header.data(), sizeof(header), native_path);

    // Read whole: one entry per row, and its last element gates the rest.
    std::vector<int64_t> wide_indptr(indptr_size);
    read_or_throw(in, wide_indptr.data(), indptr_size * sizeof(int64_t),
                  interchange_path);
    if (wide_indptr.front() != 0 || wide_indptr.back() != nnz) {
        throw std::invalid_argument("Inconsistent CSR indptr in: " +
                                    interchange_path);
    }
    // The interchange indptr is already int64 (== offset_t), so it is written
    // straight through rather than copied. A negative entry is still rejected
    // (map_vectors re-validates monotonicity when the native file is loaded).
    for (const int64_t off : wide_indptr) {
        if (off < 0) {
            throw std::invalid_argument(
                "CSR indptr has a negative offset in: " + interchange_path);
        }
    }
    write_or_throw(out, wide_indptr.data(),
                   wide_indptr.size() * sizeof(offset_t), native_path);

    // The large array; stream it.
    for (size_t done = 0; done < nnz_size;) {
        const size_t count = std::min(kChunkElements, nnz_size - done);
        std::vector<int32_t> wide(count);
        read_or_throw(in, wide.data(), count * sizeof(int32_t),
                      interchange_path);
        for (const int32_t term : wide) {
            if (term >= num_cols) {
                throw std::invalid_argument("CSR term out of range in: " +
                                            interchange_path);
            }
        }
        const auto narrowed = narrow<term_t>(wide, "term", interchange_path);
        write_or_throw(out, narrowed.data(), count * sizeof(term_t),
                       native_path);
        done += count;
    }

    const size_t values_pos = kHeaderBytes +
                              indptr_size * sizeof(offset_t) +
                              nnz_size * sizeof(term_t);
    const std::array<uint8_t, alignof(float)> pad{};
    if (const size_t pad_bytes = padding(values_pos); pad_bytes > 0) {
        write_or_throw(out, pad.data(), pad_bytes, native_path);
    }

    // Same width in both layouts, so copied through unchanged.
    for (size_t done = 0; done < nnz_size;) {
        const size_t count = std::min(kChunkElements, nnz_size - done);
        std::vector<float> values(count);
        read_or_throw(in, values.data(), count * sizeof(float),
                      interchange_path);
        write_or_throw(out, values.data(), count * sizeof(float), native_path);
        done += count;
    }

    out.flush();
    if (!out) {
        throw std::runtime_error("Failed writing CSR file: " + native_path);
    }
}

}  // namespace nsparse::csr_layout
