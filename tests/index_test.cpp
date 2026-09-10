/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/index.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <fstream>
#include <string>
#include <vector>

#include "nsparse/types.h"

namespace {

// Records add() arguments so read_csr can be checked without depending on a
// concrete index implementation.
class RecordingIndex : public nsparse::Index {
public:
    explicit RecordingIndex(int dim) : Index(dim) {}

    std::array<char, 4> id() const override { return {'T', 'E', 'S', 'T'}; }

    void add(nsparse::idx_t n, const nsparse::offset_t* indptr,
             const nsparse::term_t* indices, const float* values) override {
        num_added = n;
        this->indptr.assign(indptr, indptr + n + 1);
        this->indices.assign(indices, indices + indptr[n]);
        this->values.assign(values, values + indptr[n]);
        ++add_calls;
    }

    int add_calls = 0;
    nsparse::idx_t num_added = 0;
    std::vector<nsparse::offset_t> indptr;
    std::vector<nsparse::term_t> indices;
    std::vector<float> values;
};

// Writes a CSR file in the layout read_csr expects, removed on destruction.
class TempCSRFile {
public:
    explicit TempCSRFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {}

    // Non-throwing overload: a destructor cannot forward an exception, and
    // Windows refuses to delete a file while a mapping over it is still open.
    ~TempCSRFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    TempCSRFile(const TempCSRFile&) = delete;
    TempCSRFile& operator=(const TempCSRFile&) = delete;

    void write(int64_t num_rows, int64_t num_cols, int64_t nnz,
               const std::vector<int64_t>& indptr,
               const std::vector<int32_t>& indices,
               const std::vector<float>& values) {
        std::ofstream file(path_, std::ios::binary);
        const std::array<int64_t, 3> header = {num_rows, num_cols, nnz};
        write_all(file, header.data(), header.size() * sizeof(int64_t));
        write_all(file, indptr.data(), indptr.size() * sizeof(int64_t));
        write_all(file, indices.data(), indices.size() * sizeof(int32_t));
        write_all(file, values.data(), values.size() * sizeof(float));
    }

    // Not path_.c_str(): path::value_type is wchar_t on Windows, so the
    // narrowed form has to be held somewhere with the same lifetime.
    const char* c_str() const { return path_str_.c_str(); }

private:
    static void write_all(std::ofstream& file, const void* data, size_t bytes) {
        file.write(static_cast<const char*>(data),
                   static_cast<std::streamsize>(bytes));
    }

    std::filesystem::path path_;
    std::string path_str_ = path_.string();
};

}  // namespace

TEST(IndexReadCSR, throws_on_null_path) {
    RecordingIndex index(10);
    ASSERT_THROW(index.read_csr(nullptr), std::invalid_argument);
}

TEST(IndexReadCSR, throws_on_missing_file) {
    RecordingIndex index(10);
    const auto missing =
        std::filesystem::temp_directory_path() / "nsparse_absent.csr";
    std::filesystem::remove(missing);

    ASSERT_THROW(index.read_csr(missing.string().c_str()),
                 std::invalid_argument);
}

TEST(IndexReadCSR, adds_vectors_from_file) {
    TempCSRFile file("nsparse_read_csr_ok.csr");
    // Rows: {0: 1.5, 2: 2.5}, {}, {1: 3.5}
    file.write(3, 4, 3, {0, 2, 2, 3}, {0, 2, 1}, {1.5F, 2.5F, 3.5F});

    RecordingIndex index(4);
    index.read_csr(file.c_str());

    ASSERT_EQ(index.add_calls, 1);
    ASSERT_EQ(index.num_added, 3);
    ASSERT_EQ(index.indptr, (std::vector<nsparse::offset_t>{0, 2, 2, 3}));
    ASSERT_EQ(index.indices, (std::vector<nsparse::term_t>{0, 2, 1}));
    ASSERT_EQ(index.values, (std::vector<float>{1.5F, 2.5F, 3.5F}));
}

// A short file is caught by the size check against the header, which runs before
// any array is read -- hence invalid_argument rather than the read's
// runtime_error.
TEST(IndexReadCSR, throws_on_truncated_indices) {
    TempCSRFile file("nsparse_read_csr_short_indices.csr");
    // Header claims 3 non-zeros but only 2 indices and no values follow.
    file.write(2, 4, 3, {0, 1, 3}, {0, 1}, {});

    RecordingIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str()), std::invalid_argument);
    ASSERT_EQ(index.add_calls, 0);
}

TEST(IndexReadCSR, throws_on_truncated_values) {
    TempCSRFile file("nsparse_read_csr_short_values.csr");
    // Indices are complete but one of the two values is missing.
    file.write(2, 4, 2, {0, 1, 2}, {0, 1}, {1.0F});

    RecordingIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str()), std::invalid_argument);
    ASSERT_EQ(index.add_calls, 0);
}

TEST(IndexReadCSR, throws_on_invalid_header) {
    TempCSRFile file("nsparse_read_csr_bad_header.csr");
    file.write(0, 4, 0, {0}, {}, {});

    RecordingIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str()), std::invalid_argument);
}

TEST(IndexReadCSR, throws_on_inconsistent_indptr) {
    TempCSRFile file("nsparse_read_csr_bad_indptr.csr");
    // Trailing indptr entry disagrees with the header's nnz.
    file.write(2, 4, 3, {0, 1, 2}, {0, 1, 2}, {1.0F, 2.0F, 3.0F});

    RecordingIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str()), std::invalid_argument);
    ASSERT_EQ(index.add_calls, 0);
}

TEST(IndexReadCSR, throws_when_columns_exceed_dimension) {
    TempCSRFile file("nsparse_read_csr_wide.csr");
    file.write(1, 8, 1, {0, 1}, {0}, {1.0F});

    RecordingIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str()), std::invalid_argument);
}

TEST(IndexReadCSR, throws_on_term_out_of_range) {
    TempCSRFile file("nsparse_read_csr_bad_term.csr");
    file.write(1, 4, 1, {0, 1}, {4}, {1.0F});

    RecordingIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str()), std::invalid_argument);
    ASSERT_EQ(index.add_calls, 0);
}
