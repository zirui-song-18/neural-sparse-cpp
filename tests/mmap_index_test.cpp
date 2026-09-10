/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/mmap_index.h"

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

// Records add() calls, which only the non-mmap fallback makes.
class TestMmapIndex : public nsparse::MmapIndex {
public:
    explicit TestMmapIndex(int dim) : MmapIndex(dim) {}

    std::array<char, 4> id() const override { return {'M', 'M', 'A', 'P'}; }

    void add(nsparse::idx_t n, const nsparse::offset_t* indptr,
             const nsparse::term_t* indices, const float* values) override {
        num_added = n;
        ++add_calls;
    }

    int add_calls = 0;
    nsparse::idx_t num_added = 0;
};

// Writes a CSR file in the native layout read_csr_ expects, removed on
// destruction.
class TempNativeCSRFile {
public:
    explicit TempNativeCSRFile(const std::string& name)
        : path_(std::filesystem::temp_directory_path() / name) {}

    // Non-throwing overload: a destructor cannot forward an exception, and
    // Windows refuses to delete a file while a mapping over it is still open.
    ~TempNativeCSRFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    TempNativeCSRFile(const TempNativeCSRFile&) = delete;
    TempNativeCSRFile& operator=(const TempNativeCSRFile&) = delete;

    void write(int64_t num_rows, int64_t num_cols, int64_t nnz,
               const std::vector<nsparse::offset_t>& indptr,
               const std::vector<nsparse::term_t>& indices,
               const std::vector<float>& values) {
        std::ofstream file(path_, std::ios::binary);
        const std::array<int64_t, 3> header = {num_rows, num_cols, nnz};
        write_all(file, header.data(), header.size() * sizeof(int64_t));
        write_all(file, indptr.data(),
                  indptr.size() * sizeof(nsparse::offset_t));
        write_all(file, indices.data(),
                  indices.size() * sizeof(nsparse::term_t));
        const size_t unaligned = header.size() * sizeof(int64_t) +
                                 indptr.size() * sizeof(nsparse::offset_t) +
                                 indices.size() * sizeof(nsparse::term_t);
        const std::array<uint8_t, sizeof(float)> padding{};
        write_all(file, padding.data(), unaligned % alignof(float) == 0
                                            ? 0
                                            : alignof(float) -
                                                  unaligned % alignof(float));
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

// The widened layout Index::read_csr accepts: int64 indptr, int32 indices.
void write_widened_csr(const std::filesystem::path& path, int64_t num_rows,
                       int64_t num_cols, int64_t nnz,
                       const std::vector<int64_t>& indptr,
                       const std::vector<int32_t>& indices,
                       const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    const std::array<int64_t, 3> header = {num_rows, num_cols, nnz};
    file.write(reinterpret_cast<const char*>(header.data()),
               header.size() * sizeof(int64_t));
    file.write(reinterpret_cast<const char*>(indptr.data()),
               static_cast<std::streamsize>(indptr.size() * sizeof(int64_t)));
    file.write(reinterpret_cast<const char*>(indices.data()),
               static_cast<std::streamsize>(indices.size() * sizeof(int32_t)));
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
}

}  // namespace

TEST(MmapIndexReadCSR, maps_vectors_without_calling_add) {
    TempNativeCSRFile file("nsparse_mmap_ok.csr");
    // Rows: {0: 1.5, 2: 2.5}, {}, {1: 3.5}
    file.write(3, 4, 3, {0, 2, 2, 3}, {0, 2, 1}, {1.5F, 2.5F, 3.5F});

    TestMmapIndex index(4);
    index.read_csr(file.c_str(), nsparse::Residency::kMmap);

    ASSERT_EQ(index.add_calls, 0);
    const auto* vectors = index.get_vectors();
    ASSERT_NE(vectors, nullptr);
    ASSERT_EQ(vectors->num_vectors(), 3);
    ASSERT_EQ(vectors->get_dimension(), 4);
    ASSERT_EQ(vectors->get_element_size(), nsparse::U32);

    auto dense = vectors->get_dense_vector_float(0);
    ASSERT_EQ(dense.size(), 4);
    ASSERT_FLOAT_EQ(dense[0], 1.5F);
    ASSERT_FLOAT_EQ(dense[1], 0.0F);
    ASSERT_FLOAT_EQ(dense[2], 2.5F);
    ASSERT_FLOAT_EQ(dense[3], 0.0F);

    ASSERT_FLOAT_EQ(vectors->get_dense_vector_float(2)[1], 3.5F);
}

TEST(MmapIndexReadCSR, values_point_into_the_mapping) {
    TempNativeCSRFile file("nsparse_mmap_zero_copy.csr");
    file.write(1, 2, 2, {0, 2}, {0, 1}, {1.5F, 2.5F});

    TestMmapIndex index(2);
    index.read_csr(file.c_str(), nsparse::Residency::kMmap);

    // A mapping keeps the arrays contiguous in file order; a copy would not.
    const auto* vectors = index.get_vectors();
    const auto* indptr_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indptr_data());
    const auto* indices_bytes =
        reinterpret_cast<const uint8_t*>(vectors->indices_data());
    ASSERT_EQ(indices_bytes - indptr_bytes, 2 * sizeof(nsparse::offset_t));
}

TEST(MmapIndexReadCSR, falls_back_to_index_read_csr_when_disabled) {
    const auto path =
        std::filesystem::temp_directory_path() / "nsparse_mmap_widened.csr";
    write_widened_csr(path, 2, 4, 2, {0, 1, 2}, {0, 1}, {1.0F, 2.0F});

    TestMmapIndex index(4);
    index.read_csr(path.string().c_str(), nsparse::Residency::kInMemory);
    std::filesystem::remove(path);

    ASSERT_EQ(index.add_calls, 1);
    ASSERT_EQ(index.num_added, 2);
}

TEST(MmapIndexReadCSR, throws_on_null_path) {
    TestMmapIndex index(4);
    ASSERT_THROW(index.read_csr(nullptr, nsparse::Residency::kMmap), std::invalid_argument);
}

TEST(MmapIndexReadCSR, throws_on_missing_file) {
    TestMmapIndex index(4);
    const auto missing =
        std::filesystem::temp_directory_path() / "nsparse_mmap_absent.csr";
    std::filesystem::remove(missing);

    ASSERT_THROW(index.read_csr(missing.string().c_str(),
                                nsparse::Residency::kMmap),
                 std::invalid_argument);
}

TEST(MmapIndexReadCSR, throws_on_widened_layout) {
    const auto path =
        std::filesystem::temp_directory_path() / "nsparse_mmap_reject.csr";
    write_widened_csr(path, 2, 4, 2, {0, 1, 2}, {0, 1}, {1.0F, 2.0F});

    TestMmapIndex index(4);
    ASSERT_THROW(index.read_csr(path.string().c_str(),
                                nsparse::Residency::kMmap),
                 std::invalid_argument);
    std::filesystem::remove(path);
}

TEST(MmapIndexReadCSR, throws_on_truncated_file) {
    TempNativeCSRFile file("nsparse_mmap_short.csr");
    // Header claims 3 non-zeros but only 2 indices and 2 values follow.
    file.write(2, 4, 3, {0, 1, 3}, {0, 1}, {1.0F, 2.0F});

    TestMmapIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str(), nsparse::Residency::kMmap), std::invalid_argument);
}

TEST(MmapIndexReadCSR, throws_on_invalid_header) {
    TempNativeCSRFile file("nsparse_mmap_bad_header.csr");
    file.write(0, 4, 0, {0}, {}, {});

    TestMmapIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str(), nsparse::Residency::kMmap), std::invalid_argument);
}

TEST(MmapIndexReadCSR, throws_when_columns_exceed_dimension) {
    TempNativeCSRFile file("nsparse_mmap_wide.csr");
    file.write(1, 8, 1, {0, 1}, {0}, {1.0F});

    TestMmapIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str(), nsparse::Residency::kMmap), std::invalid_argument);
}

TEST(MmapIndexReadCSR, throws_on_inconsistent_indptr) {
    TempNativeCSRFile file("nsparse_mmap_bad_indptr.csr");
    // Trailing indptr entry disagrees with the header's nnz.
    file.write(2, 4, 3, {0, 1, 2}, {0, 1, 2}, {1.0F, 2.0F, 3.0F});

    TestMmapIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str(), nsparse::Residency::kMmap), std::invalid_argument);
}

TEST(MmapIndexReadCSR, throws_on_decreasing_indptr) {
    TempNativeCSRFile file("nsparse_mmap_decreasing_indptr.csr");
    // Row 1 ends before it starts, so its length underflows.
    file.write(3, 4, 2, {0, 2, 1, 2}, {0, 1}, {1.0F, 2.0F});

    TestMmapIndex index(4);
    ASSERT_THROW(index.read_csr(file.c_str(), nsparse::Residency::kMmap), std::invalid_argument);
}

// Terms are NOT validated on this path: scanning all nnz would fault in the
// whole indices array at open, defeating the mapping. csr_layout::convert is
// where an out-of-range term is rejected, so a native file is trusted to have
// come from there.
TEST(MmapIndexReadCSR, does_not_scan_terms) {
    TempNativeCSRFile file("nsparse_mmap_bad_term.csr");
    file.write(1, 4, 1, {0, 1}, {4}, {1.0F});

    TestMmapIndex index(4);
    ASSERT_NO_THROW(index.read_csr(file.c_str(), nsparse::Residency::kMmap));
}

TEST(MmapIndexReadCSR, throws_on_second_load) {
    TempNativeCSRFile file("nsparse_mmap_reload.csr");
    file.write(1, 4, 1, {0, 1}, {0}, {1.0F});

    TestMmapIndex index(4);
    index.read_csr(file.c_str(), nsparse::Residency::kMmap);
    ASSERT_THROW(index.read_csr(file.c_str(), nsparse::Residency::kMmap), std::runtime_error);
}
