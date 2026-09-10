/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/utils/csr_layout.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nsparse/types.h"

namespace {

using nsparse::idx_t;
using nsparse::offset_t;
using nsparse::term_t;
namespace layout = nsparse::csr_layout;

// Writes an interchange file from the arrays exactly as given: the header may
// disagree with them, which is what the negative tests need.
void write_interchange(const std::filesystem::path& path, int64_t num_rows,
                       int64_t num_cols, int64_t nnz,
                       const std::vector<int64_t>& indptr,
                       const std::vector<int32_t>& indices,
                       const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const std::array<int64_t, 3> header = {num_rows, num_cols, nnz};
    file.write(reinterpret_cast<const char*>(header.data()),
               sizeof(int64_t) * header.size());
    file.write(reinterpret_cast<const char*>(indptr.data()),
               static_cast<std::streamsize>(sizeof(int64_t) * indptr.size()));
    file.write(reinterpret_cast<const char*>(indices.data()),
               static_cast<std::streamsize>(sizeof(int32_t) * indices.size()));
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(sizeof(float) * values.size()));
}

struct NativeFile {
    std::array<int64_t, 3> header{};
    std::vector<offset_t> indptr;
    std::vector<term_t> indices;
    std::vector<float> values;
    std::vector<uint8_t> pad;
};

// Parses a native file the way MmapIndex does, so the assertions below fail if
// the writer and the offset helpers ever disagree.
NativeFile read_native(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    NativeFile result;
    file.read(reinterpret_cast<char*>(result.header.data()),
              sizeof(int64_t) * result.header.size());

    const auto indptr_size = static_cast<size_t>(result.header[0]) + 1;
    const auto nnz = static_cast<size_t>(result.header[2]);
    result.indptr.resize(indptr_size);
    result.indices.resize(nnz);
    result.values.resize(nnz);
    result.pad.resize(layout::padding(layout::kHeaderBytes +
                                      indptr_size * sizeof(offset_t) +
                                      nnz * sizeof(term_t)));

    file.read(reinterpret_cast<char*>(result.indptr.data()),
              static_cast<std::streamsize>(sizeof(offset_t) * indptr_size));
    file.read(reinterpret_cast<char*>(result.indices.data()),
              static_cast<std::streamsize>(sizeof(term_t) * nnz));
    file.read(reinterpret_cast<char*>(result.pad.data()),
              static_cast<std::streamsize>(result.pad.size()));
    EXPECT_EQ(file.tellg(), static_cast<std::streamoff>(
                                layout::native_values_offset(indptr_size, nnz)));
    file.read(reinterpret_cast<char*>(result.values.data()),
              static_cast<std::streamsize>(sizeof(float) * nnz));
    EXPECT_TRUE(file.good());
    return result;
}

// Each test gets its own directory, removed afterwards, so a failure cannot
// leave a stale file for the next run to pick up.
class CsrLayoutConvert : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               (std::string("nsparse_csr_layout_") +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);
    }

    void TearDown() override { std::filesystem::remove_all(dir_); }

    [[nodiscard]] std::filesystem::path in_path() const {
        return dir_ / "in.csr";
    }
    [[nodiscard]] std::filesystem::path out_path() const {
        return dir_ / "out.mcsr";
    }

    std::filesystem::path dir_;
};

}  // namespace

TEST(CsrLayoutPaths, native_path_appends_the_suffix) {
    ASSERT_EQ(layout::native_path("/tmp/base.csr"),
              std::string("/tmp/base.csr") + layout::kNativeSuffix);
}

TEST(CsrLayoutSizes, padding_aligns_values_to_float) {
    static_assert(layout::padding(0) == 0);
    static_assert(layout::padding(sizeof(term_t)) == 2);
    static_assert(layout::padding(sizeof(float)) == 0);
    static_assert(layout::padding(layout::kHeaderBytes + 2 * sizeof(idx_t) +
                                 sizeof(term_t)) == 2);

    // The native layout only ever needs padding for an odd nnz, since the
    // header and indptr are multiples of 4 bytes wide.
    static_assert(layout::native_values_offset(2, 1) % alignof(float) == 0);
    static_assert(layout::native_values_offset(2, 2) % alignof(float) == 0);
    static_assert(layout::native_file_size(2, 1) ==
                  layout::native_values_offset(2, 1) + sizeof(float));
    static_assert(layout::interchange_file_size(2, 1) ==
                  layout::kHeaderBytes + 2 * sizeof(int64_t) +
                      sizeof(int32_t) + sizeof(float));
}

TEST_F(CsrLayoutConvert, rewrites_arrays_at_native_widths) {
    // Rows: {0: 1.5, 2: 2.5}, {}, {1: 3.5}
    write_interchange(in_path(), 3, 4, 3, {0, 2, 2, 3}, {0, 2, 1},
                      {1.5F, 2.5F, 3.5F});

    layout::convert(in_path().string(), out_path().string());

    ASSERT_EQ(std::filesystem::file_size(out_path()),
              layout::native_file_size(4, 3));
    const auto native = read_native(out_path());
    ASSERT_EQ(native.header[0], 3);
    ASSERT_EQ(native.header[1], 4);
    ASSERT_EQ(native.header[2], 3);
    ASSERT_EQ(native.indptr, std::vector<offset_t>({0, 2, 2, 3}));
    ASSERT_EQ(native.indices, std::vector<term_t>({0, 2, 1}));
    ASSERT_EQ(native.values, std::vector<float>({1.5F, 2.5F, 3.5F}));
}

TEST_F(CsrLayoutConvert, pads_before_values_when_nnz_is_odd) {
    write_interchange(in_path(), 1, 4, 1, {0, 1}, {3}, {1.5F});

    layout::convert(in_path().string(), out_path().string());

    const auto native = read_native(out_path());
    ASSERT_EQ(native.pad.size(), 2);
    ASSERT_EQ(native.pad, std::vector<uint8_t>({0, 0}));
    ASSERT_EQ(native.values, std::vector<float>({1.5F}));
}

TEST_F(CsrLayoutConvert, writes_no_padding_when_nnz_is_even) {
    write_interchange(in_path(), 1, 4, 2, {0, 2}, {0, 3}, {1.5F, 2.5F});

    layout::convert(in_path().string(), out_path().string());

    ASSERT_TRUE(read_native(out_path()).pad.empty());
}

TEST_F(CsrLayoutConvert, handles_an_empty_matrix) {
    write_interchange(in_path(), 2, 4, 0, {0, 0, 0}, {}, {});

    layout::convert(in_path().string(), out_path().string());

    ASSERT_EQ(std::filesystem::file_size(out_path()),
              layout::native_file_size(3, 0));
    const auto native = read_native(out_path());
    ASSERT_EQ(native.indptr, std::vector<offset_t>({0, 0, 0}));
    ASSERT_TRUE(native.indices.empty());
}

TEST_F(CsrLayoutConvert, streams_across_the_chunk_boundary) {
    // One row wider than the 64K element conversion chunk, so the last chunk is
    // a partial one and the values array starts at an odd nnz.
    constexpr int64_t kNnz = (1 << 16) + 1;
    constexpr int64_t kCols = 1 << 16;
    std::vector<int32_t> indices(kNnz);
    std::vector<float> values(kNnz);
    for (int64_t i = 0; i < kNnz; ++i) {
        indices[i] = static_cast<int32_t>(i % kCols);
        values[i] = static_cast<float>(i);
    }
    write_interchange(in_path(), 1, kCols, kNnz, {0, kNnz}, indices, values);

    layout::convert(in_path().string(), out_path().string());

    ASSERT_EQ(std::filesystem::file_size(out_path()),
              layout::native_file_size(2, kNnz));
    const auto native = read_native(out_path());
    ASSERT_EQ(native.indices.size(), kNnz);
    ASSERT_EQ(native.indices.front(), 0);
    ASSERT_EQ(native.indices[kCols - 1], kCols - 1);
    ASSERT_EQ(native.indices.back(), 0);  // wrapped by i % kCols
    ASSERT_FLOAT_EQ(native.values[kCols - 1], static_cast<float>(kCols - 1));
    ASSERT_FLOAT_EQ(native.values.back(), static_cast<float>(kNnz - 1));
}

TEST_F(CsrLayoutConvert, throws_on_missing_source) {
    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
    ASSERT_FALSE(std::filesystem::exists(out_path()));
}

TEST_F(CsrLayoutConvert, throws_on_truncated_header) {
    std::ofstream file(in_path(), std::ios::binary | std::ios::trunc);
    const int64_t partial = 1;
    file.write(reinterpret_cast<const char*>(&partial), sizeof(partial));
    file.close();

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::runtime_error);
}

TEST_F(CsrLayoutConvert, throws_on_non_positive_rows) {
    write_interchange(in_path(), 0, 4, 0, {0}, {}, {});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_on_non_positive_cols) {
    write_interchange(in_path(), 1, 0, 0, {0, 0}, {}, {});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_on_negative_nnz) {
    write_interchange(in_path(), 1, 4, -1, {0, 0}, {}, {});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_when_the_file_is_shorter_than_the_header_says) {
    // Header claims 3 non-zeros but only 2 indices and 2 values follow.
    write_interchange(in_path(), 1, 4, 3, {0, 3}, {0, 1}, {1.0F, 2.0F});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_on_trailing_bytes) {
    write_interchange(in_path(), 1, 4, 1, {0, 1}, {0}, {1.0F});
    std::ofstream extra(in_path(), std::ios::binary | std::ios::app);
    const char junk = 0;
    extra.write(&junk, 1);
    extra.close();

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_when_handed_a_native_file) {
    // A native file of the same shape is smaller, so the size check catches the
    // mix-up that no header field would reveal.
    write_interchange(in_path(), 1, 4, 2, {0, 2}, {0, 1}, {1.0F, 2.0F});
    layout::convert(in_path().string(), out_path().string());

    ASSERT_THROW(layout::convert(out_path().string(),
                                 layout::native_path(out_path().string())),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_when_indptr_does_not_start_at_zero) {
    write_interchange(in_path(), 1, 4, 1, {1, 1}, {0}, {1.0F});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_when_indptr_disagrees_with_nnz) {
    write_interchange(in_path(), 2, 4, 3, {0, 1, 2}, {0, 1, 2},
                      {1.0F, 2.0F, 3.0F});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_on_negative_indptr) {
    write_interchange(in_path(), 2, 4, 2, {0, -1, 2}, {0, 1}, {1.0F, 2.0F});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_on_negative_term) {
    write_interchange(in_path(), 1, 4, 1, {0, 1}, {-1}, {1.0F});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_on_term_beyond_the_column_count) {
    write_interchange(in_path(), 1, 4, 1, {0, 1}, {4}, {1.0F});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_on_term_wider_than_term_t) {
    // In range for the declared column count, but not for the native width.
    constexpr int64_t kCols = 1 << 17;
    write_interchange(in_path(), 1, kCols, 1, {0, 1}, {1 << 16}, {1.0F});

    ASSERT_THROW(layout::convert(in_path().string(), out_path().string()),
                 std::invalid_argument);
}

TEST_F(CsrLayoutConvert, throws_when_the_destination_cannot_be_opened) {
    write_interchange(in_path(), 1, 4, 1, {0, 1}, {0}, {1.0F});

    ASSERT_THROW(layout::convert(in_path().string(),
                                 (dir_ / "absent" / "out.mcsr").string()),
                 std::runtime_error);
}
