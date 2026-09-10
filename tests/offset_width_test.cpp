/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "nsparse/index.h"
#include "nsparse/index_factory.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/sparse_vectors.h"
#include "nsparse/types.h"
#include "nsparse/utils/csr_layout.h"
#include "nsparse/utils/mmap_cursor.h"

namespace nsparse {
namespace {

// offset_t is the widened CSR nnz-offset type: a signed 64-bit integer, distinct
// from the 32-bit idx_t used for doc-ids/labels.
static_assert(std::is_same_v<offset_t, int64_t>);
static_assert(sizeof(offset_t) == 8);
static_assert(std::is_signed_v<offset_t>);
static_assert(sizeof(idx_t) == 4);

// The storage root hands out 64-bit offsets, not 32-bit doc indices.
static_assert(std::is_same_v<decltype(std::declval<const SparseVectors>()
                                          .indptr_data()),
                             const offset_t*>);

int64_t aligned_values_offset(int64_t indptr_size, int64_t nnz) {
    const int64_t unaligned = 3 * 8 + indptr_size * 8 + nnz * 2;
    return unaligned + (4 - unaligned % 4) % 4;
}

// The native .mcsr layout stores indptr at 8 bytes/entry (offset_t), so the
// values section sits 4 bytes further out per entry than the old int32 layout.
TEST(OffsetWidth, NativeLayoutUsesEightByteIndptr) {
    EXPECT_EQ(csr_layout::native_values_offset(3, 0), 24 + 24);
    EXPECT_EQ(csr_layout::native_values_offset(6, 7),
              aligned_values_offset(6, 7));
    EXPECT_EQ(csr_layout::native_values_offset(101, 50),
              aligned_values_offset(101, 50));
    EXPECT_EQ(csr_layout::native_file_size(6, 7, sizeof(float)),
              csr_layout::native_values_offset(6, 7) + 7 * sizeof(float));
}

// Round-trips a small corpus through both serialize/deserialize and
// serialize/mmap_deserialize, asserting indptr comes back identical as 64-bit
// words -- exercising the widened read_padded<offset_t> / read_array<offset_t>
// format path. The terminal offset fits int32 here; the >2^31 gate is the
// DISABLED_ test below.
TEST(OffsetWidth, IndptrRoundTripsAsSixtyFourBit) {
    SparseVectors original({.element_size = U32, .dimension = 8});
    const std::vector<offset_t> indptr = {0, 2, 3, 6};
    const std::vector<term_t> indices = {0, 1, 2, 3, 4, 5};
    const std::vector<uint8_t> values(6 * U32, 0);
    original.add_vectors(indptr, indices, values);

    BufferedIOWriter writer;
    original.serialize(&writer);
    const std::vector<uint8_t> bytes = writer.data();

    BufferedIOReader reader(bytes);
    SparseVectors deserialized;
    deserialized.deserialize(&reader);

    MmapCursor cursor(bytes.data(), bytes.size());
    SparseVectors mapped;
    mapped.mmap_deserialize(&cursor);

    for (const SparseVectors* v : {&original, &deserialized, &mapped}) {
        ASSERT_EQ(v->num_vectors(), 3U);
        for (size_t i = 0; i < indptr.size(); ++i) {
            EXPECT_EQ(v->indptr_data()[i], indptr[i]);
        }
    }
}

// The real acceptance gate: build a corpus whose cumulative nnz exceeds
// INT32_MAX and search it end to end, so the widened offset_t flows through
// add -> build (clustering, inverted lists, forward index) -> search instead
// of wrapping negative past the ~2.1-billionth nnz. Disabled by default: the
// corpus needs tens of GB. Run on a large host with
// --gtest_also_run_disabled_tests.
TEST(OffsetWidth, DISABLED_CrossesInt32BoundaryEndToEnd) {
    const idx_t n_docs = 36'000;
    const int dim = 60'000;  // < term_t max; every doc carries all `dim` terms
    const int64_t per_doc = dim;
    const int64_t total_nnz = static_cast<int64_t>(n_docs) * per_doc;
    ASSERT_GT(total_nnz,
              static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

    std::vector<offset_t> indptr(n_docs + 1);
    for (idx_t d = 0; d <= n_docs; ++d) indptr[d] = static_cast<offset_t>(d) *
                                                    per_doc;
    std::vector<term_t> indices(static_cast<size_t>(total_nnz));
    for (idx_t d = 0; d < n_docs; ++d) {
        for (int t = 0; t < dim; ++t) {
            indices[static_cast<size_t>(indptr[d]) + t] =
                static_cast<term_t>(t);
        }
    }
    std::vector<float> values(static_cast<size_t>(total_nnz), 1.0F);
    ASSERT_GT(indptr[n_docs],
              static_cast<offset_t>(std::numeric_limits<int32_t>::max()));

    std::unique_ptr<Index> index(index_factory(dim, "seismic"));
    index->add(n_docs, indptr.data(), indices.data(), values.data());
    index->build();
    ASSERT_EQ(index->num_vectors(), static_cast<size_t>(n_docs));

    std::vector<term_t> q_indices(dim);
    for (int t = 0; t < dim; ++t) q_indices[t] = static_cast<term_t>(t);
    const std::vector<float> q_values(dim, 1.0F);
    const offset_t q_indptr[2] = {0, dim};
    const int k = 10;
    std::vector<float> distances(k);
    std::vector<idx_t> labels(k);
    index->search(1, q_indptr, q_indices.data(), q_values.data(), k,
                  distances.data(), labels.data());

    EXPECT_GE(labels[0], 0);
    EXPECT_LT(labels[0], n_docs);
    EXPECT_GT(distances[0], 0.0F);
}

}  // namespace
}  // namespace nsparse
