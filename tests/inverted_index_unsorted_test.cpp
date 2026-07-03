/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

// Regression test that ACTUALLY reproduces the PR #15 SIGBUS.
//
// Unlike search_multi_window_no_crash (which passes even without the fix
// because the add()->build() path always yields sorted, non-negative
// doc_ids), this test exercises the DESERIALIZATION path (read_index), which
// is how production/JNI populates the index. It hand-builds an INVT byte
// stream whose posting list is UNSORTED (and a second case: NEGATIVE), then
// loads it via read_index and searches.
//
// Expected behavior:
//   - On the buggy code (no bounds guard, no entry-point validation): the
//     search writes to window_scores[negative slot] -> heap-buffer-overflow /
//     SIGBUS. This test CRASHES (i.e. reproduces the bug).
//   - With the proper entry-point fix (sort + reject-negative in read_index):
//     the unsorted list is corrected and search returns CORRECT results; the
//     negative-doc_id list is rejected with std::runtime_error at load time.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nsparse/inverted_index.h"
#include "nsparse/io/buffered_io.h"
#include "nsparse/io/index_io.h"
#include "nsparse/types.h"

namespace nsparse {
namespace {

template <class T>
void put(std::vector<uint8_t>& buf, const T& v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
}
void put_bytes(std::vector<uint8_t>& buf, const void* p, size_t n) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(p);
    buf.insert(buf.end(), b, b + n);
}

// Serialize an INVT index by hand, matching write_header + write_index, using
// the posting order given verbatim (so we can inject unsorted / negative ids).
std::vector<uint8_t> make_index_bytes(
    int dimension,
    const std::vector<std::vector<std::pair<idx_t, float>>>& lists) {
    std::vector<uint8_t> buf;
    uint32_t id_val = fourcc(std::array<char, 4>{'I', 'N', 'V', 'T'});
    put(buf, id_val);
    put(buf, dimension);

    size_t n_terms = lists.size();
    put(buf, n_terms);
    size_t element_size = 4;  // U32
    put(buf, element_size);

    std::vector<float> max_term_scores(n_terms, 0.0F);
    for (size_t t = 0; t < n_terms; ++t) {
        size_t list_size = lists[t].size();
        put(buf, list_size);
        if (list_size > 0) {
            for (const auto& [doc, val] : lists[t]) put(buf, doc);
            for (const auto& [doc, val] : lists[t]) {
                float v = val;
                put_bytes(buf, &v, sizeof(float));
                if (v > max_term_scores[t]) max_term_scores[t] = v;
            }
        }
    }
    size_t scores_size = max_term_scores.size();
    put(buf, scores_size);
    if (scores_size > 0)
        put_bytes(buf, max_term_scores.data(), scores_size * sizeof(float));
    return buf;
}

// Loads an index from a deserialized byte stream with an UNSORTED posting list
// and searches it. This is the true PR #15 reproducer.
//
// On buggy code this SIGBUSes (negative slot write). With the entry-point fix
// the list is sorted at load time and results are correct.
TEST(InvertedIndexUnsorted, deserialized_unsorted_list_search) {
    // term0 postings stored UNSORTED: doc 5000 before doc 3.
    // min_doc_id=5000 -> window_base=4096; doc 3 then yields slot = 3-4096.
    std::vector<std::vector<std::pair<idx_t, float>>> lists(1);
    lists[0] = {{5000, 0.9F}, {3, 0.8F}};
    auto bytes = make_index_bytes(/*dim=*/1, lists);

    BufferedIOReader reader(bytes);
    Index* idx = read_index(&reader);
    ASSERT_NE(idx, nullptr);

    std::vector<idx_t> q_indptr = {0, 1};
    std::vector<term_t> q_indices = {0};
    std::vector<float> q_values = {1.0F};
    std::vector<idx_t> labels(10, -1);
    std::vector<float> distances(10, -1.0F);

    // Buggy code crashes here. Fixed code returns correct top result.
    idx->search(1, q_indptr.data(), q_indices.data(), q_values.data(), 10,
                distances.data(), labels.data());

    // With the fix, doc 5000 (value 0.9) is the top result and doc 3 (0.8)
    // second -- and crucially, doc 3 is NOT dropped (which the inner-loop
    // guard would do).
    EXPECT_EQ(labels[0], 5000);
    EXPECT_FLOAT_EQ(distances[0], 0.9F);
    EXPECT_EQ(labels[1], 3);
    EXPECT_FLOAT_EQ(distances[1], 0.8F);

    delete idx;
}

// Companion: an unsorted NON-ESSENTIAL list corrupts scores via advance_to's
// std::lower_bound even WITH the inner-loop guard. The entry-point fix cures
// it. This documents that the guard is insufficient for correctness.
TEST(InvertedIndexUnsorted, deserialized_unsorted_nonessential_recall) {
    // term1 (essential, high value) selects windows; term0 (non-essential,
    // low value) must contribute to doc 100 via advance_to.
    std::vector<std::vector<std::pair<idx_t, float>>> lists(2);
    // term0 stored UNSORTED with high doc first -> advance_to(100) early-outs.
    lists[0] = {{9999, 0.5F}, {100, 0.5F}};
    lists[1] = {{1, 1.02F}, {2, 1.01F}, {3, 1.00F}, {100, 0.9F}};
    auto bytes = make_index_bytes(/*dim=*/2, lists);

    BufferedIOReader reader(bytes);
    Index* idx = read_index(&reader);
    ASSERT_NE(idx, nullptr);

    std::vector<idx_t> q_indptr = {0, 2};
    std::vector<term_t> q_indices = {0, 1};
    std::vector<float> q_values = {1.0F, 1.0F};
    std::vector<idx_t> labels(3, -1);
    std::vector<float> distances(3, -1.0F);

    idx->search(1, q_indptr.data(), q_indices.data(), q_values.data(), 3,
                distances.data(), labels.data());

    // Correct: doc 100 = term0(0.5) + term1(0.9) = 1.40 is the top result.
    EXPECT_EQ(labels[0], 100);
    EXPECT_FLOAT_EQ(distances[0], 1.40F);

    delete idx;
}

// A negative doc_id in a deserialized posting list is a corruption signal
// (idx_t is int32; a mis-mapped external id or overflow can arrive negative).
// read_index must reject it at load time with std::runtime_error rather than
// letting slot = doc - window_base go negative and corrupt memory at query
// time. This fails fast at the deserialization boundary.
TEST(InvertedIndexUnsorted, deserialized_negative_doc_id_throws) {
    std::vector<std::vector<std::pair<idx_t, float>>> lists(1);
    lists[0] = {{-100, 0.9F}};  // negative doc_id
    auto bytes = make_index_bytes(/*dim=*/1, lists);

    BufferedIOReader reader(bytes);
    // read_index calls sanitize_posting_list, which throws on negative ids.
    EXPECT_THROW(
        {
            Index* idx = read_index(&reader);
            delete idx;
        },
        std::runtime_error);
}

}  // namespace
}  // namespace nsparse
