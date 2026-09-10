/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#include "nsparse/sparse_vectors.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nsparse/io/buffered_io.h"
#include "nsparse/types.h"
#include "nsparse/utils/mmap_cursor.h"

namespace {

// Which of the two ways a SparseVectors can hold its arrays. Everything a
// reader can observe should be identical either way, so the tests below run
// once per residency.
enum class Residency : uint8_t { kOwned, kMapped };

class SparseVectorsResidency : public ::testing::TestWithParam<Residency> {
protected:
    // Builds vectors over one CSR, copied in via add_vectors or borrowed via
    // map_vectors. element_size follows sizeof(Value).
    //
    // Mapped vectors borrow, so the arrays live in storage_ and outlive the
    // result rather than being locals of the caller.
    template <class Value>
    nsparse::SparseVectors make(size_t dimension,
                                std::vector<nsparse::offset_t> indptr,
                                std::vector<nsparse::term_t> indices,
                                const std::vector<Value>& values) {
        const nsparse::SparseVectorsConfig config = {
            .element_size = sizeof(Value), .dimension = dimension};
        const Storage& held =
            store(std::move(indptr), std::move(indices), values);

        if (GetParam() == Residency::kOwned) {
            nsparse::SparseVectors vectors(config);
            vectors.add_vectors(held.indptr.data(), held.indptr.size(),
                                held.indices.data(), held.indices.size(),
                                held.values(), held.value_bytes);
            return vectors;
        }
        return nsparse::SparseVectors::map_vectors(
            config, held.indptr.data(), held.indptr.size(),
            held.indices.data(), held.indices.size(), held.values(),
            held.value_bytes);
    }

private:
    struct Storage {
        std::vector<nsparse::offset_t> indptr;
        std::vector<nsparse::term_t> indices;
        // Words, not bytes: map_vectors rejects a values pointer misaligned for
        // element_size, and an allocation is only guaranteed aligned for its own
        // element type. uint64_t covers every element size.
        std::vector<uint64_t> value_words;
        size_t value_bytes = 0;

        [[nodiscard]] const uint8_t* values() const {
            return reinterpret_cast<const uint8_t*>(value_words.data());
        }
    };

    template <class Value>
    const Storage& store(std::vector<nsparse::offset_t> indptr,
                         std::vector<nsparse::term_t> indices,
                         const std::vector<Value>& values) {
        // A deque keeps earlier elements put, so a second make() in the same
        // test does not dangle the first one's arrays.
        Storage& held = storage_.emplace_back();
        held.indptr = std::move(indptr);
        held.indices = std::move(indices);
        held.value_bytes = values.size() * sizeof(Value);
        held.value_words.resize(
            (held.value_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
        if (held.value_bytes > 0) {
            std::memcpy(held.value_words.data(), values.data(),
                        held.value_bytes);
        }
        return held;
    }

    std::deque<Storage> storage_;
};

INSTANTIATE_TEST_SUITE_P(
    Residencies, SparseVectorsResidency,
    ::testing::Values(Residency::kOwned, Residency::kMapped),
    [](const ::testing::TestParamInfo<Residency>& info) {
        return info.param == Residency::kOwned ? "owned" : "mapped";
    });

// Byte buffer whose data() is deliberately offset, for the alignment check.
std::vector<uint8_t> misaligned_storage(size_t bytes) {
    return std::vector<uint8_t>(bytes + 1, 0);
}

}  // namespace

// ---------------------------------------------------------------------------
// Residency-independent behaviour: run once owned, once mapped.
// ---------------------------------------------------------------------------

TEST_P(SparseVectorsResidency, reports_the_vector_count) {
    // Rows: {0, 1}, {}, {2, 3, 4}
    auto vectors = make<float>(10, {0, 2, 2, 5}, {0, 1, 2, 3, 4},
                               {1.0F, 2.0F, 3.0F, 4.0F, 5.0F});

    ASSERT_EQ(vectors.num_vectors(), 3);
}

TEST_P(SparseVectorsResidency, reports_the_config) {
    auto vectors = make<uint16_t>(7, {0, 2}, {0, 1}, {1000, 2000});

    ASSERT_EQ(vectors.get_dimension(), 7);
    ASSERT_EQ(vectors.get_element_size(), nsparse::U16);
}

TEST_P(SparseVectorsResidency, get_dense_vector_float_single) {
    auto vectors = make<float>(5, {0, 3}, {0, 2, 4}, {1.0F, 2.0F, 3.0F});

    auto dense = vectors.get_dense_vector_float(0);

    ASSERT_EQ(dense.size(), 5);
    ASSERT_FLOAT_EQ(dense[0], 1.0F);
    ASSERT_FLOAT_EQ(dense[1], 0.0F);
    ASSERT_FLOAT_EQ(dense[2], 2.0F);
    ASSERT_FLOAT_EQ(dense[3], 0.0F);
    ASSERT_FLOAT_EQ(dense[4], 3.0F);
}

TEST_P(SparseVectorsResidency, get_dense_vector_float_uint8_element) {
    auto vectors = make<uint8_t>(3, {0, 2}, {0, 2}, {100, 200});

    auto dense = vectors.get_dense_vector_float(0);

    ASSERT_EQ(dense.size(), 3);
    ASSERT_FLOAT_EQ(dense[0], 100.0F);
    ASSERT_FLOAT_EQ(dense[1], 0.0F);
    ASSERT_FLOAT_EQ(dense[2], 200.0F);
}

TEST_P(SparseVectorsResidency, get_dense_vector_float_uint16_element) {
    auto vectors = make<uint16_t>(3, {0, 2}, {0, 2}, {1000, 2000});

    auto dense = vectors.get_dense_vector_float(0);

    ASSERT_EQ(dense.size(), 3);
    ASSERT_FLOAT_EQ(dense[0], 1000.0F);
    ASSERT_FLOAT_EQ(dense[1], 0.0F);
    ASSERT_FLOAT_EQ(dense[2], 2000.0F);
}

TEST_P(SparseVectorsResidency, get_dense_vector_float_multiple_vectors) {
    auto vectors = make<float>(3, {0, 1, 3}, {0, 1, 2}, {1.0F, 2.0F, 3.0F});

    auto dense0 = vectors.get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense0[0], 1.0F);
    ASSERT_FLOAT_EQ(dense0[1], 0.0F);
    ASSERT_FLOAT_EQ(dense0[2], 0.0F);

    auto dense1 = vectors.get_dense_vector_float(1);
    ASSERT_FLOAT_EQ(dense1[0], 0.0F);
    ASSERT_FLOAT_EQ(dense1[1], 2.0F);
    ASSERT_FLOAT_EQ(dense1[2], 3.0F);
}

TEST_P(SparseVectorsResidency, get_dense_vector_float_empty_row) {
    auto vectors = make<float>(3, {0, 1, 1}, {0}, {1.0F});

    auto dense = vectors.get_dense_vector_float(1);

    ASSERT_EQ(dense.size(), 3);
    ASSERT_FLOAT_EQ(dense[0], 0.0F);
    ASSERT_FLOAT_EQ(dense[1], 0.0F);
    ASSERT_FLOAT_EQ(dense[2], 0.0F);
}

TEST_P(SparseVectorsResidency, get_dense_vector_float_out_of_range) {
    auto vectors = make<float>(5, {0, 1}, {0}, {1.0F});

    ASSERT_THROW(vectors.get_dense_vector_float(1), std::out_of_range);
    ASSERT_THROW(vectors.get_dense_vector_float(-1), std::out_of_range);
}

TEST_P(SparseVectorsResidency, get_dense_vector_uint8) {
    auto vectors = make<uint8_t>(4, {0, 2}, {1, 3}, {50, 150});

    auto dense = vectors.get_dense_vector(0);

    ASSERT_EQ(dense.size(), 4);
    ASSERT_EQ(dense[0], 0);
    ASSERT_EQ(dense[1], 50);
    ASSERT_EQ(dense[2], 0);
    ASSERT_EQ(dense[3], 150);
}

TEST_P(SparseVectorsResidency, get_dense_vector_uint16) {
    auto vectors = make<uint16_t>(3, {0, 2}, {0, 2}, {1000, 2000});

    auto dense = vectors.get_dense_vector(0);

    ASSERT_EQ(dense.size(), 3 * sizeof(uint16_t));
    const auto* typed = reinterpret_cast<const uint16_t*>(dense.data());
    ASSERT_EQ(typed[0], 1000);
    ASSERT_EQ(typed[1], 0);
    ASSERT_EQ(typed[2], 2000);
}

TEST_P(SparseVectorsResidency, get_dense_vector_out_of_range) {
    auto vectors = make<uint8_t>(5, {0, 1}, {0}, {1});

    ASSERT_THROW(vectors.get_dense_vector(1), std::out_of_range);
}

TEST_P(SparseVectorsResidency, indptr_data) {
    auto vectors = make<float>(5, {0, 2, 3}, {0, 1, 4}, {1.0F, 2.0F, 3.0F});

    const nsparse::offset_t* indptr = vectors.indptr_data();
    ASSERT_EQ(indptr[0], 0);
    ASSERT_EQ(indptr[1], 2);
    ASSERT_EQ(indptr[2], 3);
}

TEST_P(SparseVectorsResidency, indices_data) {
    auto vectors = make<float>(5, {0, 2}, {2, 4}, {1.0F, 2.0F});

    const nsparse::term_t* indices = vectors.indices_data();
    ASSERT_EQ(indices[0], 2);
    ASSERT_EQ(indices[1], 4);
}

TEST_P(SparseVectorsResidency, values_data_float) {
    auto vectors = make<float>(5, {0, 2}, {0, 1}, {1.5F, 2.5F});

    const float* values = vectors.values_data_float();
    ASSERT_FLOAT_EQ(values[0], 1.5F);
    ASSERT_FLOAT_EQ(values[1], 2.5F);
}

TEST_P(SparseVectorsResidency, typed_values_data) {
    auto vectors = make<uint16_t>(5, {0, 2}, {0, 1}, {1000, 2000});

    const uint16_t* values = vectors.typed_values_data<uint16_t>();
    ASSERT_EQ(values[0], 1000);
    ASSERT_EQ(values[1], 2000);
}

TEST_P(SparseVectorsResidency, get_all_data) {
    auto vectors = make<float>(5, {0, 2}, {0, 1}, {1.0F, 2.0F});

    auto data = vectors.get_all_data();
    ASSERT_NE(data.indptr_data, nullptr);
    ASSERT_NE(data.indices_data, nullptr);
    ASSERT_NE(data.values_data, nullptr);
    ASSERT_EQ(data.indptr_data[0], 0);
    ASSERT_EQ(data.indices_data[0], 0);
    ASSERT_FLOAT_EQ(data.values_data[0], 1.0F);
}

TEST_P(SparseVectorsResidency, serialize_deserialize_float) {
    auto original = make<float>(5, {0, 3}, {0, 2, 4}, {1.0F, 2.0F, 3.0F});

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.num_vectors(), 1);
    ASSERT_EQ(loaded.get_dimension(), 5);
    ASSERT_EQ(loaded.get_element_size(), nsparse::U32);

    auto dense = loaded.get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense[0], 1.0F);
    ASSERT_FLOAT_EQ(dense[2], 2.0F);
    ASSERT_FLOAT_EQ(dense[4], 3.0F);
}

TEST_P(SparseVectorsResidency, serialize_deserialize_multiple_vectors) {
    auto original = make<uint8_t>(4, {0, 2, 4}, {0, 1, 2, 3}, {10, 20, 30, 40});

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.num_vectors(), 2);

    auto dense0 = loaded.get_dense_vector(0);
    ASSERT_EQ(dense0[0], 10);
    ASSERT_EQ(dense0[1], 20);

    auto dense1 = loaded.get_dense_vector(1);
    ASSERT_EQ(dense1[2], 30);
    ASSERT_EQ(dense1[3], 40);
}

TEST_P(SparseVectorsResidency, move_constructor) {
    auto original = make<float>(5, {0, 2}, {0, 1}, {1.0F, 2.0F});

    nsparse::SparseVectors moved(std::move(original));

    ASSERT_EQ(moved.num_vectors(), 1);
    auto dense = moved.get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense[0], 1.0F);
    ASSERT_FLOAT_EQ(dense[1], 2.0F);
}

// An odd number of terms is the case padding exists for: it leaves the values
// array 2 bytes short of a float boundary unless the writer pads.
TEST_P(SparseVectorsResidency, round_trips_an_odd_term_count) {
    auto original = make<float>(4, {0, 3}, {0, 1, 3}, {1.5F, 2.5F, 3.5F});

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    auto dense = loaded.get_dense_vector_float(0);
    ASSERT_FLOAT_EQ(dense[0], 1.5F);
    ASSERT_FLOAT_EQ(dense[1], 2.5F);
    ASSERT_FLOAT_EQ(dense[2], 0.0F);
    ASSERT_FLOAT_EQ(dense[3], 3.5F);
}

// ---------------------------------------------------------------------------
// Construction and the copying add path.
// ---------------------------------------------------------------------------

TEST(SparseVectors, default_constructor) {
    nsparse::SparseVectors vectors;
    ASSERT_EQ(vectors.num_vectors(), 0);
}

TEST(SparseVectors, config_constructor) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});
    ASSERT_EQ(vectors.num_vectors(), 0);
    ASSERT_EQ(vectors.get_dimension(), 10);
    ASSERT_EQ(vectors.get_element_size(), nsparse::U32);
}

TEST(SparseVectors, config_constructor_throws_on_zero_dimension) {
    ASSERT_THROW(
        nsparse::SparseVectors({.element_size = nsparse::U32, .dimension = 0}),
        std::invalid_argument);
}

TEST(SparseVectors, add_vector_single_float) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 5});
    std::vector<nsparse::term_t> indices = {0, 2, 4};
    std::vector<float> values = {1.0F, 2.0F, 3.0F};

    vectors.add_vector(indices.data(), indices.size(),
                       reinterpret_cast<const uint8_t*>(values.data()),
                       values.size() * sizeof(float));

    ASSERT_EQ(vectors.num_vectors(), 1);
}

TEST(SparseVectors, add_vector_single_uint8) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = 5});
    std::vector<nsparse::term_t> indices = {1, 3};
    std::vector<uint8_t> values = {100, 200};

    vectors.add_vector(indices, values);

    ASSERT_EQ(vectors.num_vectors(), 1);
}

// Offsets are rebased onto what is already stored, so a second call must not
// restart at zero.
TEST(SparseVectors, add_vector_multiple) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});

    std::vector<nsparse::term_t> indices1 = {0, 1};
    std::vector<float> values1 = {1.0F, 2.0F};
    vectors.add_vector(indices1.data(), indices1.size(),
                       reinterpret_cast<const uint8_t*>(values1.data()),
                       values1.size() * sizeof(float));

    std::vector<nsparse::term_t> indices2 = {2, 3, 4};
    std::vector<float> values2 = {3.0F, 4.0F, 5.0F};
    vectors.add_vector(indices2.data(), indices2.size(),
                       reinterpret_cast<const uint8_t*>(values2.data()),
                       values2.size() * sizeof(float));

    ASSERT_EQ(vectors.num_vectors(), 2);
    ASSERT_EQ(vectors.indptr_data()[2], 5);
    ASSERT_FLOAT_EQ(vectors.get_dense_vector_float(1)[4], 5.0F);
}

TEST(SparseVectors, add_vectors_batch_float) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});

    // Two vectors: [0,1] and [2,3,4]
    std::vector<nsparse::offset_t> indptr = {0, 2, 5};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3, 4};
    std::vector<float> values = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};

    vectors.add_vectors(indptr.data(), indptr.size(), indices.data(),
                        indices.size(),
                        reinterpret_cast<const uint8_t*>(values.data()),
                        values.size() * sizeof(float));

    ASSERT_EQ(vectors.num_vectors(), 2);
}

TEST(SparseVectors, add_vectors_batch_uint8) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = 10});

    std::vector<nsparse::offset_t> indptr = {0, 2, 4};
    std::vector<nsparse::term_t> indices = {0, 1, 2, 3};
    std::vector<uint8_t> values = {10, 20, 30, 40};

    vectors.add_vectors(indptr, indices, values);

    ASSERT_EQ(vectors.num_vectors(), 2);
}

TEST(SparseVectors, add_vectors_batch_appends) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U8, .dimension = 4});

    vectors.add_vectors({0, 2}, {0, 1}, {10, 20});
    vectors.add_vectors({0, 2}, {2, 3}, {30, 40});

    ASSERT_EQ(vectors.num_vectors(), 2);
    ASSERT_EQ(vectors.indptr_data()[2], 4);
    ASSERT_EQ(vectors.get_dense_vector(1)[3], 40);
}

TEST(SparseVectors, add_vectors_empty_indptr) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});

    std::vector<nsparse::offset_t> indptr = {0};  // Less than 2 elements
    std::vector<nsparse::term_t> indices = {};
    std::vector<uint8_t> values = {};

    vectors.add_vectors(indptr, indices, values);

    ASSERT_EQ(vectors.num_vectors(), 0);
}

TEST(SparseVectors, add_vectors_throws_on_size_mismatch) {
    nsparse::SparseVectors vectors(
        {.element_size = nsparse::U32, .dimension = 10});

    std::vector<nsparse::offset_t> indptr = {0, 2};
    std::vector<nsparse::term_t> indices = {0, 1};
    std::vector<uint8_t> values = {1, 2};  // Should be 8 bytes for 2 floats

    ASSERT_THROW(vectors.add_vectors(indptr, indices, values),
                 std::invalid_argument);
}

TEST(SparseVectors, serialize_deserialize_empty) {
    nsparse::SparseVectors original;

    nsparse::BufferedIOWriter writer;
    original.serialize(&writer);

    nsparse::BufferedIOReader reader(writer.data());
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.num_vectors(), 0);
}

// ---------------------------------------------------------------------------
// map_vectors: the borrowing path's own contract.
// ---------------------------------------------------------------------------

TEST(SparseVectorsMapVectors, borrows_without_copying) {
    const std::vector<nsparse::offset_t> indptr = {0, 2};
    const std::vector<nsparse::term_t> indices = {0, 1};
    const std::vector<float> values = {1.0F, 2.0F};

    auto vectors = nsparse::SparseVectors::map_vectors(
        {.element_size = nsparse::U32, .dimension = 4}, indptr.data(),
        indptr.size(), indices.data(), indices.size(),
        reinterpret_cast<const uint8_t*>(values.data()),
        values.size() * sizeof(float));

    ASSERT_EQ(vectors.indptr_data(), indptr.data());
    ASSERT_EQ(vectors.indices_data(), indices.data());
    ASSERT_EQ(vectors.values_data_float(), values.data());
}

TEST(SparseVectorsMapVectors, an_empty_indptr_maps_nothing) {
    auto vectors = nsparse::SparseVectors::map_vectors(
        {.element_size = nsparse::U32, .dimension = 4}, nullptr, 0, nullptr, 0,
        nullptr, 0);

    ASSERT_EQ(vectors.num_vectors(), 0);
    ASSERT_EQ(vectors.get_dimension(), 4);
}

TEST(SparseVectorsMapVectors, cannot_be_appended_to) {
    const std::vector<nsparse::offset_t> indptr = {0, 1};
    const std::vector<nsparse::term_t> indices = {0};
    const std::vector<uint8_t> values = {7};

    auto vectors = nsparse::SparseVectors::map_vectors(
        {.element_size = nsparse::U8, .dimension = 4}, indptr.data(),
        indptr.size(), indices.data(), indices.size(), values.data(),
        values.size());

    // Growing would mean writing into memory this SparseVectors does not own.
    ASSERT_THROW(vectors.add_vector({1}, {9}), std::runtime_error);
    ASSERT_THROW(vectors.add_vectors({0, 1}, {1}, {9}), std::runtime_error);
}

TEST(SparseVectorsMapVectors, throws_on_zero_dimension) {
    const std::vector<nsparse::offset_t> indptr = {0, 1};
    const std::vector<nsparse::term_t> indices = {0};
    const std::vector<uint8_t> values = {7};

    ASSERT_THROW(nsparse::SparseVectors::map_vectors(
                     {.element_size = nsparse::U8, .dimension = 0},
                     indptr.data(), indptr.size(), indices.data(),
                     indices.size(), values.data(), values.size()),
                 std::invalid_argument);
}

TEST(SparseVectorsMapVectors, throws_on_unsupported_element_size) {
    const std::vector<nsparse::offset_t> indptr = {0, 1};
    const std::vector<nsparse::term_t> indices = {0};
    const std::vector<uint8_t> values = {1, 2, 3};

    ASSERT_THROW(nsparse::SparseVectors::map_vectors(
                     {.element_size = 3, .dimension = 4}, indptr.data(),
                     indptr.size(), indices.data(), indices.size(),
                     values.data(), values.size()),
                 std::invalid_argument);
}

TEST(SparseVectorsMapVectors, throws_on_null_arrays) {
    const std::vector<nsparse::offset_t> indptr = {0, 1};
    const std::vector<nsparse::term_t> indices = {0};
    const std::vector<uint8_t> values = {7};
    const nsparse::SparseVectorsConfig config = {.element_size = nsparse::U8,
                                                 .dimension = 4};

    ASSERT_THROW(
        nsparse::SparseVectors::map_vectors(config, nullptr, indptr.size(),
                                            indices.data(), indices.size(),
                                            values.data(), values.size()),
        std::invalid_argument);
    ASSERT_THROW(
        nsparse::SparseVectors::map_vectors(config, indptr.data(),
                                            indptr.size(), nullptr,
                                            indices.size(), values.data(),
                                            values.size()),
        std::invalid_argument);
    ASSERT_THROW(nsparse::SparseVectors::map_vectors(
                     config, indptr.data(), indptr.size(), indices.data(),
                     indices.size(), nullptr, values.size()),
                 std::invalid_argument);
}

TEST(SparseVectorsMapVectors, throws_on_size_mismatch) {
    const std::vector<nsparse::offset_t> indptr = {0, 2};
    const std::vector<nsparse::term_t> indices = {0, 1};
    const std::vector<uint8_t> values = {1, 2};  // 2 floats need 8 bytes

    ASSERT_THROW(nsparse::SparseVectors::map_vectors(
                     {.element_size = nsparse::U32, .dimension = 4},
                     indptr.data(), indptr.size(), indices.data(),
                     indices.size(), values.data(), values.size()),
                 std::invalid_argument);
}

TEST(SparseVectorsMapVectors, throws_on_misaligned_values) {
    const std::vector<nsparse::offset_t> indptr = {0, 2};
    const std::vector<nsparse::term_t> indices = {0, 1};
    auto storage = misaligned_storage(2 * sizeof(float));
    const uint8_t* values = storage.data() + 1;
    ASSERT_NE(reinterpret_cast<uintptr_t>(values) % sizeof(float), 0);

    ASSERT_THROW(nsparse::SparseVectors::map_vectors(
                     {.element_size = nsparse::U32, .dimension = 4},
                     indptr.data(), indptr.size(), indices.data(),
                     indices.size(), values, 2 * sizeof(float)),
                 std::invalid_argument);
}

TEST(SparseVectorsMapVectors, throws_when_indptr_does_not_start_at_zero) {
    const std::vector<nsparse::offset_t> indptr = {1, 2};
    const std::vector<nsparse::term_t> indices = {0, 1};
    const std::vector<uint8_t> values = {7, 8};

    ASSERT_THROW(nsparse::SparseVectors::map_vectors(
                     {.element_size = nsparse::U8, .dimension = 4},
                     indptr.data(), indptr.size(), indices.data(),
                     indices.size(), values.data(), values.size()),
                 std::invalid_argument);
}

TEST(SparseVectorsMapVectors, throws_on_non_monotonic_indptr) {
    // Row 1 ends before it starts, so its length would underflow.
    const std::vector<nsparse::offset_t> indptr = {0, 2, 1, 2};
    const std::vector<nsparse::term_t> indices = {0, 1};
    const std::vector<uint8_t> values = {7, 8};

    ASSERT_THROW(nsparse::SparseVectors::map_vectors(
                     {.element_size = nsparse::U8, .dimension = 4},
                     indptr.data(), indptr.size(), indices.data(),
                     indices.size(), values.data(), values.size()),
                 std::invalid_argument);
}

TEST(SparseVectorsMapVectors, throws_when_indptr_does_not_end_at_the_count) {
    const std::vector<nsparse::offset_t> indptr = {0, 1};
    const std::vector<nsparse::term_t> indices = {0, 1};
    const std::vector<uint8_t> values = {7, 8};

    ASSERT_THROW(nsparse::SparseVectors::map_vectors(
                     {.element_size = nsparse::U8, .dimension = 4},
                     indptr.data(), indptr.size(), indices.data(),
                     indices.size(), values.data(), values.size()),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// The serialized layout: padding, and borrowing on read.
// ---------------------------------------------------------------------------

namespace {

// Header written before the arrays: vector_count, dimension, element_size.
constexpr size_t kSerializedHeaderBytes = 3 * sizeof(size_t);

nsparse::SparseVectors owned_vectors(size_t dimension, size_t element_size,
                                     const std::vector<nsparse::offset_t>& indptr,
                                     const std::vector<nsparse::term_t>& indices,
                                     const std::vector<uint8_t>& values) {
    nsparse::SparseVectors vectors(
        {.element_size = element_size, .dimension = dimension});
    vectors.add_vectors(indptr, indices, values);
    return vectors;
}

// Serializes and returns the bytes, for asserting on the layout directly.
std::vector<uint8_t> serialized(const nsparse::SparseVectors& vectors) {
    nsparse::BufferedIOWriter writer;
    vectors.serialize(&writer);
    return writer.data();
}

}  // namespace

// 3 terms leaves values 2 bytes past a float boundary, so 2 bytes go in.
TEST(SparseVectorsLayout, pads_values_to_the_element_width) {
    auto vectors = owned_vectors(4, nsparse::U32, {0, 3}, {0, 1, 3},
                                 std::vector<uint8_t>(3 * sizeof(float), 0));

    const size_t bytes = serialized(vectors).size();

    const size_t unpadded = kSerializedHeaderBytes +
                            2 * sizeof(nsparse::offset_t) +
                            3 * sizeof(nsparse::term_t) + 3 * sizeof(float);
    ASSERT_EQ(bytes, unpadded + 2);
    ASSERT_EQ(bytes % alignof(float), 0);
}

TEST(SparseVectorsLayout, pads_nothing_when_already_aligned) {
    auto vectors = owned_vectors(4, nsparse::U32, {0, 2}, {0, 1},
                                 std::vector<uint8_t>(2 * sizeof(float), 0));

    const size_t expected = kSerializedHeaderBytes +
                            2 * sizeof(nsparse::offset_t) +
                            2 * sizeof(nsparse::term_t) + 2 * sizeof(float);
    ASSERT_EQ(serialized(vectors).size(), expected);
}

// U8 values need no alignment, so an odd term count must not pad.
TEST(SparseVectorsLayout, pads_nothing_for_byte_wide_values) {
    auto vectors =
        owned_vectors(4, nsparse::U8, {0, 3}, {0, 1, 3}, {10, 20, 30});

    const size_t expected = kSerializedHeaderBytes +
                            2 * sizeof(nsparse::offset_t) +
                            3 * sizeof(nsparse::term_t) + 3;
    ASSERT_EQ(serialized(vectors).size(), expected);
}

// The padding is what makes the layout mappable, but deserialize() copies: an
// owned buffer is not the reader's memory.
TEST(SparseVectorsLayout, deserialize_owns_its_buffers) {
    auto original = owned_vectors(4, nsparse::U32, {0, 3}, {0, 1, 3},
                                  std::vector<uint8_t>(3 * sizeof(float), 0));
    const auto bytes = serialized(original);

    nsparse::BufferedIOReader reader(bytes);
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    const auto* base = bytes.data();
    const auto* end = base + bytes.size();
    const auto* indptr = reinterpret_cast<const uint8_t*>(loaded.indptr_data());
    const auto* values = loaded.values_data();
    ASSERT_TRUE(indptr < base || indptr >= end);
    ASSERT_TRUE(values < base || values >= end);
    // A copy is freshly allocated, so it is aligned for the element type
    // regardless of where the array sat in the stream.
    ASSERT_EQ(reinterpret_cast<uintptr_t>(loaded.values_data_float()) %
                  alignof(float),
              0);
}

// The padding has to be skipped, not read as data: the array after it would
// otherwise come back shifted, and whatever an enclosing index serialized next
// would be misread.
TEST(SparseVectorsLayout, leaves_the_stream_after_the_padding) {
    auto vectors = owned_vectors(4, nsparse::U32, {0, 3}, {0, 1, 3},
                                 std::vector<uint8_t>(3 * sizeof(float), 0));
    auto bytes = serialized(vectors);
    const size_t vectors_bytes = bytes.size();
    // A trailing marker, as an enclosing index's payload would be.
    const uint32_t marker = 0xFEEDFACE;
    const auto* marker_bytes = reinterpret_cast<const uint8_t*>(&marker);
    bytes.insert(bytes.end(), marker_bytes, marker_bytes + sizeof(marker));

    nsparse::BufferedIOReader reader(bytes);
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(reader.pos(), vectors_bytes);

    uint32_t read_back = 0;
    ASSERT_EQ(reader.read(&read_back, sizeof(read_back), 1), 1);
    ASSERT_EQ(read_back, marker);
}

TEST(SparseVectorsLayout, empty_vectors_write_only_the_count) {
    nsparse::SparseVectors empty;

    ASSERT_EQ(serialized(empty).size(), sizeof(size_t));
}

// U64 is where empty rows bite: the empty values array pads to 8 where the indptr
// before it only reached a multiple of 4, so 4 bytes go in. A reader that skips
// padding only for a non-empty array leaves them for the next payload to misread.
TEST(SparseVectorsLayout, empty_rows_round_trip_at_a_wide_element_size) {
    auto vectors = owned_vectors(4, nsparse::U64, {0, 0, 0}, {}, {});
    ASSERT_EQ(vectors.num_vectors(), 2);

    auto bytes = serialized(vectors);
    const size_t vectors_bytes = bytes.size();
    ASSERT_EQ(vectors_bytes % alignof(uint64_t), 0)
        << "the values array must land on its element boundary";

    const uint32_t marker = 0xFEEDFACE;
    const auto* marker_bytes = reinterpret_cast<const uint8_t*>(&marker);
    bytes.insert(bytes.end(), marker_bytes, marker_bytes + sizeof(marker));

    nsparse::BufferedIOReader reader(bytes);
    nsparse::SparseVectors loaded;
    loaded.deserialize(&reader);

    ASSERT_EQ(loaded.num_vectors(), 2);
    ASSERT_EQ(reader.pos(), vectors_bytes);

    uint32_t read_back = 0;
    ASSERT_EQ(reader.read(&read_back, sizeof(read_back), 1), 1);
    ASSERT_EQ(read_back, marker);
}

// The two readers must agree on where an enclosing index's payload begins.
TEST(SparseVectorsLayout, mapped_and_copied_reads_consume_the_same_bytes) {
    auto vectors = owned_vectors(4, nsparse::U64, {0, 0, 0}, {}, {});
    const auto bytes = serialized(vectors);

    nsparse::BufferedIOReader reader(bytes);
    nsparse::SparseVectors copied;
    copied.deserialize(&reader);

    nsparse::MmapCursor cursor(bytes.data(), bytes.size());
    nsparse::SparseVectors mapped;
    mapped.mmap_deserialize(&cursor);

    ASSERT_EQ(cursor.pos(), reader.pos());
    ASSERT_EQ(cursor.pos(), bytes.size());
}

// element_size reaches padding_for() as an alignment, where zero used to trap
// rather than throw. Both readers have to reject it the same way.
TEST(SparseVectorsLayout, rejects_a_zero_element_width) {
    auto vectors = owned_vectors(4, nsparse::U32, {0, 2}, {0, 1},
                                 std::vector<uint8_t>(2 * sizeof(float), 0));
    auto bytes = serialized(vectors);

    const size_t element_size_offset = 2 * sizeof(size_t);
    const size_t zero = 0;
    std::memcpy(bytes.data() + element_size_offset, &zero, sizeof(zero));

    nsparse::BufferedIOReader reader(bytes);
    nsparse::SparseVectors copied;
    ASSERT_THROW(copied.deserialize(&reader), std::runtime_error);

    nsparse::MmapCursor cursor(bytes.data(), bytes.size());
    nsparse::SparseVectors mapped;
    ASSERT_THROW(mapped.mmap_deserialize(&cursor), std::runtime_error);
}
