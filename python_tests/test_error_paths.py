# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

"""How the bindings behave when a caller gets it wrong.

C++ exceptions are translated, so these are ordinary in-process assertions:
std::invalid_argument arrives as ValueError, everything else derived from
std::exception as RuntimeError.

A wrong dtype on any buffer argument raises TypeError. The buffer typemaps
zero-initialise their Py_buffer view, so when one argument fails validation and
runs SWIG_fail, the freearg typemaps for the remaining arguments release a
zeroed view (a no-op) rather than uninitialised stack memory.
"""

import numpy as np
import pytest

import nsparse
from support import make_corpus, make_index, search

DIM = 512


@pytest.fixture(scope="module")
def small_corpus():
    return make_corpus(200, DIM, 20, 0x11)


@pytest.fixture(scope="module")
def small_queries():
    return make_corpus(4, DIM, 6, 0x22)


@pytest.mark.parametrize(
    "description,message",
    [
        ("nonsense", "Unknown index type"),
        ("", "Description cannot be null or empty"),
        ("idmap", "idmap requires a delegate index type"),
    ],
)
def test_bad_index_spec_raises(description, message):
    with pytest.raises(ValueError, match=message):
        nsparse.index_factory(DIM, description)


def test_build_on_brutal_raises(small_corpus):
    """brutal has no build step: Index::build() is the base throw and brutal
    does not override it."""
    with pytest.raises(RuntimeError, match="not implemented"):
        make_index("brutal", small_corpus, needs_build=True)


def test_write_index_on_brutal_raises(small_corpus, tmp_path):
    """brutal is not serialisable."""
    index = make_index("brutal", small_corpus, needs_build=False)
    with pytest.raises(RuntimeError, match="does not support"):
        nsparse.write_index(index, str(tmp_path / "brutal.idx"))


def test_non_positive_k_raises(small_corpus, small_queries):
    index = make_index("inverted", small_corpus)
    with pytest.raises(ValueError, match="must be positive"):
        search(index, small_queries, k=0)


def test_term_id_out_of_range_raises(small_corpus):
    """Terms must be < dim. Rejected at build, not at add."""
    indices = small_corpus.indices.copy()
    indices[0] = DIM  # one past the last valid term
    index = nsparse.index_factory(DIM, "inverted")
    index.add(small_corpus.n, small_corpus.indptr, indices, small_corpus.values)
    with pytest.raises(ValueError, match="term_id out of range"):
        index.build()


def test_wrong_values_dtype_raises(small_corpus):
    """`values` is the last buffer argument, so its rejection is graceful."""
    index = nsparse.index_factory(DIM, "inverted")
    with pytest.raises(TypeError, match="float32"):
        index.add(
            small_corpus.n,
            small_corpus.indptr,
            small_corpus.indices,
            small_corpus.values.astype(np.float64),
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: (c.indptr.astype(np.float64), c.indices, c.values), "indptr"),
        (lambda c: (c.indptr, c.indices.astype(np.int32), c.values), "indices"),
    ],
    ids=["bad_indptr_dtype", "bad_indices_dtype"],
)
def test_wrong_dtype_on_non_final_buffer_raises(small_corpus, mutate, match):
    """A wrong dtype on indptr or indices raises TypeError, not a segfault: the
    zero-initialised Py_buffer views make every argument fail gracefully."""
    index = nsparse.index_factory(DIM, "inverted")
    indptr, indices, values = mutate(small_corpus)
    with pytest.raises(TypeError, match=match):
        index.add(small_corpus.n, indptr, indices, values)
