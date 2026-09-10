/**
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

#ifndef IO_ALIGN_H
#define IO_ALIGN_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "nsparse/io/io.h"
#include "nsparse/types.h"
#include "nsparse/utils/buf.h"
#include "nsparse/utils/mmap_cursor.h"

// Alignment padding for serialized arrays.
//
// Nothing about a stream of writes lands each array on a boundary its element
// type can be dereferenced from -- a term_t array of odd length leaves the next
// float array 2 bytes off. So a serializer pads before writing each array and a
// deserializer skips the same padding before reading it, at a cost of under 8
// bytes per array.
//
// Today's readers copy, and a copy is aligned wherever it came from. The padding
// is for the reader that does not: borrowing an array in place needs it aligned
// in the file, since misaligned loads are UB on x86 and fault on ARM. Writing it
// now keeps that option open without a second on-disk format.
namespace nsparse::io_align {

// Largest alignment any serialized element needs, and so the most padding a
// single call can insert.
constexpr size_t kMaxAlignment = 8;
static_assert(alignof(offset_t) <= kMaxAlignment,
              "offset_t alignment must fit the array-padding budget");

// Bytes to insert at `pos` to reach an `alignment` boundary. Zero is reachable
// input, not a caller bug: the alignment can be an element width read from the
// file, and `pos % 0` would trap rather than throw.
constexpr size_t padding_for(size_t pos, size_t alignment) {
    if (alignment == 0) {
        throw std::runtime_error("index file declares a zero element width");
    }
    return (alignment - pos % alignment) % alignment;
}

// Pads the stream so the next write starts on an `alignment` boundary.
inline void pad_to(IOWriter* writer, size_t alignment) {
    const size_t bytes = padding_for(writer->pos(), alignment);
    if (bytes > 0) {
        std::array<uint8_t, kMaxAlignment> zeros{};
        writer->write(zeros.data(), 1, bytes);
    }
}

// Consumes the padding pad_to() wrote at the same point in the layout.
inline void skip_padding(IOReader* reader, size_t alignment) {
    const size_t bytes = padding_for(reader->pos(), alignment);
    if (bytes > 0) {
        std::array<uint8_t, kMaxAlignment> discard{};
        reader->read(discard.data(), 1, bytes);
    }
}

// Same, for the mapped reader.
inline void skip_padding(MmapCursor* cursor, size_t alignment) {
    cursor->skip(padding_for(cursor->pos(), alignment));
}

// The three below are the only way to move an aligned array on or off the wire.
// They pad unconditionally, count == 0 included: an empty array must consume the
// same bytes on both sides or every later offset shifts, and a `count > 0` guard
// on the padding is what lets writer and reader disagree.
//
// `alignment` defaults to T's; a byte array reinterpreted at a wider element size
// on read needs that width instead.

template <class T>
void write_padded(IOWriter* writer, const T* data, size_t count,
                  size_t alignment = alignof(T)) {
    pad_to(writer, alignment);
    if (count > 0) {
        writer->write(const_cast<T*>(data), sizeof(T), count);
    }
}

// read_owned, preceded by the padding write_padded() wrote.
template <class T>
Buf<T> read_padded(IOReader* reader, size_t count,
                   size_t alignment = alignof(T)) {
    skip_padding(reader, alignment);
    return read_owned<T>(reader, count);
}

// Borrows the array in place where read_padded() copies it.
template <class T>
Buf<T> borrow_padded(MmapCursor* cursor, size_t count,
                     size_t alignment = alignof(T)) {
    skip_padding(cursor, alignment);
    return Buf<T>::borrow(cursor->read_array<T>(count), count);
}

}  // namespace nsparse::io_align

#endif  // IO_ALIGN_H
