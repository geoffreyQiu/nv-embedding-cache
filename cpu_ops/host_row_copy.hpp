/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace nve {

// GCC/Clang alias types permit access to a caller-owned, typeless row buffer.
// Both readers and writers select the same aligned word size from the TABLE
// address and row size, independently of the alignment of their private buffer.
// Relaxed access deliberately allows mixed generations within a row. It adds
// no waiting, fences, or ordering between lookup and update. Raw tensor writes,
// accumulation and serialization must still happen without concurrent readers.
template <typename Word, bool Write>
inline void copy_host_words(void* local, void* table, std::size_t bytes) {
  static_assert(__atomic_always_lock_free(sizeof(Word), nullptr),
                "Host row access requires lock-free atomic words");
  auto* words = static_cast<Word*>(table);
  auto* buffer = static_cast<unsigned char*>(local);
  for (std::size_t i = 0; i < bytes / sizeof(Word); ++i) {
    Word value;
    if constexpr (Write) {
      std::memcpy(&value, buffer + i * sizeof(Word), sizeof(Word));
      __atomic_store_n(words + i, value, __ATOMIC_RELAXED);
    } else {
      value = __atomic_load_n(words + i, __ATOMIC_RELAXED);
      std::memcpy(buffer + i * sizeof(Word), &value, sizeof(Word));
    }
  }
}

template <bool Write>
inline void copy_host_row(void* local, void* table, std::size_t bytes) {
  using Word64 = uint64_t __attribute__((__may_alias__));
  using Word32 = uint32_t __attribute__((__may_alias__));
  using Word16 = uint16_t __attribute__((__may_alias__));
  const auto alignment = reinterpret_cast<std::uintptr_t>(table) | bytes;
  if ((alignment & 7) == 0) copy_host_words<Word64, Write>(local, table, bytes);
  else if ((alignment & 3) == 0) copy_host_words<Word32, Write>(local, table, bytes);
  else if ((alignment & 1) == 0) copy_host_words<Word16, Write>(local, table, bytes);
  else copy_host_words<unsigned char, Write>(local, table, bytes);
}

}  // namespace nve
