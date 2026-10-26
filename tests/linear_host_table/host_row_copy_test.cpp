// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include "cpu_ops/host_row_copy.hpp"
#include <array>
#include <atomic>
#include <thread>

TEST(HostRowCopy, ConcurrentMixedRowsAndFinalValues) {
  for (size_t bytes : {1, 2, 6, 8, 14, 32, 1024}) {
    for (size_t offset = 0; offset < 8; ++offset) {
      alignas(8) std::array<unsigned char, 1032> storage{};
      std::array<unsigned char, 1024> first{}, second{}, output{};
      first.fill(0x55);
      second.fill(0xaa);
      auto* row = storage.data() + offset;
      std::atomic<bool> ready{false}, done{false};
      bool valid = true;
      std::thread reader([&] {
        ready.store(true);
        do {
          nve::copy_host_row<false>(output.data(), row, bytes);
          for (size_t i = 0; i < bytes; ++i)
            valid &= output[i] == 0 || output[i] == 0x55 || output[i] == 0xaa;
        } while (!done.load());
      });
      while (!ready.load()) std::this_thread::yield();
      for (int i = 0; i < 5000; ++i)
        nve::copy_host_row<true>((i % 2 ? second : first).data(), row, bytes);
      done.store(true);
      reader.join();
      EXPECT_TRUE(valid);
      nve::copy_host_row<false>(output.data(), row, bytes);
      for (size_t i = 0; i < bytes; ++i) EXPECT_EQ(output[i], 0xaa);
    }
  }
}
