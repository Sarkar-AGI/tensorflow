/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <fcntl.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"

namespace tflite {
namespace {

uint64_t GetFdSizeBytes(int fd) {
  if (fd < 0) {
    return 0;
  }

  struct stat fd_stat;
  if (fstat(fd, &fd_stat) != 0) {
    return 0;
  }

  return fd_stat.st_size < 0 ? 0 : fd_stat.st_size;
}

}  // namespace

MMAPAllocation::MMAPAllocation(const char* filename,
                               ErrorReporter* error_reporter, bool map_private)
    : MMAPAllocation(error_reporter, open(filename, O_RDONLY), map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Could not open '%s': %s", filename,
                         strerror(errno));
  }
}

MMAPAllocation::MMAPAllocation(int fd, ErrorReporter* error_reporter,
                               bool map_private)
    : MMAPAllocation(error_reporter, dup(fd), map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to dup '%d' file descriptor: %s", fd,
                         strerror(errno));
  }
}

MMAPAllocation::MMAPAllocation(const char* filename, off_t offset,
                               size_t length, ErrorReporter* error_reporter,
                               bool map_private)
    : MMAPAllocation(error_reporter, open(filename, O_RDONLY), offset, length,
                     map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Could not open '%s': %s", filename,
                         strerror(errno));
  }
}

MMAPAllocation::MMAPAllocation(int fd, off_t offset, size_t length,
                               ErrorReporter* error_reporter, bool map_private)
    : MMAPAllocation(error_reporter, dup(fd), offset, length, map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to dup '%d' file descriptor: %s", fd,
                         strerror(errno));
  }
}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd,
                               bool map_private)
    : MMAPAllocation(error_reporter, owned_fd, /*offset=*/0,
                     /*length=*/GetFdSizeBytes(owned_fd), map_private) {}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd,
                               uint64_t offset, uint64_t length,
                               bool map_private)
    : Allocation(error_reporter, Allocation::Type::kMMap),
      mmap_fd_(owned_fd),
      mmapped_buffer_(MAP_FAILED),
      buffer_size_bytes_(static_cast<size_t>(length)) {
  if (owned_fd < 0) {
    return;
  }

  if (offset > SIZE_MAX || length > SIZE_MAX) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Asked to mmap '%llu' bytes from fd '%d' at offset "
        "'%llu', which exceeds the maximum allowed size_t (%llu).",
        static_cast<unsigned long long>(length), mmap_fd_,
        static_cast<unsigned long long>(offset),
        static_cast<unsigned long long>(SIZE_MAX));
    return;
  }

  static const int64_t kPageSize = sysconf(_SC_PAGE_SIZE);
  if (kPageSize <= 0) {
    TF_LITE_REPORT_ERROR(error_reporter, "Could not get page size: %s",
                         strerror(errno));
    return;
  }

  offset_in_buffer_ = offset % kPageSize;
  offset_of_buffer_in_file_ = offset - offset_in_buffer_;

  uint64_t file_size = GetFdSizeBytes(mmap_fd_);
  if (offset > file_size || length > file_size - offset) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Asked to mmap '%llu' bytes from fd '%d' at offset "
                         "'%llu'. This is over the length of file '%llu'.",
                         static_cast<unsigned long long>(length), mmap_fd_,
                         static_cast<unsigned long long>(offset),
                         static_cast<unsigned long long>(file_size));
    return;
  }

  if (length > SIZE_MAX - offset_in_buffer_) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Asked to mmap '%llu' bytes from fd '%d' at offset "
        "'%llu', which exceeds the maximum allowed mmap size (%llu).",
        static_cast<unsigned long long>(length + offset_in_buffer_), mmap_fd_,
        static_cast<unsigned long long>(offset),
        static_cast<unsigned long long>(SIZE_MAX));
    return;
  }

  mmapped_buffer_ =
      mmap(nullptr, /*length=*/static_cast<size_t>(length + offset_in_buffer_),
           map_private ? (PROT_READ | PROT_WRITE) : PROT_READ,
           map_private ? MAP_PRIVATE : MAP_SHARED, mmap_fd_,
           /*offset=*/offset_of_buffer_in_file_);
  if (mmapped_buffer_ == MAP_FAILED) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Mmap of fd '%d' at offset '%llu' failed with error '%s'.", mmap_fd_,
        static_cast<unsigned long long>(offset), strerror(errno));
    return;
  }
}

MMAPAllocation::~MMAPAllocation() {
  if (valid()) {
    munmap(const_cast<void*>(mmapped_buffer_),
           buffer_size_bytes_ + offset_in_buffer_);
  }
  if (mmap_fd_ >= 0) {
    close(mmap_fd_);
  }
}

const void* MMAPAllocation::base() const {
  return static_cast<const char*>(mmapped_buffer_) + offset_in_buffer_;
}

size_t MMAPAllocation::bytes() const { return buffer_size_bytes_; }

bool MMAPAllocation::valid() const { return mmapped_buffer_ != MAP_FAILED; }

bool MMAPAllocation::IsSupported() { return true; }

}  // namespace tflite
