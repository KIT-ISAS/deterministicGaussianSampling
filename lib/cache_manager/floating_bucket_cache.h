#ifndef FLOATING_BUCKET_CACHE_H
#define FLOATING_BUCKET_CACHE_H

#include <cassert>
#include <cstdint>
#include <iostream>
#include <unordered_map>

#define FLOATING_BUCKET_CACHE_SIZE 1.25

class floatingBucketCacheManager {
 public:
  floatingBucketCacheManager(size_t size) {
    cache.reserve((size_t)((double)size * FLOATING_BUCKET_CACHE_SIZE));
  };

  inline bool get(double floatingKey, double* value) const {
    const uint64_t hashKey = hashFloatingKey(floatingKey);
    auto it = cache.find(hashKey);
    if (it != cache.end()) {
      *value = it->second;
      return true;
    }
    return false;
  }

  inline void set(double floatingKey, double value) {
    const uint64_t hashKey = hashFloatingKey(floatingKey);
    cache[hashKey] = value;
  }

  inline void clear() { cache.clear(); }

 private:
  static constexpr double epsilon = 1e-15;

  std::unordered_map<uint64_t, double> cache;

  inline uint64_t hashFloatingKey(double floatingKey) const {
    return static_cast<uint64_t>(floatingKey / epsilon);
  }
};

class floatingBucketCacheManagerIntKey {
 public:
  floatingBucketCacheManagerIntKey(size_t size) {
    cache.reserve((size_t)((double)size * FLOATING_BUCKET_CACHE_SIZE));
  }

  inline bool get(double floatingKey, size_t intKey, double* value) const {
    const uint64_t hashKey = hashFloatingKey(floatingKey, intKey);
    auto it = cache.find(hashKey);
    if (it != cache.end()) {
      *value = it->second;
      return true;
    }
    return false;
  }

  inline void set(double floatingKey, size_t intKey, double value) {
    const uint64_t hashKey = hashFloatingKey(floatingKey, intKey);
    cache[hashKey] = value;
  }

  inline void clear() { cache.clear(); }

 private:
  static constexpr double epsilon = 1e-9;
  static constexpr uint64_t floatingBits = 40;
  static constexpr uint64_t intBits = 24;

  static constexpr uint64_t floatingMask = (uint64_t(1) << floatingBits) - 1;
  static constexpr uint64_t intMask = (uint64_t(1) << intBits) - 1;

  std::unordered_map<uint64_t, double> cache;

  inline uint64_t hashFloatingKey(double floatingKey, size_t intKey) const {
    assert(intKey <= intMask);
    return (static_cast<uint64_t>(floatingKey / epsilon) & floatingMask) |
           ((static_cast<uint64_t>(intKey) & intMask) << floatingBits);
  }
};

#endif  // FLOATING_BUCKET_CACHE_H
