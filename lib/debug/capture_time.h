#ifndef CAPTURE_TIME_H
#define CAPTURE_TIME_H

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

class CaptureTime {
 private:
  // Struct to store time data for each named timer
  struct TimerData {
    std::chrono::high_resolution_clock::time_point startTime;
    std::vector<std::chrono::nanoseconds> recordedTimes;
  };

  // Static map to store multiple timers by name
  static std::map<std::string, TimerData> timers;

  // Instance-specific variables for RAII-based timer
  std::string instanceName;
  std::chrono::high_resolution_clock::time_point instanceStartTime;
  bool instanceActive;

 public:
  // Constructor (RAII-based timer that prints on destruction)
  explicit CaptureTime(const std::string& name)
      : instanceName(name),
        instanceStartTime(std::chrono::high_resolution_clock::now()),
        instanceActive(true) {}

  // Destructor (prints elapsed time)
  ~CaptureTime() {
    if (instanceActive) {
      auto endTime = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          endTime - instanceStartTime);
      std::cout << "[" << instanceName
                << "] Elapsed time: " << formatTime(elapsed) << "\n";
    }
  }

  // Start a named static timer
  static void start(const std::string& name) {
    timers[name].startTime = std::chrono::high_resolution_clock::now();
  }

  // Stop a named static timer and record elapsed time
  static void stop(const std::string& name) {
    auto endTime = std::chrono::high_resolution_clock::now();
    timers[name].recordedTimes.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            endTime - timers[name].startTime));
  }

  // Print statistics for a specific named timer
  static void printStatistics(const std::string& name) {
    if (timers.find(name) == timers.end() ||
        timers[name].recordedTimes.empty()) {
      std::cout << "[" << name << "] No data recorded.\n";
      return;
    }

    std::vector<std::chrono::nanoseconds> sortedTimes =
        timers[name].recordedTimes;
    std::sort(sortedTimes.begin(), sortedTimes.end());

    auto minTime = sortedTimes.front();
    auto maxTime = sortedTimes.back();
    auto medianTime = sortedTimes[sortedTimes.size() / 2];
    auto avgTime = std::accumulate(sortedTimes.begin(), sortedTimes.end(),
                                   std::chrono::nanoseconds(0)) /
                   sortedTimes.size();

    std::cout << "[" << name << "] Statistics:\n"
              << "  Min: " << formatTime(minTime) << "\n"
              << "  Max: " << formatTime(maxTime) << "\n"
              << "  Median: " << formatTime(medianTime) << "\n"
              << "  Average: " << formatTime(avgTime) << "\n";
  }

  // Print statistics for all named timers
  static void printAllStatistics() {
    for (const auto& [name, _] : timers) {
      printStatistics(name);
    }
  }

 private:
  // Convert nanoseconds to a readable format
  static std::string formatTime(std::chrono::nanoseconds ns) {
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(ns).count();
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(ns).count() %
        1000;
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(ns).count() %
        1000;
    auto ns_only = ns.count() % 1000;

    return "<" + std::to_string(sec) + "s-" + std::to_string(ms) + "ms-" +
           std::to_string(us) + "us-" + std::to_string(ns_only) + "ns>";
  }
};

#endif
