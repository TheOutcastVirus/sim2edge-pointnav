#pragma once

#include "sim2edge/actor.hpp"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cmath>

namespace sim2edge {
// One in-flight evaluation and one replaceable input; never queue old observations.
class AsyncActor final : public Actor {
public:
  explicit AsyncActor(std::shared_ptr<Actor> actor, double max_age_s = 0.5)
  : actor_(std::move(actor)), max_age_s_(max_age_s), worker_([this] { work(); }) {}
  ~AsyncActor() override {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    ready_.notify_one();
    worker_.join();
  }
  const ActorInfo & info() const override { return actor_->info(); }
  Command run(const Observation & obs) override {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_ = obs;
    pending_at_ = Clock::now();
    have_pending_ = true;
    ready_.notify_one();
    return valid_ && std::chrono::duration<double>(Clock::now() - result_at_).count()
      <= max_age_s_ ? result_ : Command{};
  }
  void invalidate() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++generation_;
    valid_ = have_pending_ = false;
  }
  struct Stats { double latency_ms; double age_s; unsigned long updates; bool valid; };
  Stats stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    const double age = std::chrono::duration<double>(Clock::now() - result_at_).count();
    return {latency_ms_, age, updates_, valid_ && age <= max_age_s_};
  }
private:
  using Clock = std::chrono::steady_clock;
  void work() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
      ready_.wait(lock, [this] { return stopping_ || have_pending_; });
      if (stopping_) return;
      const auto obs = pending_;
      const auto started = pending_at_;
      const auto generation = generation_;
      have_pending_ = false;
      lock.unlock();
      Command command{};
      bool ok = true;
      try { command = actor_->run(obs); }
      catch (...) { ok = false; }
      ok = ok && std::isfinite(command.v) && std::isfinite(command.omega);
      lock.lock();
      latency_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
      ++updates_;
      if (generation == generation_) {
        result_ = command;
        result_at_ = started; // Age includes inference time, not just delivery time.
        valid_ = ok;
      }
    }
  }
  std::shared_ptr<Actor> actor_;
  double max_age_s_;
  std::mutex mutex_;
  std::condition_variable ready_;
  Observation pending_{};
  Command result_{};
  Clock::time_point pending_at_{}, result_at_{};
  unsigned long generation_{0};
  unsigned long updates_{0};
  double latency_ms_{0.0};
  bool valid_{false}, have_pending_{false}, stopping_{false};
  std::thread worker_;
};
} // namespace sim2edge
