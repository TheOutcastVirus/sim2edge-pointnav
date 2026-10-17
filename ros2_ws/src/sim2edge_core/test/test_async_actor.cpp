#include <gtest/gtest.h>
#include <sim2edge/async_actor.hpp>
#include <atomic>
#include <future>

namespace {
class GatedActor : public sim2edge::Actor {
public:
  std::atomic<int> calls{0};
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  sim2edge::ActorInfo metadata;
  sim2edge::Command run(const sim2edge::Observation &) override {
    ++calls;
    gate.wait();
    return {0.2, 0.3};
  }
  const sim2edge::ActorInfo & info() const override { return metadata; }
};
bool waitFor(const std::function<bool()> & predicate) {
  for (int i = 0; i < 1000; ++i) {
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}
TEST(AsyncActor, ControlDoesNotWaitAndInvalidationRejectsInflightResult) {
  auto backend = std::make_shared<GatedActor>();
  sim2edge::AsyncActor actor(backend);
  sim2edge::Observation obs{};
  EXPECT_EQ(actor.run(obs).v, 0.0);
  EXPECT_TRUE(waitFor([&] { return backend->calls == 1; }));
  const auto started = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) EXPECT_EQ(actor.run(obs).v, 0.0);
  EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(), 0.05);
  actor.invalidate();
  backend->release.set_value();
  EXPECT_TRUE(waitFor([&] { return actor.stats().updates == 1; }));
  EXPECT_FALSE(actor.stats().valid);
  EXPECT_EQ(actor.run(obs).v, 0.0);
  EXPECT_TRUE(waitFor([&] { return actor.stats().valid; }));
  EXPECT_NEAR(actor.run(obs).v, 0.2, 1e-6);
}
TEST(AsyncActor, ResultAgeIncludesInferenceLatency) {
  auto backend = std::make_shared<GatedActor>();
  sim2edge::AsyncActor actor(backend, 0.02);
  actor.run(sim2edge::Observation{});
  EXPECT_TRUE(waitFor([&] { return backend->calls == 1; }));
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  backend->release.set_value();
  EXPECT_TRUE(waitFor([&] { return actor.stats().updates == 1; }));
  EXPECT_FALSE(actor.stats().valid);
}
}
