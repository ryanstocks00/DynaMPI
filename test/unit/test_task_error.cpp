/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <dynampi/task_error.hpp>
#include <string>

using namespace dynampi;
using namespace dynampi::detail;

TEST(TaskErrorMessage, LongMessageIsTruncatedWithEllipsis) {
  const std::string long_what(kMaxTaskErrorMessage + 50, 'x');
  const std::string truncated = truncate_task_error_message(long_what.c_str());
  EXPECT_EQ(truncated.size(), kMaxTaskErrorMessage);
  EXPECT_EQ(truncated.substr(truncated.size() - 3), "...");
}

TEST(TaskErrorLog, WarnIfUnreportedDoesNotThrowWithPendingErrors) {
  TaskErrorLog log;
  log.record(TaskError{0, "boom"});
  // Exercises the fprintf(stderr, ...) path; only observable via stderr, so
  // this just confirms it is noexcept and safe to call.
  EXPECT_NO_THROW(log.warn_if_unreported("TestDistributor"));
}

TEST(TaskErrorCodec, DecodeRoundTripsAnEncodedError) {
  const TaskError original{3, "boom"};
  const TaskError decoded = decode_task_error(encode_task_error(original));
  EXPECT_EQ(decoded.worker_rank, original.worker_rank);
  EXPECT_EQ(decoded.message, original.message);
}

TEST(TaskErrorCodec, DecodeFallsBackToRankMinusOneOnCorruptedRankPrefix) {
  const TaskError decoded = decode_task_error("not-a-number\nmessage");
  EXPECT_EQ(decoded.worker_rank, -1);
  EXPECT_EQ(decoded.message, "message");
}

TEST(TaskErrorCodec, DecodeWithoutSeparatorUsesRankMinusOne) {
  const TaskError decoded = decode_task_error("no separator here");
  EXPECT_EQ(decoded.worker_rank, -1);
  EXPECT_EQ(decoded.message, "no separator here");
}

TEST(RunTaskGuarded, NonStdExceptionIsCaughtAndReported) {
  auto worker = [](int) -> int { throw 42; };
  int out = -1;
  auto failure = run_task_guarded<int>(worker, 0, out);
  ASSERT_TRUE(failure.has_value());
  EXPECT_NE(failure->find("unknown exception"), std::string::npos);
  EXPECT_EQ(out, 0);
}
