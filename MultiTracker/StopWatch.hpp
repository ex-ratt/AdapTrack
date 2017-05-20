/*
 * StopWatch.hpp
 *
 *  Created on: 23.09.2015
 *      Author: poschmann
 */

#ifndef STOPWATCH_HPP_
#define STOPWATCH_HPP_

#include <chrono>

/**
 * Stop watch for measuring execution times.
 */
class StopWatch {
public:

	/**
	 * Starts a new time measurement.
	 *
	 * @return Newly created stop watch that is used to stop the time measurement and retrieve the measured duration.
	 */
	static StopWatch start() {
		return StopWatch();
	}

	/**
	 * Stops the time measurement and returns the measured duration.
	 *
	 * Subsequent calls to this function will always return the duration between the original starting time and now,
	 * so the stop watch will never be restarted with a new time.
	 *
	 * @return Duration of the measurement.
	 */
	std::chrono::milliseconds stop() {
		stopTime = std::chrono::steady_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
	}

private:

	/**
	 * Constructs a new stop watch, starting the time measurement.
	 */
	StopWatch() : startTime(std::chrono::steady_clock::now()) {}

  std::chrono::steady_clock::time_point startTime; ///< Time of the measurement start.
  std::chrono::steady_clock::time_point stopTime; ///< Time of the measurement end.
};

#endif /* STOPWATCH_HPP_ */
