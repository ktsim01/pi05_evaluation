// Copyright 2022 Yifeng Zhu

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/gripper_state.h>
#include <yaml-cpp/yaml.h>

#include <spdlog/spdlog.h>

#include "utils/log_utils.h"
#include "utils/robot_utils.h"
#include "utils/zmq_utils.h"

#include "franka_controller.pb.h"
#include "franka_robot_state.pb.h"
// ... keep your existing includes and code above ...

#include <condition_variable>  // ADD THIS

int main(int argc, char **argv) {
  // Load configs
  YAML::Node config = YAML::LoadFile(argv[1]);

  double pub_rate = 40.;
  if (config["GRIPPER"]["PUB_RATE"]) {
    pub_rate = config["GRIPPER"]["PUB_RATE"].as<double>();
  }

  const std::string robot_ip = config["ROBOT"]["IP"].as<std::string>();

  // Subscribing gripper command
  const std::string subscriber_ip = config["PC"]["IP"].as<std::string>();
  const std::string sub_port =
      config["NUC"]["GRIPPER_SUB_PORT"].as<std::string>();

  // Publishing gripper command
  const std::string pub_port =
      config["NUC"]["GRIPPER_PUB_PORT"].as<std::string>();

  zmq_utils::ZMQPublisher zmq_pub(pub_port);
  zmq_utils::ZMQSubscriber zmq_sub(subscriber_ip, sub_port);

  // Initialize logger
  log_utils::initialize_logger(
      config["GRIPPER_LOGGER"]["CONSOLE"]["LOGGER_NAME"].as<std::string>(),
      config["GRIPPER_LOGGER"]["CONSOLE"]["LEVEL"].as<std::string>(),
      config["GRIPPER_LOGGER"]["CONSOLE"]["USE"].as<bool>(),
      config["GRIPPER_LOGGER"]["FILE"]["LOGGER_NAME"].as<std::string>(),
      config["GRIPPER_LOGGER"]["FILE"]["LEVEL"].as<std::string>(),
      config["GRIPPER_LOGGER"]["FILE"]["USE"].as<bool>());

  try {
    franka::Gripper gripper(robot_ip);

    std::atomic_bool running{true};
    bool executing = false;

    // Add mutex and condition_variable for synchronizing execution
    std::mutex exec_mutex;
    std::condition_variable exec_cv;

    robot_utils::FrankaGripperStateUtils gripper_state_utils;
    struct {
      std::mutex mutex;
      franka::GripperState state;
    } gripper_state{};

    struct {
      std::mutex mutex;
      FrankaGripperControlMessage control_msg;
    } gripper_cmd{};

    auto gripper_logger = log_utils::get_logger(
        config["GRIPPER_LOGGER"]["CONSOLE"]["LOGGER_NAME"].as<std::string>());

    // Logging thread IDs for debugging
    std::stringstream ss_pub, ss_sub;
    ss_pub << std::this_thread::get_id();
    ss_sub << std::this_thread::get_id();

    spdlog::info("[NUC] gripper_pub_thread starting on thread {}", ss_pub.str());
    spdlog::info("[NUC] gripper_sub_thread starting on thread {}", ss_sub.str());

    gripper_logger->info("Gripper state publisher: {0}Hz", pub_rate);

    // Publisher thread
    std::thread gripper_pub_thread([&]() {
      std::stringstream ss;
      ss << std::this_thread::get_id();
      spdlog::info("[NUC] gripper_pub_thread starting on thread {}", ss.str());
      franka::GripperState current_gripper_state;
      while (running) {
        {
          std::lock_guard<std::mutex> lock(gripper_state.mutex);
          gripper_state.state = gripper.readOnce();
          current_gripper_state = gripper_state.state;
        }
        FrankaGripperStateMessage gripper_state_msg;
        gripper_state_utils.LoadGripperStateToMsg(current_gripper_state,
                                                  gripper_state_msg);
        std::string serialized_gripper_state_msg;
        gripper_state_msg.SerializeToString(&serialized_gripper_state_msg);
        zmq_pub.send(serialized_gripper_state_msg);
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(1. / pub_rate * 1000)));
      }
    });

    // Subscriber thread
    std::thread gripper_sub_thread([&]() {
      std::stringstream ss;
      ss << std::this_thread::get_id();
      spdlog::info("[NUC] gripper_sub_thread starting on thread {}", ss.str());

      while (running) {
        std::string msg = zmq_sub.recv(false);
        FrankaGripperControlMessage control_msg;
        if (control_msg.ParseFromString(msg)) {
          // Log reception time
          auto now_recv = std::chrono::steady_clock::now();
          auto ms_recv = std::chrono::duration_cast<std::chrono::milliseconds>(
              now_recv.time_since_epoch())
                            .count();
          spdlog::info("[NUC] Received gripper command at {} ms", ms_recv);

          {
            std::lock_guard<std::mutex> lock(gripper_cmd.mutex);
            gripper_cmd.control_msg = control_msg;
          }

          {
            // Notify main loop to execute command
            std::lock_guard<std::mutex> lock(exec_mutex);
            executing = true;
          }
          exec_cv.notify_one();
        }

        auto gripper_control = control_msg.control_msg();
        FrankaGripperStopMessage stop_msg;
        if (gripper_control.UnpackTo(&stop_msg)) {
          gripper.stop();
        }

        if (control_msg.termination()) {
          running = false;

          // Notify main loop to exit wait
          exec_cv.notify_one();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });

    gripper.homing();
    gripper_logger->info("Gripper homing complete");
    bool has_grasped = false;

    // Main loop: wait on condition variable for new commands
    while (running) {
      std::unique_lock<std::mutex> lock(exec_mutex);
      exec_cv.wait(lock, [&]() { return executing || !running; });

      if (!running) break;

      lock.unlock();  // release lock early for command execution

      // Log execution time
      auto now_exec = std::chrono::steady_clock::now();
      auto ms_exec =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              now_exec.time_since_epoch())
              .count();
      spdlog::info("[NUC] Executing gripper command at {} ms", ms_exec);

      FrankaGripperStopMessage homing_msg;
      FrankaGripperMoveMessage move_msg;
      FrankaGripperGraspMessage grasp_msg;
      FrankaGripperStopMessage stop_msg;

      FrankaGripperControlMessage last_control_msg;
      {
        std::lock_guard<std::mutex> lock(gripper_cmd.mutex);
        last_control_msg = gripper_cmd.control_msg;
      }

      auto gripper_control = last_control_msg.control_msg();

      // Time actual gripper API call
      auto start = std::chrono::steady_clock::now();

      if (gripper_control.UnpackTo(&homing_msg)) {
        spdlog::info("[NUC] Gripper homing called");
        gripper.homing();
        spdlog::info("[NUC] Gripper homing done");
        has_grasped = false;

      } else if (gripper_control.UnpackTo(&move_msg)) {
        spdlog::info("[NUC] Gripper move called");
        gripper.move(move_msg.width(), move_msg.speed());
        spdlog::info("[NUC] Gripper move done");
        has_grasped = false;

      } else if (gripper_control.UnpackTo(&grasp_msg)) {
        if (has_grasped) {
          executing = false;
          continue;
        }
        double epsilon_inner, epsilon_outer;
        if (grasp_msg.epsilon_inner() == 0. && grasp_msg.epsilon_outer() == 0.) {
          epsilon_inner = 0.08;
          epsilon_outer = 0.08;
        } else {
          epsilon_inner = grasp_msg.epsilon_inner();
          epsilon_outer = grasp_msg.epsilon_outer();
        }

        double force;
        if (grasp_msg.force() == 0.) {
          force = 2.0;
        } else {
          force = grasp_msg.force();
        }

        spdlog::info("[NUC] Gripper grasp called");
        has_grasped = gripper.grasp(grasp_msg.width(), grasp_msg.speed(),
                                    force, epsilon_inner, epsilon_outer);
        spdlog::info("[NUC] Gripper grasp done");

        gripper_logger->info("Grasped? {0}", has_grasped);

      } else if (gripper_control.UnpackTo(&stop_msg)) {
        spdlog::info("[NUC] Gripper stop called");
        gripper.stop();
        spdlog::info("[NUC] Gripper stop done");
        has_grasped = false;

      } else {
        gripper_logger->warn("Unpack failed");
      }

      auto end = std::chrono::steady_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count();
      spdlog::info("[NUC] Gripper command execution time: {} ms", duration);

      {
        std::lock_guard<std::mutex> lock(exec_mutex);
        executing = false;
      }
    }

    gripper_sub_thread.join();
    gripper_pub_thread.join();

  } catch (franka::Exception const &e) {
    auto gripper_logger = log_utils::get_logger(
        config["GRIPPER_LOGGER"]["CONSOLE"]["LOGGER_NAME"].as<std::string>());
    gripper_logger->error(e.what());
    return -1;
  }

  return 0;
}