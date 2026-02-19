// Copyright 2022 Yifeng Zhu

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <condition_variable>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// Remove Franka includes:
// #include <franka/exception.h>
// #include <franka/gripper.h>
// #include <franka/gripper_state.h>

// Add Modbus include
#include <modbus/modbus.h>

#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>

#include "utils/log_utils.h"
#include "utils/robot_utils.h"
#include "utils/zmq_utils.h"

#include "franka_controller.pb.h"
#include "franka_robot_state.pb.h"

// Robotiq Gripper Class
class RobotiqGripper {
private:
    modbus_t* ctx;
    bool connected;
    
    // Modbus register addresses (based on Robotiq documentation)
    static const int COMMAND_REGISTER = 0x03E8;  // 1000 decimal
    static const int STATUS_REGISTER = 0x07D0;   // 2000 decimal
    static const int MODBUS_UNIT_ID = 9;
    
public:
    struct GripperState {
        bool activated;
        bool in_motion;
        int motion_status;      // 0=stopped, 1=opening, 2=closing, 3=stopped
        int object_detection;   // 0=moving, 1=contact opening, 2=contact closing, 3=at position
        uint8_t position;       // 0=open, 255=closed
        uint8_t current;        // Motor current
        uint8_t fault;          // Fault status
        double width;           // Converted to meters for compatibility
        bool is_grasped;        // Derived from object_detection
        double temperature;     // Not available on Robotiq, set to 0
    };
    
    RobotiqGripper(const std::string& device_path, int baud_rate = 115200) 
        : ctx(nullptr), connected(false) {
        // Create RTU context
        ctx = modbus_new_rtu(device_path.c_str(), baud_rate, 'N', 8, 1);
        if (ctx == nullptr) {
            throw std::runtime_error("Failed to create Modbus RTU context");
        }
        
        // Set slave ID
        modbus_set_slave(ctx, MODBUS_UNIT_ID);
        
        // Set timeout
        modbus_set_response_timeout(ctx, 0, 500000); // 500ms
        modbus_set_byte_timeout(ctx, 0, 500000);
    }
    
    ~RobotiqGripper() {
        disconnect();
        if (ctx) {
            modbus_free(ctx);
        }
    }
    
    bool connect() {
        if (modbus_connect(ctx) == -1) {
            std::cerr << "❌ Connection failed: " << modbus_strerror(errno) << std::endl;
            return false;
        }
        connected = true;
        std::cout << "✅ Connected to Robotiq gripper" << std::endl;
        return true;
    }
    
    void disconnect() {
        if (connected && ctx) {
            modbus_close(ctx);
            connected = false;
            std::cout << "🔌 Disconnected from gripper" << std::endl;
        }
    }
    
    bool sendCommand(uint8_t action_request, uint8_t position, uint8_t speed, uint8_t force) {
        if (!connected) return false;
        
        // Prepare registers as in Python code
        uint16_t reg1 = (action_request << 8) | 0x00;
        uint16_t reg2 = (0x00 << 8) | position;
        uint16_t reg3 = (speed << 8) | force;
        
        uint16_t registers[3] = {reg1, reg2, reg3};
        
        int result = modbus_write_registers(ctx, COMMAND_REGISTER, 3, registers);
        if (result == -1) {
            std::cerr << "❌ Write failed: " << modbus_strerror(errno) << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool reset() {
        spdlog::info("[ROBOTIQ] Resetting gripper...");
        return sendCommand(0x00, 0, 0, 0);
    }
    
    bool activate() {
        spdlog::info("[ROBOTIQ] Activating gripper...");
        if (reset()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return sendCommand(0x01, 0, 255, 255);
        }
        return false;
    }
    
    bool move(uint8_t position, uint8_t speed = 255, uint8_t force = 255) {
        if (!connected) return false;
        // Set go-to bit (0x08) + activate bit (0x01) = 0x09
        return sendCommand(0x09, position, speed, force);
    }
    
    bool stop() {
        spdlog::info("[ROBOTIQ] Stopping gripper...");
        return sendCommand(0x00, 0, 0, 0);
    }
    
    GripperState readOnce() {
        GripperState state = {};
        
        if (!connected) return state;
        
        uint16_t registers[3];
        int result = modbus_read_registers(ctx, STATUS_REGISTER, 3, registers);
        
        if (result == -1) {
            spdlog::error("[ROBOTIQ] Status read failed: {}", modbus_strerror(errno));
            return state;
        }
        
        // Decode status (same logic as Python)
        uint8_t gripper_status = (registers[0] & 0xFF00) >> 8;
        uint8_t fault_status = registers[0] & 0x00FF;
        uint8_t position_request_echo = (registers[1] & 0xFF00) >> 8;
        uint8_t position = registers[1] & 0x00FF;
        uint8_t current = (registers[2] & 0xFF00) >> 8;
        
        // Decode bits
        state.activated = (gripper_status & 0x01) != 0;
        state.in_motion = (gripper_status & 0x08) != 0;
        state.motion_status = (gripper_status & 0x30) >> 4;
        state.object_detection = (gripper_status & 0xC0) >> 6;
        state.position = position;
        state.current = current;
        state.fault = fault_status;
        
        // Convert to Franka-compatible values
        state.width = convertPositionToWidth(position);
        state.is_grasped = (state.object_detection == 1 || state.object_detection == 2);
        state.temperature = 0.0; // Not available on Robotiq
        
        return state;
    }
    
    bool isMoving() {
        auto state = readOnce();
        return state.in_motion;
    }
    
    bool isObjectDetected() {
        auto state = readOnce();
        return state.is_grasped;
    }
    
    // Franka-compatible API methods
    bool homing() {
        spdlog::info("[ROBOTIQ] Homing (activating) gripper...");
        bool result = activate();
        if (result) {
            // Wait for activation to complete
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        }
        return result;
    }
    
    bool move(double width, double speed) {
        spdlog::info("[ROBOTIQ] Moving to width: {} m, speed: {} m/s", width, speed);
        uint8_t position = convertWidthToPosition(width);
        uint8_t robotiq_speed = convertSpeedToRobotiq(speed);
        
        return move(position, robotiq_speed, 255); // Default force
    }
    
    bool grasp(double width, double speed, double force, double epsilon_inner, double epsilon_outer) {
        spdlog::info("[ROBOTIQ] Grasping at width: {} m, speed: {} m/s, force: {} N", width, speed, force);
        
        uint8_t position = convertWidthToPosition(width);
        uint8_t robotiq_speed = convertSpeedToRobotiq(speed);
        uint8_t robotiq_force = convertForceToRobotiq(force);
        
        bool success = move(position, robotiq_speed, robotiq_force);
        
        if (success) {
            // Wait for movement to complete
            auto start_time = std::chrono::steady_clock::now();
            const auto timeout = std::chrono::seconds(10); // 10 second timeout
            
            while (isMoving()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                
                // Check for timeout
                if (std::chrono::steady_clock::now() - start_time > timeout) {
                    spdlog::warn("[ROBOTIQ] Grasp operation timed out");
                    break;
                }
            }
            
            return isObjectDetected();
        }
        return false;
    }
    
private:
    // Conversion functions
    uint8_t convertWidthToPosition(double width_m) {
        // Robotiq 2F-85: 0-0.085m range
        const double MAX_WIDTH = 0.085;
        double normalized = std::max(0.0, std::min(1.0, width_m / MAX_WIDTH));
        return static_cast<uint8_t>((1.0 - normalized) * 255); // 0=open, 255=closed
    }
    
    double convertPositionToWidth(uint8_t position) {
        // Convert Robotiq position back to width in meters
        const double MAX_WIDTH = 0.085;
        return (255.0 - static_cast<double>(position)) / 255.0 * MAX_WIDTH;
    }
    
    uint8_t convertSpeedToRobotiq(double speed_ms) {
        const double MAX_SPEED = 0.15; // m/s
        double normalized = std::max(0.0, std::min(1.0, speed_ms / MAX_SPEED));
        return static_cast<uint8_t>(normalized * 255);
    }
    
    uint8_t convertForceToRobotiq(double force_n) {
        const double MAX_FORCE = 235.0; // N for 2F-85
        if (force_n == 0.0) force_n = 20.0; // Default
        double normalized = std::max(0.0, std::min(1.0, force_n / MAX_FORCE));
        return static_cast<uint8_t>(normalized * 255);
    }
};

// Helper class to replace robot_utils::FrankaGripperStateUtils
class RobotiqGripperStateUtils {
public:
    void LoadGripperStateToMsg(const RobotiqGripper::GripperState& state, 
                              FrankaGripperStateMessage& msg) {
        // Map Robotiq state to your existing message structure
        // Adjust these fields based on your actual protobuf message definition
        
        // Example mapping - modify based on your actual message structure:
        msg.set_width(state.width);
        msg.set_is_grasped(state.is_grasped);
        msg.set_temperature(state.temperature);
        msg.set_time(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
        
        // Add other fields as needed based on your message structure
        // For example:
        // msg.set_max_width(0.085);  // Robotiq 2F-85 max width
        // msg.set_speed(0.0);        // Current speed not available from Robotiq
        // msg.set_force(static_cast<double>(state.current) / 255.0 * 100.0); // Rough force estimate
    }
};

int main(int argc, char **argv) {
    // Load configs
    YAML::Node config = YAML::LoadFile(argv[1]);

    double pub_rate = 40.;
    if (config["GRIPPER"]["PUB_RATE"]) {
        pub_rate = config["GRIPPER"]["PUB_RATE"].as<double>();
    }

    // Get Robotiq connection parameters
    const std::string gripper_device = config["GRIPPER"]["DEVICE"].as<std::string>();
    int baud_rate = 115200;
    if (config["GRIPPER"]["BAUD_RATE"]) {
        baud_rate = config["GRIPPER"]["BAUD_RATE"].as<int>();
    }

    // Subscribing gripper command
    const std::string subscriber_ip = config["PC"]["IP"].as<std::string>();
    const std::string sub_port = config["NUC"]["GRIPPER_SUB_PORT"].as<std::string>();

    // Publishing gripper command
    const std::string pub_port = config["NUC"]["GRIPPER_PUB_PORT"].as<std::string>();

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
        // Initialize Robotiq gripper
        RobotiqGripper gripper(gripper_device, baud_rate);
        
        if (!gripper.connect()) {
            throw std::runtime_error("Failed to connect to Robotiq gripper");
        }

        std::atomic_bool running{true};
        bool executing = false;

        // Add mutex and condition_variable for synchronizing execution
        std::mutex exec_mutex;
        std::condition_variable exec_cv;

        RobotiqGripperStateUtils gripper_state_utils;
        struct {
            std::mutex mutex;
            RobotiqGripper::GripperState state;
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
            RobotiqGripper::GripperState current_gripper_state;
            while (running) {
                {
                    std::lock_guard<std::mutex> lock(gripper_state.mutex);
                    gripper_state.state = gripper.readOnce();
                    current_gripper_state = gripper_state.state;
                }
                FrankaGripperStateMessage gripper_state_msg;
                gripper_state_utils.LoadGripperStateToMsg(current_gripper_state, gripper_state_msg);
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
                        now_recv.time_since_epoch()).count();
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
            auto ms_exec = std::chrono::duration_cast<std::chrono::milliseconds>(
                now_exec.time_since_epoch()).count();
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
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            spdlog::info("[NUC] Gripper command execution time: {} ms", duration);

            {
                std::lock_guard<std::mutex> lock(exec_mutex);
                executing = false;
            }
        }

        gripper_sub_thread.join();
        gripper_pub_thread.join();

    } catch (const std::exception& e) {
        auto gripper_logger = log_utils::get_logger(
            config["GRIPPER_LOGGER"]["CONSOLE"]["LOGGER_NAME"].as<std::string>());
        gripper_logger->error(e.what());
        return -1;
    }

    return 0;
}