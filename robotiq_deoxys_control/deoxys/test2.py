import time
from pymodbus.client.sync import ModbusSerialClient
from pymodbus.exceptions import ModbusIOException
from math import ceil
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class RobotiqGripperCommunication:
    def __init__(self):
        self.client = None
        
    def connectToDevice(self, device):
        """Connection to the client - matches official Robotiq implementation"""
        self.client = ModbusSerialClient(
            method="rtu",
            port=device,
            stopbits=1,
            bytesize=8,
            baudrate=115200,
            timeout=0.5,  # Using official timeout
        )
        if not self.client.connect():
            print(f"❌ Unable to connect to {device}")
            return False
        print(f"✅ Connected to {device}")
        return True
    
    def disconnectFromDevice(self):
        """Close connection"""
        if self.client:
            self.client.close()
            print("🔌 Disconnected")
    
    def sendCommand(self, data):
        """
        Send a command to the Gripper - Official Robotiq method
        Takes a list of uint8 bytes as argument
        """
        # make sure data has an even number of elements
        if len(data) % 2 == 1:
            data.append(0)

        # Initiate message as an empty list
        message = []

        # Fill message by combining two bytes in one register
        for i in range(0, len(data) // 2):
            message.append((data[2 * i] << 8) + data[2 * i + 1])

        try:
            response = self.client.write_registers(0x03E8, message, unit=0x0009)
            if response.isError():
                print("❌ Modbus error during write:", response)
                return False
        except Exception as e:
            print("❌ Modbus write operation failure:", e)
            return False
        
        print(f"✅ Command sent: {[hex(b) for b in data]}")
        return True

    def getStatus(self, numBytes):
        """
        Official Robotiq status reading method
        Gets the number of bytes to read as an argument
        """
        numRegs = int(ceil(numBytes / 2.0))

        try:
            response = self.client.read_holding_registers(0x07D0, numRegs, unit=0x0009)
        except Exception as e:
            print("❌ Status read failed:", e)
            return None

        if response is None:
            print("❌ Failed to receive status")
            return None
        elif isinstance(response, ModbusIOException):
            return None

        # Instantiate output as an empty list
        output = []

        # Fill the output with the bytes in the appropriate order
        for i in range(0, numRegs):
            output.append((response.getRegister(i) & 0xFF00) >> 8)
            output.append(response.getRegister(i) & 0x00FF)

        return output
    
    def resetGripper(self):
        """Reset the gripper using official byte format"""
        print("🔄 Resetting gripper...")
        # Reset command: all zeros
        data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        return self.sendCommand(data)
    
    def activateGripper(self):
        """Activate the gripper using official byte format"""
        print("🔄 Activating gripper...")
        # Activation command: rACT=1, others=0
        data = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00]
        return self.sendCommand(data)
    
    def moveGripper(self, position=255, speed=255, force=255):
        """
        Move gripper to position using official byte format
        Args:
            position: 0-255 (0=open, 255=closed)
            speed: 0-255 
            force: 0-255
        """
        print(f"🤖 Moving gripper - Pos:{position}, Speed:{speed}, Force:{force}")
        # Movement command: rACT=1, rGTO=1 (go to position)
        action_byte = 0x09  # rACT (bit 0) + rGTO (bit 3) = 0x01 + 0x08 = 0x09
        data = [action_byte, 0x00, 0x00, position, speed, force]
        return self.sendCommand(data)
    
    def closeGripper(self, position=255, speed=255, force=255):
        """Close the gripper"""
        print(f"🤏 Closing gripper to position {position}...")
        return self.moveGripper(position, speed, force)
    
    def openGripper(self, speed=255, force=255):
        """Open the gripper"""
        print("✋ Opening gripper...")
        return self.moveGripper(0, speed, force)
    
    def stopGripper(self):
        """Stop gripper motion"""
        print("⏹️ Stopping gripper...")
        # Stop command: rACT=1, rGTO=0
        data = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00]
        return self.sendCommand(data)

    def getGripperStatus(self):
        """Read and decode gripper status using official method"""
        try:
            status_bytes = self.getStatus(6)  # Read 6 bytes
            
            if status_bytes is None or len(status_bytes) < 6:
                print("❌ Failed to get status")
                return None
            
            # Decode status bytes (official format)
            gripper_status = status_bytes[0]    # Byte 0: Gripper Status
            fault_status = status_bytes[2]      # Byte 2: Fault Status  
            position_request_echo = status_bytes[3]  # Byte 3: Position Request Echo
            position = status_bytes[4]          # Byte 4: Position
            current = status_bytes[5]           # Byte 5: Current
            
            # Decode gripper status bits
            gACT = (gripper_status & 0x01) != 0  # Activation status
            gGTO = (gripper_status & 0x08) != 0  # Go to position
            gSTA = (gripper_status & 0x30) >> 4  # Motion status
            gOBJ = (gripper_status & 0xC0) >> 6  # Object detection
            
            print(f"📊 Gripper Status:")
            print(f"   Activated (gACT): {gACT}")
            print(f"   Go to position (gGTO): {gGTO}")
            print(f"   Motion status (gSTA): {gSTA}")
            print(f"     0=stopped, 1=opening, 2=closing, 3=stopped")
            print(f"   Object detection (gOBJ): {gOBJ}")
            print(f"     0=moving, 1=contact opening, 2=contact closing, 3=at position")
            print(f"   Position: {position}/255")
            print(f"   Position request echo: {position_request_echo}/255")
            print(f"   Current: {current}")
            print(f"   Fault: 0x{fault_status:02X}")
            
            if fault_status != 0:
                fault_messages = {
                    0x05: "Action delayed; activation must be completed first",
                    0x07: "Activation bit must be set prior to action",
                    0x08: "Maximum temperature exceeded",
                    0x09: "No communication for at least 1 second",
                    0x0A: "Under minimum operating voltage",
                    0x0B: "Automatic release in progress"
                }
                if fault_status in fault_messages:
                    print(f"   ⚠️ Fault: {fault_messages[fault_status]}")
                else:
                    print(f"   ⚠️ Unknown fault code: 0x{fault_status:02X}")
            
            return {
                'activated': gACT,
                'go_to_position': gGTO,
                'motion_status': gSTA,
                'object_detection': gOBJ,
                'position': position,
                'position_request_echo': position_request_echo,
                'current': current,
                'fault': fault_status,
                'raw_bytes': status_bytes
            }
            
        except Exception as e:
            print("❌ Status read failed:", e)
            return None

# === MAIN TEST ===
if __name__ == "__main__":
    device_path = "/dev/ttyUSB1"  # Change if needed
    gripper = RobotiqGripperCommunication()
    
    if gripper.connectToDevice(device_path):
        try:
            print("=== Robotiq Gripper Test (Official Protocol) ===")
            
            print("\nStep 1: Reading initial status...")
            initial_status = gripper.getGripperStatus()
            
            print("\nStep 2: Resetting gripper...")
            if gripper.resetGripper():
                time.sleep(1)
                
                print("\nStep 3: Activating gripper...")
                if gripper.activateGripper():
                    time.sleep(2)  # Wait for activation
                    
                    print("\nStep 4: Checking activation status...")
                    status = gripper.getGripperStatus()
                    
                    if status and status['activated']:
                        print("\n✅ Gripper activated successfully!")
                        
                        print("\nStep 5: Testing movement...")
                        # Close gripper
                        if gripper.closeGripper(position=200, speed=150, force=150):
                            time.sleep(3)
                            
                            print("\nStep 6: Checking closed position...")
                            gripper.getGripperStatus()
                            
                            # Open gripper
                            print("\nStep 7: Opening gripper...")
                            gripper.openGripper(speed=150, force=100)
                            time.sleep(3)
                            
                            print("\nStep 8: Final status check...")
                            gripper.getGripperStatus()
                    else:
                        print("❌ Gripper failed to activate properly")
                else:
                    print("❌ Failed to send activation command")
            else:
                print("❌ Failed to send reset command")
                
        except KeyboardInterrupt:
            print("\n⚠️ Operation interrupted")
        finally:
            gripper.disconnectFromDevice()
    else:
        print("Failed to connect to gripper")
