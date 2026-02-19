from pymodbus.client.sync import ModbusSerialClient
from pymodbus.exceptions import ModbusIOException
from math import ceil
import logging

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)


class communication:
    def __init__(self):
        self.client = None

    def connectToDevice(self, device):
        self.client = ModbusSerialClient(
            method="rtu",
            port=device,
            stopbits=1,
            bytesize=8,
            baudrate=115200,
            timeout=0.5,
        )
        if not self.client.connect():
            print(f"❌ Unable to connect to {device}")
            return False
        print(f"✅ Connected to {device}")
        return True

    def disconnectFromDevice(self):
        if self.client:
            self.client.close()
            print("🔌 Disconnected")

    def getStatus(self, numBytes):
        numRegs = int(ceil(numBytes / 2.0))
        try:
            response = self.client.read_holding_registers(0, numRegs, unit=9)
        except Exception as e:
            print("❌ Read failed:", e)
            return None

        if response is None or isinstance(response, ModbusIOException):
            print("❌ No response or ModbusIOException")
            return None

        output = []
        for i in range(numRegs):
            reg = response.getRegister(i)
            output.append((reg & 0xFF00) >> 8)
            output.append(reg & 0x00FF)

        return output


# === MAIN TEST ===
if __name__ == "__main__":
    device_path = "/dev/ttyUSB1"  # Change this if needed
    comm = communication()

    if comm.connectToDevice(device_path):
        status = comm.getStatus(6)  # Number of bytes to read (adjust as needed)
        if status:
            print("📊 Gripper status:", status)
        else:
            print("⚠️ Failed to get status.")
        comm.disconnectFromDevice()
