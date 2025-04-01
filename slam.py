import numpy as np
import ydlidar
from Transbot_Lib import Transbot

# Initialize robot controller
bot = Transbot()

# Lidar parameters
angle_dif = 50
angle_min = 180 - angle_dif
angle_max = 180 + angle_dif

# Find available ports and set Lidar port
ports = ydlidar.lidarPortList()
for key, value in ports.items():
    port = value

# Initialize and configure Lidar
laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 512000)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TOF)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
laser.setlidaropt(ydlidar.LidarPropSampleRate, 20)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)
laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
laser.setlidaropt(ydlidar.LidarPropMaxRange, 200.0)
laser.setlidaropt(ydlidar.LidarPropMinRange, 0.01)

# Laser scan object
scan = ydlidar.LaserScan()


def process_scan(scan):
    global scan_buffer

    angles = []
    distances = []

    for point in scan.points:
        angle_degrees = np.degrees(point.angle) % 360  # Convert radians to degrees
        distance = point.range  # Distance in meters

        if angle_min <= angle_degrees <= angle_max:
            angles.append(round(angle_degrees, 2))
            distances.append(round(distance, 2))


# Initialize Lidar and start processing
if laser.initialize():
    if laser.turnOn():
        while True:
            if laser.doProcessSimple(scan):
                process_scan(scan)

# Shutdown sequence
laser.turnOff()
laser.disconnecting()
