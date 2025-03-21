import ydlidar
import numpy as np
import time
import plotext as pltx
from Transbot_Lib import Transbot
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

# Initialize robot controller
bot = Transbot()

# Lidar parameters
angle_dif = 30
angle_min = 180 - angle_dif
angle_max = 180 + angle_dif

# Buffer for scan averaging
scan_buffer = []
NUM_SCANS = 1  # Number of scans to collect before averaging

# Find available ports and set Lidar port
ports = ydlidar.lidarPortList()
port = "/dev/ydlidar"
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
    """Collects scan data for a specific amount of rotations (NUM_SCANS)"""
    global scan_buffer

    scan_data = []
    
    for point in scan.points:
        angle_degrees = round(np.degrees(point.angle) % 360, 2)
        distance = round(point.range, 2)

        if angle_min <= angle_degrees <= angle_max:
            scan_data.append((angle_degrees, distance))

    scan_buffer.append(scan_data)

    # Maintain buffer size
    if len(scan_buffer) > NUM_SCANS:
        scan_buffer.pop(0)

    # Only process when we have collected enough scans
    if len(scan_buffer) == NUM_SCANS:
        process_averaged_scan()

def process_averaged_scan():
    """Averages points that appear across multiple scans and directly includes unique ones."""
    angle_distance_map = defaultdict(list)

    # Collect all angle-distance pairs from multiple scans
    for scan in scan_buffer:
        for angle, distance in scan:
            angle_distance_map[angle].append(distance)

    # Compute averaged distances for recurring angles
    averaged_angles = []
    averaged_distances = []
    for angle, distances in angle_distance_map.items():
        avg_distance = np.mean(distances) if len(distances) > 1 else distances[0]  # Average only recurring
        averaged_angles.append(angle)
        averaged_distances.append(avg_distance)

    # Apply Gaussian smoothing to reduce noise
    smoothed_distances = gaussian_filter1d(averaged_distances, sigma=2)

    # Plot results
    # plot_scan(averaged_angles, smoothed_distances)
    plot_scan(averaged_angles, averaged_distances)

def plot_scan(angles, distances):
    """Plots the processed Lidar scan."""
    pltx.scatter(angles, distances)
    pltx.xlim(angle_min, angle_max)
    pltx.ylim(0, 4)
    pltx.show()
    pltx.clt()
    pltx.cld()

# Initialize Lidar and start processing
if laser.initialize():
    if laser.turnOn():
        while True:
            if laser.doProcessSimple(scan):
                process_scan(scan)

# Shutdown sequence
laser.turnOff()
laser.disconnecting()
