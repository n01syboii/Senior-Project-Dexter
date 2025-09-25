import ydlidar
import numpy as np
import time
import plotext as pltx
from Transbot_Lib import Transbot
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

bot = Transbot()  # Robot controller

# Lidar parameters
NUM_SCANS = 5  # Number of scans to average
angle_dif = 30  # +/- angle range from 180 deg
distance_max = 3  # Max detection distance

angle_min = 180 - angle_dif
angle_max = 180 + angle_dif

scan_buffer = []  # Stores recent scans

# Lidar port setup
ports = ydlidar.lidarPortList()
port = "/dev/ydlidar"
for key, value in ports.items():
    port = value

# Lidar configuration
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

scan = ydlidar.LaserScan()

def process_scan(scan):
    """Collects scan data and manages averaging buffer."""
    global scan_buffer

    scan_data = []
    for point in scan.points:
        angle_degrees = round(np.degrees(point.angle) % 360, 2)
        distance = round(point.range, 2)
        if angle_min <= angle_degrees <= angle_max and 0 < distance < distance_max:
            scan_data.append((angle_degrees, distance))

    scan_buffer.append(scan_data)
    if len(scan_buffer) > NUM_SCANS:
        scan_buffer.pop(0)
    if len(scan_buffer) == NUM_SCANS:
        process_averaged_scan()

def process_averaged_scan():
    """Averages points across scans and detects edges/corners."""
    angle_distance_map = defaultdict(list)
    for scan in scan_buffer:
        for angle, distance in scan:
            angle_distance_map[angle].append(distance)

    averaged_angles = []
    averaged_distances = []
    for angle, distances in angle_distance_map.items():
        avg_distance = np.mean(distances) if len(distances) > 1 else distances[0]
        averaged_angles.append(angle)
        averaged_distances.append(avg_distance)

    # Derivative-based edge detection
    first_derivative = np.gradient(averaged_distances, averaged_angles, edge_order=2)
    second_derivative = np.gradient(first_derivative, averaged_angles, edge_order=2)
    zero_crossings = np.where(np.diff(np.sign(second_derivative)))[0]
    edges = [(averaged_angles[i], averaged_distances[i]) for i in zero_crossings]

    # Corner detection (function must be defined elsewhere)
    corners = detect_corners(edges)
    plot_scan(averaged_angles, averaged_distances, edges, corners)

def plot_scan(angles, distances, edges, corners):
    """Plots the processed Lidar scan."""
    pltx.scatter(angles, distances, label="Distances")
    # To highlight edges/corners, uncomment below:
    # if edges:
    #     edge_angles, edge_distances = zip(*edges)
    #     pltx.scatter(edge_angles, edge_distances, color="red", label="Edges")
    # if corners:
    #     corner_angles, corner_distances = zip(*corners)
    #     pltx.scatter(corner_angles, corner_distances, color="red", label="Corners")
    pltx.xlim(angle_min, angle_max)
    pltx.ylim(0, 4)
    pltx.show()
    pltx.clt()
    pltx.cld()

# Example corner detection function (uncomment and implement as needed)
# def detect_corners(edges, angle_threshold=1):
#     """Identifies corners from detected edges by checking angle changes."""
#     corners = []
#     for i in range(1, len(edges) - 1):
#         angle1, _ = edges[i - 1]
#         angle2, _ = edges[i]
#         angle3, _ = edges[i + 1]
#         angle_change1 = abs(angle2 - angle1)
#         angle_change2 = abs(angle3 - angle2)
#         if angle_change1 > angle_threshold and angle_change2 > angle_threshold:
#             corners.append(edges[i])
#     return corners

# Main Lidar loop
if laser.initialize():
    if laser.turnOn():
        while True:
            if laser.doProcessSimple(scan):
                process_scan(scan)

# Lidar shutdown
laser.turnOff()
laser.disconnecting()
