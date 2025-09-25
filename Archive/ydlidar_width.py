import ydlidar
import numpy as np
import time
import plotext as pltx
from Transbot_Lib import Transbot

bot = Transbot()  # Robot controller

# Lidar scan parameters
angle_dif = 50
angle_min = 180 - angle_dif
angle_max = 180 + angle_dif

scan_buffer = []
NUM_SCANS = 1  # Number of scans to average
MAX_SCAN_POINTS = 2000  # Pad scans to this length

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
    """Collects and buffers a single Lidar scan for averaging."""
    global scan_buffer

    angles = []
    distances = []

    for point in scan.points:
        angle_degrees = np.degrees(point.angle) % 360
        distance = point.range

        if angle_min <= angle_degrees <= angle_max:
            angles.append(round(angle_degrees, 2))
            distances.append(round(distance, 2))

    # Pad scan to fixed length
    while len(distances) < MAX_SCAN_POINTS:
        distances.append(np.nan)

    scan_buffer.append(distances)

    # Keep buffer at NUM_SCANS size
    if len(scan_buffer) > NUM_SCANS:
        scan_buffer.pop(0)

    if len(scan_buffer) < NUM_SCANS:
        return

    scan_buffer_np = np.array(scan_buffer, dtype=np.float32)
    avg_ranges = np.nanmean(scan_buffer_np, axis=0)

    # Optionally: apply smoothing here
    # smoothed_ranges = gaussian_filter1d(avg_ranges, sigma=2)

    process_averaged_scan(angles, avg_ranges)

def process_averaged_scan(angles, distances):
    """Detects corners and computes channel width from averaged scan."""
    corners = find_corners(angles, distances)

    # Use leftmost and rightmost corners
    corners = sorted(corners, key=lambda x: x[0])
    left_corner, right_corner = corners[0], corners[-1]

    width = channel_width(left_corner, right_corner)

    # Plot scan and detected corners
    pltx.scatter(angles, distances)
    pltx.scatter([left_corner[0], right_corner[0]], [left_corner[1], right_corner[1]], color="red")
    pltx.xlim(angle_min, angle_max)
    pltx.ylim(0, 4)
    pltx.show()
    pltx.clt()
    pltx.cld()

def find_corners(angles, ranges):
    """Detects corners based on sudden depth changes."""
    corners = []
    threshold = 0.04  # Edge detection threshold

    for i in range(1, len(ranges) - 1):
        if ranges[i] <= 2:
            if abs(ranges[i] - ranges[i - 1]) > threshold and abs(ranges[i] - ranges[i + 1]) > threshold:
                corners.append((angles[i], ranges[i]))
    return corners

def channel_width(left_corner, right_corner):
    """Computes channel width using law of cosines."""
    d1 = left_corner[1]
    d2 = right_corner[1]
    theta = np.radians(right_corner[0] - left_corner[0])
    width = np.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * np.cos(theta))
    return width if width >= 0.33 else None  # Ignore widths < 0.33m

# Main Lidar loop
if laser.initialize():
    if laser.turnOn():
        while True:
            if laser.doProcessSimple(scan):
                process_scan(scan)

# Lidar shutdown
laser.turnOff()
laser.disconnecting()
