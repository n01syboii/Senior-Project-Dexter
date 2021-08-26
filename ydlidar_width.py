import ydlidar
import numpy as np
import time
import plotext as pltx
from Transbot_Lib import Transbot

# Initialize robot controller
bot = Transbot()

# Lidar parameters
angle_dif = 50
angle_min = 180 - angle_dif
angle_max = 180 + angle_dif

# Buffer for scan averaging
scan_buffer = []
NUM_SCANS = 1  # Number of scans to average
MAX_SCAN_POINTS = 2000  # Ensure scans are padded to this length

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
    """ Processes a single Lidar scan and stores it for averaging. """
    global scan_buffer

    angles = []
    distances = []

    for point in scan.points:
        angle_degrees = np.degrees(point.angle) % 360  # Convert radians to degrees
        distance = point.range  # Distance in meters

        if angle_min <= angle_degrees <= angle_max:
            angles.append(round(angle_degrees, 2))
            distances.append(round(distance, 2))

    # Ensure all scans have the same length by padding with NaN
    while len(distances) < MAX_SCAN_POINTS:
        distances.append(np.nan)

    # Store scan in buffer
    scan_buffer.append(distances)

    # Maintain buffer size
    if len(scan_buffer) > NUM_SCANS:
        scan_buffer.pop(0)

    # Only process once enough scans are collected
    if len(scan_buffer) < NUM_SCANS:
        return

    # Convert scan buffer to a NumPy array and compute averaged scan
    scan_buffer_np = np.array(scan_buffer, dtype=np.float32)
    avg_ranges = np.nanmean(scan_buffer_np, axis=0)  # Compute mean ignoring NaNs


    # # Apply Gaussian smoothing to remove noise
    # smoothed_ranges = gaussian_filter1d(avg_ranges, sigma=2)

    # Process the averaged scan (without smoothing)
    process_averaged_scan(angles, avg_ranges)


def process_averaged_scan(angles, distances):
    """ Processes the averaged scan to detect corners and compute channel width. """
    corners = find_corners(angles, distances)

    # Sort corners by angle and select the leftmost and rightmost points
    corners = sorted(corners, key=lambda x: x[0])
    left_corner, right_corner = corners[0], corners[-1]

    # Compute the channel width
    width = channel_width(left_corner, right_corner)

    # # Print width if it's valid
    # if width:
    #     print(f"Channel width: {width:.2f} meters")
    # else:
    #     print("Channel too narrow (less than 0.33m), ignoring measurement.")

    # Plot Lidar data and detected corners
    pltx.scatter(angles, distances)
    pltx.scatter([left_corner[0], right_corner[0]], [left_corner[1], right_corner[1]], color="red")

    pltx.xlim(angle_min, angle_max)
    pltx.ylim(0, 4)
    pltx.show()

    pltx.clt()
    pltx.cld()


def find_corners(angles, ranges):
    """ Identifies corner points based on sudden depth changes. """
    corners = []
    threshold = 0.04  # Change threshold for detecting edges

    for i in range(1, len(ranges) - 1):
        if ranges[i] <= 2:  # Ignore points beyond 2m
            if abs(ranges[i] - ranges[i - 1]) > threshold and abs(ranges[i] - ranges[i + 1]) > threshold:
                corners.append((angles[i], ranges[i]))

    return corners


def channel_width(left_corner, right_corner):
    """ Computes the channel width using the chord equation. """
    d1 = left_corner[1]  # Distance of left corner
    d2 = right_corner[1]  # Distance of right corner

    theta = np.radians(right_corner[0] - left_corner[0])  # Angle difference in radians
    width = np.sqrt(d1 ** 2 + d2 ** 2 - 2 * d1 * d2 * np.cos(theta))  # Law of cosines

    return width if width >= 0.33 else None  # Ignore widths < 0.33m


# Initialize Lidar and start processing
if laser.initialize():
    if laser.turnOn():
        while True:
            if laser.doProcessSimple(scan):
                process_scan(scan)

# Shutdown sequence
laser.turnOff()
laser.disconnecting()
