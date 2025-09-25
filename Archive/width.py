#!/usr/bin/env python3
import rospy
import numpy as np
import plotext as pltx
from sensor_msgs.msg import LaserScan
import time

# Angle range for channel detection (degrees)
angle_dif = 50
angle_min = 180 - angle_dif
angle_max = 180 + angle_dif

scan_buffer = []
NUM_SCANS = 15  # Number of scans to average

def process_scan(scan):
    global scan_buffer

    # Add new scan to buffer
    scan_buffer.append(scan.ranges)

    # Keep only the last NUM_SCANS scans
    if len(scan_buffer) > NUM_SCANS:
        scan_buffer.pop(0)

    # Wait until buffer is full
    if len(scan_buffer) < NUM_SCANS:
        return

    # Average ranges across buffer
    avg_ranges = np.nanmean(scan_buffer, axis=0)

    # Process averaged scan
    process_averaged_scan(scan, avg_ranges)

def process_averaged_scan(scan, avg_ranges):
    angles = []
    distances = []
    angle_increment = scan.angle_increment

    # Convert scan data to angle/distance lists within region of interest
    for i, distance in enumerate(avg_ranges):
        angle = np.degrees(scan.angle_min + i * angle_increment) % 360

        if angle_min <= angle <= angle_max and np.isfinite(distance):
            angles.append(round(angle, 2))
            distances.append(round(distance, 2))

    corners = find_corners(angles, distances)

    # Sort corners by angle and select leftmost and rightmost
    corners = sorted(corners, key=lambda x: x[0])

    left_corner = corners[0]
    right_corner = corners[-1]

    # Visualization
    pltx.scatter(angles, distances)
    pltx.scatter([left_corner[0], right_corner[0]], [left_corner[1], right_corner[1]], color="red")
    pltx.xlim(angle_min, angle_max)
    pltx.ylim(0, 4)
    pltx.show()
    pltx.clt()
    pltx.cld()

# Alternative process_averaged_scan version (for reference)
# def process_averaged_scan(scan, avg_ranges):
#     angles = []
#     distances = []
#     angle_increment = scan.angle_increment
#     for i, distance in enumerate(avg_ranges):
#         angle = np.degrees(scan.angle_min + i * angle_increment) % 360
#         if angle_min <= angle <= angle_max and np.isfinite(distance):
#             angles.append(round(angle, 2))
#             distances.append(round(distance, 2))
#     corners = find_corners(angles, distances)
#     print(corners)
#     width = channel_width(corners)
#     left_corner, right_corner = corners
#     pltx.scatter([left_corner[0], right_corner[0]], [left_corner[1], right_corner[1]], color="red")
#     pltx.xlim(angle_min, angle_max)
#     pltx.ylim(0, 7)
#     pltx.show()
#     pltx.clt()
#     pltx.cld()
            
def find_corners(angles, ranges):
    # Detects corners based on range discontinuities
    corners = []
    threshold = 0.04
    for i in range(1, len(ranges) - 1):
        if ranges[i] <= 2:  # Only consider points within 2 meters
            if abs(ranges[i] - ranges[i-1]) > threshold and abs(ranges[i] - ranges[i+1]) > threshold:
                corners.append((angles[i], ranges[i]))
    return corners

def channel_width(left_corner, right_corner):
    # Computes width between two corners using law of cosines
    d1 = left_corner[1]
    d2 = right_corner[1]
    theta = np.radians(right_corner[0] - left_corner[0])
    width = np.sqrt(d1**2 + d2**2 - 2*d1*d2*np.cos(theta))
    if width >= 0.33:
        return width
    else:
        return None
    
def listener():
    # ROS node and subscriber setup
    rospy.init_node('lidar_width_processor', anonymous=True)
    rospy.Subscriber('/scan', LaserScan, process_scan)
    rospy.spin()

if __name__ == '__main__':
    listener()
