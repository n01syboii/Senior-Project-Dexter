#!/usr/bin/env python3
import rospy
import numpy as np
import plotext as pltx
from sensor_msgs.msg import LaserScan
import time

angle_dif = 50
angle_min = 180 - angle_dif
angle_max = 180 + angle_dif

scan_buffer = []
NUM_SCANS = 10

def process_scan(scan):
    global scan_buffer

    scan_buffer.append(scan.ranges)

    # Keep only the last 10 scans
    if len(scan_buffer) > NUM_SCANS:
        scan_buffer.pop(0)

    # Only process if we have 10 full scans
    if len(scan_buffer) < NUM_SCANS:
        return

    # Compute average ranges (averaging each angle across 10 scans)
    avg_ranges = np.nanmean(scan_buffer, axis=0)

    process_averaged_scan(scan, avg_ranges)

def process_averaged_scan(scan, avg_ranges):
    angles = []
    distances = []
    angle_increment = scan.angle_increment

    for i, distance in enumerate(avg_ranges):
        angle = np.degrees(scan.angle_min + i * angle_increment) % 360

        if angle_min <= angle <= angle_max and np.isfinite(distance):
            angles.append(round(angle, 2))
            distances.append(round(distance, 2))

    corners = find_corners(angles, distances)

    if len(corners) < 2:
        rospy.logwarn("Not enough corners found, skipping this scan.")
        return

    # Sort corners by angle and take leftmost + rightmost
    corners = sorted(corners, key=lambda x: x[0])

    left_corner = corners[0]
    right_corner = corners[-1]

    print(f"Left Corner: {left_corner}, Right Corner: {right_corner}")

    width = channel_width(left_corner, right_corner)
    if width:
        print(f"Channel width: {width:.2f} meters")

    # # Plot for visualization
    # pltx.scatter(angles, distances)
    # pltx.scatter([left_corner[0], right_corner[0]], [left_corner[1], right_corner[1]], color="red")

    # pltx.xlim(angle_min, angle_max)
    # pltx.ylim(0, 4)
    # pltx.show()

    # pltx.clt()
    # pltx.cld()


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
    
#     # if len(corners) == 2:
#     #     print(corners)
        
#     width = channel_width(corners)
#         # if width is not None:
#         #     print(f"Channel width: {width:.2f} meters")
#         # else:
#         #     print("Channel too narrow (less than 0.33m), ignoring measurement.")


#     left_corner, right_corner = corners

#     # pltx.scatter(angles, distances)
#     pltx.scatter([left_corner[0], right_corner[0]], [left_corner[1], right_corner[1]], color="red")

#     pltx.xlim(angle_min, angle_max)
#     pltx.ylim(0, 7)
#     pltx.show()

#     pltx.clt()
#     pltx.cld()
            
def find_corners(angles, ranges):
    corners = []
    threshold = 0.04

    for i in range(1, len(ranges) - 1):
        if ranges[i] <= 2: #limiting range to 2m 
            if abs(ranges[i] - ranges[i-1]) > threshold and abs(ranges[i] - ranges[i+1]) > threshold:
                corners.append((angles[i], ranges[i]))

    return corners

def channel_width(left_corner, right_corner):
    d1 = left_corner[1]
    d2 = right_corner[1]
    
    theta = np.radians(right_corner[0] - left_corner[0])
    width = np.sqrt(d1**2 + d2**2 - 2*d1*d2*np.cos(theta))
    
    if width >= 0.33:
        return width
    else:
        return None


def listener():
    rospy.init_node('lidar_width_processor', anonymous=True)
    rospy.Subscriber('/scan', LaserScan, process_scan)
    rospy.spin()

if __name__ == '__main__':
    listener()
