from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Lidar scan and edge detection parameters
num_scans = 1
angle_dif = 30
angle_min = 180 - angle_dif
angle_max = 180 + angle_dif

LOG_SIGMA = 0.5  # Gaussian smoothing for LoG
LOG_THRESHOLD = 0.5  # LoG edge detection threshold

NUM_WIDTH_SAMPLES = 10  # Number of width measurements to average
width_measurements = []  # Stores width measurements for averaging
running = True  # Main loop control flag


def get_scan_data(scan, distance_max=10):
    # Collects and averages scan data within angle and distance limits
    scan_buffer = []

    for _ in range(num_scans):
        temp_data = []
        for point in scan.points:
            angle = round(np.degrees(point.angle) % 360, 2)
            distance = round(point.range, 2)
            if angle_min <= angle <= angle_max and 0 < distance < distance_max:
                temp_data.append((angle, distance))
        scan_buffer.append(temp_data)

    angle_map = defaultdict(list)
    for scan in scan_buffer:
        for angle, distance in scan:
            angle_map[angle].append(distance)

    averaged_angles = []
    averaged_distances = []
    for angle, dists in sorted(angle_map.items()):
        averaged_angles.append(angle)
        averaged_distances.append(np.mean(dists))

    return averaged_angles, averaged_distances


def compute_log_edge_detection(angles, distances, sigma=1.1, threshold=0.05):
    # LoG edge detection: smooth, differentiate, find zero crossings
    sorted_indices = np.argsort(angles)
    angles = np.array(angles)[sorted_indices]
    distances = np.array(distances)[sorted_indices]

    smoothed_distances = gaussian_filter1d(distances, sigma=sigma)

    first_derivative = np.gradient(smoothed_distances, angles, edge_order=2)
    second_derivative = np.gradient(first_derivative, angles, edge_order=2)

    zero_crossings = []
    for i in range(len(second_derivative) - 1):
        if second_derivative[i] * second_derivative[i + 1] < 0:
            edge_strength = abs(second_derivative[i] - second_derivative[i + 1])
            if edge_strength > threshold:
                if second_derivative[i] != second_derivative[i + 1]:
                    t = -second_derivative[i] / (
                        second_derivative[i + 1] - second_derivative[i]
                    )
                    crossing_angle = angles[i] + t * (angles[i + 1] - angles[i])
                    distance_jump = abs(distances[i] - distances[i + 1])
                    if distance_jump > 1.0:
                        if distances[i] < distances[i + 1]:
                            raw_crossing_distance = distances[i]
                        else:
                            raw_crossing_distance = distances[i + 1]
                    else:
                        raw_crossing_distance = min(distances[i], distances[i + 1])
                    zero_crossings.append(
                        (crossing_angle, raw_crossing_distance, edge_strength)
                    )

    return angles, second_derivative, zero_crossings


def channel_width(left_edge, right_edge):
    # Computes channel width using chord formula
    left_angle, left_distance, _ = left_edge
    right_angle, right_distance, _ = right_edge

    left_angle_rad = np.radians(left_angle)
    right_angle_rad = np.radians(right_angle)

    radius = (left_distance + right_distance) / 2
    central_angle = abs(right_angle_rad - left_angle_rad)

    width = 2 * radius * np.sin(central_angle / 2)
    return width


def plot(scan, distance_max):
    global width_measurements, running

    scan_angles, scan_distances = get_scan_data(scan, distance_max)

    if len(scan_angles) > 0:
        angles, log_response, edge_points = compute_log_edge_detection(
            scan_angles, scan_distances, sigma=LOG_SIGMA, threshold=LOG_THRESHOLD
        )

        if len(edge_points) > 0:
            # Take the two strongest edges
            edge_points_sorted_by_strength = sorted(
                edge_points, key=lambda x: x[2], reverse=True
            )
            top_edges = edge_points_sorted_by_strength[:2]
            sorted_by_angle = sorted(top_edges, key=lambda x: x[0])

            width_value = None
            if len(sorted_by_angle) >= 2:
                center_angle = 180.0
                distances_from_center = [
                    (abs(point[0] - center_angle), i)
                    for i, point in enumerate(sorted_by_angle)
                ]
                distances_from_center.sort()
                left_idx = None
                right_idx = None

                if len(distances_from_center) >= 2:
                    idx1 = distances_from_center[0][1]
                    idx2 = distances_from_center[1][1]
                    if sorted_by_angle[idx1][0] < sorted_by_angle[idx2][0]:
                        left_idx, right_idx = idx1, idx2
                    else:
                        left_idx, right_idx = idx2, idx1

                    left_edge = sorted_by_angle[left_idx]
                    right_edge = sorted_by_angle[right_idx]

                    width_value = channel_width(left_edge, right_edge)
                    if width_value:
                        width_measurements.append(width_value)
                        num_samples = len(width_measurements)
                        avg_width = sum(width_measurements) / num_samples
                        if num_samples >= NUM_WIDTH_SAMPLES:
                            return avg_width
                            running = False
