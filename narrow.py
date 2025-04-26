from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Lidar parameters
num_scans = 1
angle_dif = 30
angle_min = 180 - angle_dif
angle_max = 180 + angle_dif

# Parameters for LoG edge detection
LOG_SIGMA = 0.5  # Sigma for Gaussian smoothing
LOG_THRESHOLD = 0.5  # Threshold for edge detection sensitivity

# Width averaging parameters
NUM_WIDTH_SAMPLES = 10  # Number of width measurements to average
width_measurements = []  # List to store width measurements
running = True  # Flag to control main loop


def get_scan_data(scan, distance_max=10):
    scan_buffer = []

    for _ in range(num_scans):
        temp_data = []
        for point in scan.points:
            angle = round(np.degrees(point.angle) % 360, 2)
            distance = round(point.range, 2)
            if angle_min <= angle <= angle_max and 0 < distance < distance_max:
                temp_data.append((angle, distance))
        scan_buffer.append(temp_data)

    # Build angle-to-distance map
    angle_map = defaultdict(list)
    for scan in scan_buffer:
        for angle, distance in scan:
            angle_map[angle].append(distance)

    # Average distances per angle
    averaged_angles = []
    averaged_distances = []
    for angle, dists in sorted(angle_map.items()):
        averaged_angles.append(angle)
        averaged_distances.append(np.mean(dists))

    return averaged_angles, averaged_distances


def compute_log_edge_detection(angles, distances, sigma=1.1, threshold=0.05):
    # Sort by angles
    sorted_indices = np.argsort(angles)
    angles = np.array(angles)[sorted_indices]
    distances = np.array(distances)[sorted_indices]

    # Apply Gaussian smoothing
    smoothed_distances = gaussian_filter1d(distances, sigma=sigma)

    # Compute second derivative (Laplacian approximation)
    first_derivative = np.gradient(smoothed_distances, angles, edge_order=2)
    second_derivative = np.gradient(first_derivative, angles, edge_order=2)

    # Find zero crossings (edges)
    zero_crossings = []
    for i in range(len(second_derivative) - 1):
        # Check if the sign changes (zero crossing)
        if second_derivative[i] * second_derivative[i + 1] < 0:
            # Calculate edge strength (magnitude of change)
            edge_strength = abs(second_derivative[i] - second_derivative[i + 1])

            # Only keep strong edges (above threshold)
            if edge_strength > threshold:
                # Linear interpolation to find precise zero-crossing point
                if (
                    second_derivative[i] != second_derivative[i + 1]
                ):  # Avoid division by zero
                    t = -second_derivative[i] / (
                        second_derivative[i + 1] - second_derivative[i]
                    )
                    # Interpolated angle
                    crossing_angle = angles[i] + t * (angles[i + 1] - angles[i])

                    # Don't interpolate distance - handle jumps appropriately
                    distance_jump = abs(distances[i] - distances[i + 1])

                    # If there's a significant jump in distance, use the value before the jump
                    if distance_jump > 1.0:  # Adjust this threshold as needed
                        if distances[i] < distances[i + 1]:
                            raw_crossing_distance = distances[i]
                        else:
                            raw_crossing_distance = distances[i + 1]
                    else:
                        # If no significant jump, use the smaller distance to be conservative
                        raw_crossing_distance = min(distances[i], distances[i + 1])

                    zero_crossings.append(
                        (crossing_angle, raw_crossing_distance, edge_strength)
                    )

    return angles, second_derivative, zero_crossings


def channel_width(left_edge, right_edge):
    # Unpack angle and distance values
    left_angle, left_distance, _ = left_edge
    right_angle, right_distance, _ = right_edge

    # Convert angles to radians for calculations
    left_angle_rad = np.radians(left_angle)
    right_angle_rad = np.radians(right_angle)

    # Calculate angle difference in radians
    # theta = abs(right_angle_rad - left_angle_rad)

    # Calculate the average distance as our approximate radius
    radius = (left_distance + right_distance) / 2

    # Calculate the central angle
    central_angle = abs(right_angle_rad - left_angle_rad)

    # Law of Cosines
    #     width = np.sqrt(left_distance**2 + right_distance**2 - 2 * left_distance * right_distance * np.cos(theta))

    # Chord formula
    width = 2 * radius * np.sin(central_angle / 2)

    # Return None if width is unreasonably small
    return width


def plot(scan, distance_max):
    global width_measurements, running

    # Extract scan data directly
    scan_angles, scan_distances = get_scan_data(scan, distance_max)

    if len(scan_angles) > 0:  # Make sure we have data
        # Use LoG edge detection
        angles, log_response, edge_points = compute_log_edge_detection(
            scan_angles, scan_distances, sigma=LOG_SIGMA, threshold=LOG_THRESHOLD
        )

        # Clear vertical lines from the second plot (but keep the horizontal zero line)
        if len(edge_points) > 0:
            # First sort by strength to get the top 2 edges
            edge_points_sorted_by_strength = sorted(
                edge_points, key=lambda x: x[2], reverse=True
            )
            top_edges = edge_points_sorted_by_strength[:2]

            # Now sort these top edges by angle for display
            sorted_by_angle = sorted(top_edges, key=lambda x: x[0])

            # Calculate channel width if we have at least 2 edges
            width_value = None
            if len(sorted_by_angle) >= 2:
                # Find the two edges closest to the center (180°)
                center_angle = 180.0

                # Calculate distance of each edge from center
                distances_from_center = [
                    (abs(point[0] - center_angle), i)
                    for i, point in enumerate(sorted_by_angle)
                ]

                # Sort by distance from center
                distances_from_center.sort()

                # Get the two closest edges to the center
                left_idx = None
                right_idx = None

                if len(distances_from_center) >= 2:
                    # Get indices of the two closest points
                    idx1 = distances_from_center[0][1]
                    idx2 = distances_from_center[1][1]

                    # Determine which is left and which is right
                    if sorted_by_angle[idx1][0] < sorted_by_angle[idx2][0]:
                        left_idx, right_idx = idx1, idx2
                    else:
                        left_idx, right_idx = idx2, idx1

                    left_edge = sorted_by_angle[left_idx]
                    right_edge = sorted_by_angle[right_idx]

                    # Calculate width
                    width_value = channel_width(left_edge, right_edge)
                    if width_value:
                        # Store the width measurement for averaging
                        width_measurements.append(width_value)

                        # Calculate current number of samples and average
                        num_samples = len(width_measurements)
                        avg_width = sum(width_measurements) / num_samples

                        # Check if we've collected enough samples
                        if num_samples >= NUM_WIDTH_SAMPLES:
                            return avg_width
                            running = False
