import numpy as np
from sklearn.cluster import DBSCAN

# DBSCAN clustering for obstacle grouping
db = DBSCAN(eps=0.1, min_samples=5)

def clamp(n, minn, maxn):
    # Clamp n between minn and maxn
    return max(min(maxn, n), minn)

def vector_to_servo(angle: float) -> float:
    # Convert vector angle to servo-compatible angle
    angle = 180 - angle

    if angle <= 180:
        return angle
    if angle <= 280:
        return 180
    return 0

def get_obstacles(repulse_cloud):
    # Cluster repulsive points and return closest obstacle in each cluster
    if repulse_cloud.size == 0:
        return np.array([])

    repulse_cloud = repulse_cloud.T
    labels = db.fit_predict(repulse_cloud)

    if max(labels) == -1:
        return np.array([])

    obstacles = np.full((max(labels) + 1, 2), np.inf)

    for i, label in enumerate(labels):
        if label == -1:
            continue
        if np.linalg.norm(obstacles[label]) > np.linalg.norm(repulse_cloud[i]):
            obstacles[label] = repulse_cloud[i]

    return obstacles.T

def tunnel(goal_position, position, point_cloud, obstacles):
    # Identify tunnel endpoints between obstacles
    obstacles = obstacles.T
    point_cloud = point_cloud.T

    for obstacle in obstacles:
        if obstacle[0] > 0:
            right_edge = obstacle
        else:
            left_edge = obstacle

    tunnel_cloud = []

    for point in point_cloud:
        min_x = left_edge[0]
        max_x = right_edge[0]
        min_y = min(right_edge[1], left_edge[1])
        max_y = goal_position[1]

        if min_x < point[0] < max_x and min_y < point[1] < max_y:
            tunnel_cloud.append(point)

    tunnel_cloud = np.array(tunnel_cloud)

    tunnel_cloud[:, 0] += position[0] - goal_position[0]
    tunnel_cloud[:, 1] += position[1] - goal_position[1]

    labels = db.fit_predict(tunnel_cloud)

    if max(labels) == -1:
        return np.array([])

    tunnel_end = np.full((max(labels) + 1, 2), np.inf)

    for i, label in enumerate(labels):
        if label == -1:
            continue
        if np.linalg.norm(tunnel_end[label]) > np.linalg.norm(tunnel_cloud[i]):
            tunnel_end[label] = tunnel_cloud[i]

    tunnel_end[:, 0] += goal_position[0] - position[0]
    tunnel_end[:, 1] += goal_position[1] - position[1]

    print(f"weidth: {np.linalg.norm(tunnel_end[0] - tunnel_end[1])}")

    return tunnel_end

def attractive_formal(goal_position, position, d_star_goal, attractive_strength):
    # Attractive force calculation towards goal
    goal_xy = [goal_position[0] - position[0], goal_position[1] - position[1]]
    distance = np.linalg.norm(goal_xy)

    if distance <= d_star_goal:
        x = attractive_strength * goal_xy[0]
        y = attractive_strength * goal_xy[1]
        return [x, y]

    x = (d_star_goal * attractive_strength * goal_xy[0]) / (distance)
    y = (d_star_goal * attractive_strength * goal_xy[1]) / (distance)

    return [x, y]

def repulsive_formal(obstacles, q_star, repulse_strength) -> float:
    # Repulsive force calculation from obstacles
    if obstacles.size == 0:
        return [0, 0]

    distance = (obstacles[0] ** 2 + obstacles[1] ** 2) ** 0.5

    x = (
        repulse_strength
        * ((1 / q_star) - (1 / distance))
        * (1 / (distance**2))
        * (obstacles[0] / distance)
    )

    y = (
        repulse_strength
        * ((1 / q_star) - (1 / distance))
        * (1 / (distance**2))
        * (obstacles[1] / distance)
    )

    return [np.sum(x), np.sum(y)]

def apf(
    goal_position,
    position,
    point_cloud,
    repulse_cloud,
    d_star_goal,
    attractive_strength,
    q_star,
    repulse_strength,
):
    # Main Artificial Potential Field (APF) function
    potential_sum = np.zeros(2)

    potential_sum += attractive_formal(
        goal_position, position, d_star_goal, attractive_strength
    )

    obstacles = get_obstacles(repulse_cloud)
    print(f"obstacle: {obstacles.T}")

    # Uncomment below to use tunnel endpoint logic
    # tunnel_end = tunnel(goal_position, position, point_cloud, obstacles)
    # print(f"tunnel_end: {tunnel_end}")

    potential_sum += repulsive_formal(obstacles, q_star, repulse_strength)

    resultant_magnitude = np.linalg.norm(potential_sum)
    resultant_angle = np.degrees(np.arctan2(potential_sum[1], potential_sum[0]))

    print(
        f"magnitude: {resultant_magnitude:.2f} | angle:{resultant_angle:.2f} | x: {position[0]:.2f} y: {position[1]:.2f} angel: {position[2]:.2f}"
    )

    resultant_angle = vector_to_servo(resultant_angle - position[2] + 90)
    resultant_magnitude = clamp(resultant_magnitude, 0, 35)

    return resultant_magnitude, resultant_angle
