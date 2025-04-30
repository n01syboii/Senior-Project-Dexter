import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.15, min_samples=5)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def vector_to_servo(angle: float) -> float:
    if angle < 0:
        angle = 360 + angle

    if angle <= 180:
        return 180 - angle
    if angle <= 275:
        return 0
    return 180


def proportional_steering(angle: float, angle_range: float = 45) -> float:
    if angle < 0:
        angle += 360

    if 0 < angle <= 180:
        servo_angle = 180 - angle
    elif angle <= 270:
        servo_angle = 0
    else:
        servo_angle = 180

    output_min, output_max = 0 + angle_range, 180 - angle_range
    input_min, input_max = 0, 180

    mapped_output = output_min + (servo_angle - input_min) * (
        output_max - output_min
    ) / (input_max - input_min)

    # Ensure output is within bounds
    return max(output_min, min(output_max, mapped_output))


def get_obstacles(repulse_cloud):
    repulse_cloud = repulse_cloud.T

    if repulse_cloud.size == 0:
        return np.array([])

    labels = db.fit_predict(repulse_cloud)

    if max(labels) == -1:
        return np.array([])

    obstacles = np.full((max(labels) + 1, 2), np.inf)

    for i, label in enumerate(labels):
        if label == -1:
            continue

        if np.linalg.norm(obstacles[label]) > np.linalg.norm(repulse_cloud[i]):
            obstacles[label] = repulse_cloud[i]

    return obstacles


def get_tunnel_width(goal_position, position, point_cloud, obstacles):
    point_cloud = point_cloud.T

    obstacles_filter = []

    for obstacle in obstacles:
        obstacles_filter.append(
            -0.10 < obstacle[1]  # and (-allowed_x < obstacle[0] < allowed_x)
        )

    obstacles = obstacles[obstacles_filter]

    if len(obstacles) == 0:
        return -1.0, np.array([0.0, 0.0])

    left_edge = np.array([-100.0, -100.0])
    right_edge = np.array([100.0, 100.0])

    for obstacle in obstacles:
        if obstacle[0] < 0:
            if left_edge[0] < obstacle[0]:
                left_edge[0] = obstacle[0]
                left_edge[1] = obstacle[1]
        elif obstacle[0] >= 0:
            if right_edge[0] > obstacle[0]:
                right_edge[0] = obstacle[0]
                right_edge[1] = obstacle[1]

    if left_edge[0] == -100 or right_edge[0] == 100:
        return -1.0, np.array([0.0, 0.0])

    entrance_point = (left_edge + right_edge) / 2

    tunnel_cloud = []
    error_range = 0.05

    for point in point_cloud:
        min_x = left_edge[0] - error_range
        max_x = right_edge[0] + error_range
        min_y = min(right_edge[1], left_edge[1]) - error_range
        max_y = goal_position[1] + error_range

        if min_x < point[0] < max_x and min_y < point[1] < max_y:
            tunnel_cloud.append(point)

    if len(tunnel_cloud) == 0:
        return 0.0, entrance_point

    tunnel_cloud = np.array(tunnel_cloud)
    labels = db.fit_predict(tunnel_cloud)

    if max(labels) != 1:
        return 0.0, entrance_point

    cluster_0 = tunnel_cloud[labels == 0]
    cluster_1 = tunnel_cloud[labels == 1]

    distances = cdist(cluster_0, cluster_1)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)

    return distances[min_idx], entrance_point


def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = np.clip(dot / norm_product, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)


def has_approximately_opposite_vectors(vectors, angle_tolerance=10):
    vectors = [np.array(v) for v in vectors]
    lower_bound = 180 - angle_tolerance
    upper_bound = 180 + angle_tolerance

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            angle = angle_between(vectors[i], vectors[j])
            if lower_bound <= angle <= upper_bound:
                return True
    return False


def is_in_tunnel(obstacles):
    return has_approximately_opposite_vectors(obstacles, 25)


def adaptive_q_star(width, default_q_star=0.8, min_q_star=0.3, safety_margin=0.1):
    if width == 0.0:
        return default_q_star

    # adjusted_width = width - safety_margin
    # new_q_star = max(adjusted_width / 2, min_q_star)
    # q_star = 0.8 * new_q_star + 0.2 * default_q_star

    return 0.4


def attractive_formal(goal_position, position, d_star_goal, attractive_strength, mul=1):
    goal_xy = [goal_position[0] - position[0], goal_position[1] - position[1]]
    distance = np.linalg.norm(goal_xy)

    if distance <= d_star_goal:
        x = attractive_strength * goal_xy[0]
        y = attractive_strength * goal_xy[1]
        return [x, y]

    x = (d_star_goal * attractive_strength * goal_xy[0] * mul) / (distance)
    y = (d_star_goal * attractive_strength * goal_xy[1] * mul) / (distance)

    return np.array([x, y])


def repulsive_formal(obstacles, q_star, repulse_strength, div) -> float:
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

    return [np.sum(x) / div, np.sum(y) / div]


prev_width = 0.0
prev_point = np.zeros(2)
width_timer = 0


def apf(
    goal_position,
    position,
    point_cloud,
    repulse_cloud,
    d_star_goal,
    attractive_strength,
    q_star,
    min_q_star,
    repulse_strength,
):
    global prev_width, prev_point, width_timer
    potential_sum = np.zeros(2)
    attractive = np.zeros(2)
    repulse = np.zeros(2)

    obstacles = get_obstacles(repulse_cloud)
    tunnel_width, entrance_point = get_tunnel_width(
        goal_position, position, point_cloud, obstacles
    )

    attractive += attractive_formal(
        goal_position, position, d_star_goal, attractive_strength
    )

    in_tunnel = is_in_tunnel(obstacles)

    if tunnel_width <= 0:
        width_timer += 1
        if width_timer < 60:
            tunnel_width = prev_width
            entrance_point = prev_point
    else:
        prev_width = tunnel_width
        prev_point = entrance_point
        width_timer = 0

    print(f"tunnel_weidth: {tunnel_width}")
    print(attractive)

    min_tunnel_width = 0.35

    if 0 <= tunnel_width < min_tunnel_width:
        entrance_point[1] -= 0.25
        if entrance_point[0] != 0 and entrance_point[1] != 0:
            obstacles = np.vstack([obstacles, entrance_point])
    elif tunnel_width >= min_tunnel_width:
        entrance_point[1] += 0.8
        q_star = adaptive_q_star(tunnel_width, q_star, min_q_star)
        print
        attractive += attractive_formal(
            (entrance_point + [position[0], position[1]]),
            position,
            d_star_goal,
            attractive_strength,
            15,
        )
    print(attractive)

    if in_tunnel:
        q_star = adaptive_q_star(tunnel_width, q_star, min_q_star)

    potential_sum += attractive

    repulse = repulsive_formal(obstacles.T, q_star, repulse_strength, 1)
    potential_sum += repulse

    # repulse_magnitude = np.linalg.norm(repulse)
    # repulse_angle = np.degrees(np.arctan2(repulse[1], repulse[0]))

    attractive_magnitude = np.linalg.norm(attractive)
    attractive_angle = np.degrees(np.arctan2(attractive[1], attractive[0]))

    resultant_magnitude = np.linalg.norm(potential_sum)
    resultant_angle = np.degrees(np.arctan2(potential_sum[1], potential_sum[0]))

    if resultant_angle < 0:
        resultant_angle = 360 + resultant_angle

    # if repulse_angle < 0:
    #     repulse_angle = 360 + repulse_angle

    if attractive_angle < 0:
        attractive_angle = 360 + attractive_angle

    print(
        f"attractive | magnitudr: {attractive_magnitude:.2f} | angle:{attractive_angle:.2f}"
    )

    # print(f"repulse | magnitude: {repulse_magnitude:.2f} | angle:{repulse_angle:.2f}")

    print(
        f"magnitude: {resultant_magnitude:.2f} | angle:{resultant_angle:.2f} | x: {position[0]:.2f} y: {position[1]:.2f} angel: {position[2]:.2f}"
    )
    print(f"angel to move to {resultant_angle - position[2] + 90.00:.2f}")

    if not in_tunnel:
        resultant_angle = vector_to_servo(resultant_angle - position[2] + 90)
    else:
        resultant_angle = proportional_steering(resultant_angle - position[2] + 90, 70)

    resultant_magnitude = clamp(resultant_magnitude, 0, 35)

    return resultant_magnitude, resultant_angle
