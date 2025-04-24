import numpy as np
from sklearn.cluster import DBSCAN


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def vector_to_servo(angle: float) -> float:
    angle = 180 - angle

    if angle <= 180:
        return angle
    if angle <= 270:
        return 180
    return 0


def attractive_formal(goal_position, position, d_star_goal, attractive_strength):
    goal_xy = [goal_position[0] - position[0], goal_position[1] - position[1]]
    distance = np.linalg.norm(goal_xy)

    print(f"distance: {distance}")

    if distance <= d_star_goal:
        x = attractive_strength * goal_xy[0]
        y = attractive_strength * goal_xy[1]
        return [x, y]

    x = (d_star_goal * attractive_strength * goal_xy[0]) / (distance)
    y = (d_star_goal * attractive_strength * goal_xy[1]) / (distance)

    return [x, y]


def get_obstacles(repulse_cloud):
    db = DBSCAN(eps=0.1, min_samples=5)
    labels = db.fit_predict(repulse_cloud)

    obstacles = np.full((max(labels), 2), np.inf)

    for i, label in enumerate(labels):
        if label == -1:
            continue

        if np.linalg.norm(obstacles[label]) > np.linalg.norm(repulse_cloud[i]):
            obstacles[label] = repulse_cloud[i]

    return obstacles


def repulsive_formal(obstacles, q_star, repulse_strength) -> float:
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

    return [np.sum(x), -np.sum(y)]


def apf(
    goal_position,
    position,
    d_star_goal,
    attractive_strength,
    repulse_cloud,
    q_star,
    repulse_strength,
):
    potential_sum = np.zeros(2)

    potential_sum += attractive_formal(
        goal_position, position, d_star_goal, attractive_strength
    )

    obstacles = get_obstacles(repulse_cloud)
    potential_sum += repulsive_formal(obstacles, q_star, repulse_strength)

    resultant_magnitude = np.linalg.norm(potential_sum)
    resultant_angle = np.degrees(np.arctan2(potential_sum[1], potential_sum[0]))

    print(
        f"magnitude: {resultant_magnitude:.2f} | angle:{resultant_angle:.2f} | x: {position[0]:.2f} y: {position[1]:.2f} angel: {position[2]:.2f}"
    )

    resultant_angle = vector_to_servo(resultant_angle - position[2] + 90)
    resultant_magnitude = clamp(resultant_magnitude, 0, 35)

    return resultant_magnitude, resultant_angle
