import math
import time

# measure the yaw of the robot with the hlp of IMU sensor
import numpy as np
import pygame
import ydlidar
from Rosmaster_Lib import Rosmaster

ports = ydlidar.lidarPortList()
port = "/dev/ydlidar"
for key, value in ports.items():
    port = value

laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 512000)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TOF)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 15.0)
laser.setlidaropt(ydlidar.LidarPropSampleRate, 20)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)
laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
laser.setlidaropt(ydlidar.LidarPropMaxRange, 32.0)
laser.setlidaropt(ydlidar.LidarPropMinRange, 0.01)
scan = ydlidar.LaserScan()
lidar_init = laser.initialize() and laser.turnOn()


bot = Rosmaster()
bot.create_receive_threading()

position = np.array([0.0, 0.0, 90.0])
mini_map = np.zeros((250, 250))
map = np.zeros((1000, 1000))

grid_size = 4
grid_threshold = 5

bot.get_motor_encoder()
time.sleep(1)
_, prev_left_encoder, _, prev_right_encoder = bot.get_motor_encoder()

encoder_to_meter: float = 1 / (2 * math.pi * 0.03) * 850
_, _, yaw = bot.get_imu_attitude_data()

fix_angel_drift: float = yaw - 90
robot_running: bool = False
yaw_final = yaw
yaw_init = yaw

q_star = 1
repulse_strength = 5
d_star_goal = 0.5
attractive_strength = 100
goal_position = [0.5, 2.0]


# pygame setup
pygame.init()
screen = pygame.display.set_mode((1000, 1000))
clock = pygame.time.Clock()
running = True
font = pygame.font.SysFont("Arial", 18)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def tranfrom_matrix(matrix, angle_degrees):
    theta = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    matrix = matrix @ rotation_matrix
    matrix[:, 0] += 500 + int(position[0] * 100)
    matrix[:, 1] += 500 + int(position[1] * 100)

    return matrix


def update_fps():
    fps = str(int(clock.get_fps()))
    fps_text = font.render(fps, 1, pygame.Color("coral"))
    return fps_text


def vector_to_servo(angle: float) -> float:
    angle = 180 - angle

    if angle <= 180:
        return angle
    if angle <= 270:
        return 180
    return 0


def repulsive_formal(repulse_cloud) -> float:
    num_points = len(repulse_cloud[0])

    if num_points == 0:
        return 0.0, 0.0

    distance = (repulse_cloud[0] ** 2 + repulse_cloud[1] ** 2) ** 0.5

    x = (
        repulse_strength
        * ((1 / q_star) - (1 / distance))
        * (1 / (distance**2))
        * (repulse_cloud[0] / distance)
    )

    y = (
        repulse_strength
        * ((1 / q_star) - (1 / distance))
        * (1 / (distance**2))
        * (repulse_cloud[1] / distance)
    )

    return [np.sum(x) / num_points, -np.sum(y) / num_points]


def attractive_formal():
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


def lidar() -> None:
    if not laser.doProcessSimple(scan):
        return

    angle_list = []
    range_list = []
    repulse_list = []

    for point in scan.points:
        point_range = point.range
        point_angle = point.angle

        if point_range < 0.09 or -0.7 < point_angle < 0.7:
            continue

        angle_list.append(point.angle + math.pi / 2)
        range_list.append(point_range)
        repulse_list.append(
            (
                point_range
                < q_star  # and not (-math.pi / 2 < point_angle < math.pi / 2)
            )
        )

    angle_list = np.array(angle_list)
    range_list = np.array(range_list)

    pos_angle = math.radians(position[2] - 90)
    angle_list -= pos_angle

    x: float = np.cos(angle_list) * range_list
    y: float = np.sin(angle_list) * range_list

    point_cloud = np.array([x, y])
    repulse_cloud = np.array([x[repulse_list], y[repulse_list]])

    return point_cloud, repulse_cloud


def make_mini_map(point_cloud):
    mini_map = np.zeros(250, 250)
    other_map = [
        np.floor(point_cloud[0] / grid_size),
        np.floor(point_cloud[1] / grid_size),
    ]

    for point in other_map:
        mini_map[point[0], point[1]] += 1 / grid_threshold

    mini_map = (mini_map >= 1).astype(int)
    return mini_map


def apf(repulse_cloud):
    potential_sum = np.zeros(2)

    potential_sum += repulsive_formal(repulse_cloud)
    potential_sum += attractive_formal()

    resultant_magnitude = np.linalg.norm(potential_sum)
    resultant_angle = np.degrees(np.arctan2(potential_sum[1], potential_sum[0]))

    print(
        f"magnitude: {resultant_magnitude:.2f} | angle:{resultant_angle:.2f} | x: {position[0]:.2f} y: {position[1]:.2f} angel: {position[2]:.2f}"
    )

    draw_points = [
        [0, 10],
        [0, -10],
        [35, -10],
        [35, -15],
        [50, 0],
        [35, 15],
        [35, 10],
    ]

    draw_points = tranfrom_matrix(draw_points, resultant_angle)
    pygame.draw.polygon(screen, (30, 0, 255), draw_points)

    resultant_angle = vector_to_servo(resultant_angle)
    resultant_magnitude = clamp(resultant_magnitude, 0, 40)

    # bot.set_motor(0, resultant_magnitude, 0, resultant_magnitude)
    bot.set_pwm_servo(1, resultant_angle)


def draw_map(point_cloud) -> None:
    x_draw: float = point_cloud[0] * 100 + 500 + position[0] * 100
    y_draw: float = point_cloud[1] * 100 + 500 + position[1] * 100

    for i, _ in enumerate(x_draw):
        pygame.draw.circle(screen, (255, 0, 0), (x_draw[i], y_draw[i]), 2)


def deadreckoning() -> None:
    global prev_left_encoder, prev_right_encoder, fix_angel_drift
    global yaw, robot_running, yaw_init, yaw_final, position

    _, current_left_encoder, _, current_right_encoder = bot.get_motor_encoder()

    left_encoder_diff: float = current_left_encoder - prev_left_encoder
    right_encoder_diff: float = current_right_encoder - prev_right_encoder
    ave_encoder_diff: float = (left_encoder_diff + right_encoder_diff) / 2

    if ave_encoder_diff != 0:
        if not robot_running:
            robot_running = True
            _, _, yaw_final = bot.get_imu_attitude_data()
            fix_angel_drift += yaw_final - yaw_init

        _, _, yaw = bot.get_imu_attitude_data()
        real_angle = yaw - fix_angel_drift

        distance = ave_encoder_diff / encoder_to_meter
        position[0] += math.cos(math.radians(real_angle)) * distance
        position[1] += math.sin(math.radians(real_angle)) * distance
        position[2] = real_angle

    else:
        if robot_running:
            robot_running = False
            _, _, yaw_init = bot.get_imu_attitude_data()

    prev_left_encoder = current_left_encoder
    prev_right_encoder = current_right_encoder


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("white")
    screen.blit(update_fps(), (10, 0))

    draw_points = np.array(
        [
            [-15, 10],
            [-15, -10],
            [15, 0],
        ]
    )

    draw_points = tranfrom_matrix(draw_points, position[2])
    pygame.draw.polygon(screen, (0, 255, 0), draw_points)
    pygame.draw.circle(
        screen,
        (255, 255, 0),
        (
            goal_position[0] * 100 - position[0] * 100 + 500,
            -goal_position[1] * 100 + position[1] * 100 + 500,
        ),
        8,
    )

    point_cloud, repulse_cloud = lidar()

    draw_map(point_cloud)
    apf(repulse_cloud)
    deadreckoning()

    key = pygame.key.get_pressed()

    steering_angle: int
    speed: int

    if key[pygame.K_r]:
        position = np.array([0.0, 0.0, 90.0])

    if key[pygame.K_w]:
        speed = 45
    elif key[pygame.K_s]:
        speed = -45
    else:
        speed = 0

    if key[pygame.K_d]:
        steering_angle = 150
    elif key[pygame.K_a]:
        steering_angle = 30
    else:
        steering_angle = 90

    bot.set_motor(0, speed, 0, speed)
    # bot.set_pwm_servo(1, steering_angle)

    pygame.display.update()
    clock.tick(60)  # limits FPS to 60


laser.turnOff()
laser.disconnecting()
pygame.quit()
