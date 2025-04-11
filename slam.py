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
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
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

map = np.zeros((480, 270))
position = np.zeros(3)

bot.get_motor_encoder()
time.sleep(1)
_, prev_left_encoder, _, prev_right_encoder = bot.get_motor_encoder()

encoder_to_meter: float = 1 / (2 * math.pi * 0.03) * 850
_, _, yaw = bot.get_imu_attitude_data()

fix_angel_drift: float = yaw
robot_running: bool = False
yaw_final = yaw
yaw_init = yaw

q_star = 0.5
repulse_strength = 1
d_star_goal = 0.2
attractive_strength = 1
goal_position = [0, 2]


# pygame setup
pygame.init()
screen = pygame.display.set_mode((1000, 1000))
clock = pygame.time.Clock()
running = True


def tictoc(func):
    def wrapper():
        before_time = time.time()
        func()
        diff_time = time.time() - before_time
        print(f"{func.__name__} ran in {diff_time} seconds")


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def vector_to_steering(angle: float) -> float:
    if angle >= 0:
        return angle

    if angle > -90:
        return 0

    return 180


def repulsive_formal(axis: float, distance: float) -> float:
    return (
        repulse_strength
        * ((1 / q_star) - (1 / distance))
        * (1 / (distance**2))
        * ((axis) / distance)
    )


def attractive_formal(axis: float) -> float:
    distance = np.linalg.norm(
        [goal_position[0] - position[0], goal_position[1] - position[1]]
    )

    if distance <= d_star_goal:
        return attractive_strength * axis

    return (d_star_goal * attractive_strength * axis) / (distance)


def lidar() -> None:
    if not laser.doProcessSimple(scan):
        return

    potential_sum = np.zeros(2)

    for point in scan.points:
        angle: float = point.angle
        ran: float = point.range

        if 0.1 > ran:
            continue

        x: float = math.cos(angle - math.radians(position[2])) * ran
        y: float = math.sin(angle - math.radians(position[2])) * ran

        # if q_star > ran:
        #     potential_sum[0] -= repulsive_formal(x, ran)
        #     potential_sum[1] -= repulsive_formal(y, ran)

        potential_sum[0] += attractive_formal(goal_position[0] - position[0])
        potential_sum[1] += attractive_formal(goal_position[1] - position[1])

        resultant_magnitude = np.linalg.norm(potential_sum)
        resultant_angle = np.degrees(np.arctan2(potential_sum[1], potential_sum[0]))

        resultant_angle = clamp(vector_to_steering(resultant_angle), 30, 160)
        resultant_magnitude = clamp(resultant_magnitude, 0, 0)

        bot.set_motor(0, resultant_magnitude, 0, resultant_magnitude)
        bot.set_pwm_servo(1, resultant_angle)

        x_draw: float = x * 100 + 500 + position[0] * 100
        y_draw: float = y * 100 + 500 + position[1] * 100

        pygame.draw.circle(screen, (255, 0, 0), (x_draw, y_draw), 2)


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
        if event.type == pygame.MOUSEBUTTONUP:
            goal_position = (pygame.mouse.get_pos()) / 100

    screen.fill("white")
    key = pygame.key.get_pressed()

    steering_angle: int
    speed: int

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

    lidar()
    deadreckoning()
    pygame.draw.circle(
        screen, (0, 255, 0), (500 + position[0] * 100, 500 + position[1] * 100), 8
    )

    bot.set_motor(0, speed, 0, speed)
    bot.set_pwm_servo(1, steering_angle)

    pygame.display.flip()

    clock.tick(1000)  # limits FPS to 60
    print(clock.get_fps())


laser.turnOff()
laser.disconnecting()
pygame.quit()
