import math
import time

import numpy as np
import ydlidar
from Rosmaster_Lib import Rosmaster

from apf import apf

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

bot.get_motor_encoder()
time.sleep(1)
_, prev_left_encoder, _, prev_right_encoder = bot.get_motor_encoder()

encoder_to_meter: float = 1 / (2 * math.pi * 0.03) * 850
_, _, yaw = bot.get_imu_attitude_data()

fix_angel_drift: float = yaw - 90
robot_running: bool = False
yaw_final = yaw
yaw_init = yaw

q_star = 0.8
min_q_star = 0.3
repulse_strength = 8
d_star_goal = 0.8
attractive_strength = 150
goal_position = [0.0, 3.5]


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

        angle_list.append(-point.angle - math.pi / 2)
        range_list.append(point_range)
        repulse_list.append((point_range < q_star))

    angle_list = np.array(angle_list)
    range_list = np.array(range_list)

    pos_angle = math.radians(position[2] - 90)
    angle_list -= pos_angle

    x: float = np.cos(angle_list) * range_list
    y: float = np.sin(angle_list) * range_list

    point_cloud = np.array([x, y])
    repulse_cloud = np.array([x[repulse_list], y[repulse_list]])

    return point_cloud, repulse_cloud


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

        if real_angle < 0:
            real_angle = 360 + real_angle

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


while True:
    start = time.perf_counter()
    point_cloud, repulse_cloud = lidar()

    resultant_magnitude, resultant_angle = apf(
        goal_position,
        position,
        point_cloud,
        repulse_cloud,
        d_star_goal,
        attractive_strength,
        q_star,
        min_q_star,
        repulse_strength,
    )

    deadreckoning()
    correct_range = 0.3

    correct_x = (
        goal_position[0] - correct_range
        < position[0]
        < goal_position[0] + correct_range
    )
    correct_y = (
        goal_position[1] - correct_range
        < position[1]
        < goal_position[1] + correct_range
    )

    if correct_x and correct_y:
        bot.set_motor(0, 0, 0, 0)
        bot.set_pwm_servo(1, 90)
    else:
        bot.set_motor(0, resultant_magnitude, 0, resultant_magnitude)
        bot.set_pwm_servo(1, resultant_angle)

    end = time.perf_counter()
    # print(f"fps {(1 / (end - start))}")


laser.turnOff()
laser.disconnecting()
