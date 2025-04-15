import math
import time

# measure the yaw of the robot with the hlp of IMU sensor
import numpy as np
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

map = np.zeros((480, 270))
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

q_star = 0.5
repulse_strength = 1
d_star_goal = 0.5
attractive_strength = 250
goal_position = [0.0, 1]


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def vector_to_steering(angle: float) -> float:
    if angle >= 0:
        return angle

    if angle > -90:
        return 0

    return 180


def repulsive_formal(repulse_cloud) -> float:
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

    return np.sum(x), np.sum(y)


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

    angle_list = []
    range_list = []
    repulse_list = []

    for point in scan.points:
        point_range = point.range

        if point_range < 0.09:
            continue

        angle_list.append(point.angle + math.pi / 2)
        range_list.append(point_range)
        repulse_list.append(point_range < q_star)

    angle_list = np.array(angle_list)
    range_list = np.array(range_list)

    pos_angle = math.radians(position[2] - 90)
    angle_list -= pos_angle

    x: float = np.cos(angle_list) * range_list
    y: float = np.sin(angle_list) * range_list

    point_cloud = [x, y]
    repulse_cloud = [x[repulse_list], y[repulse_list]]

    return point_cloud, repulse_cloud


def apf(repulse_cloud):
    potential_sum = np.zeros(2)

    # x, y = repulsive_formal(repulse_cloud)
    # potential_sum[0] -= x
    # potential_sum[1] -= y

    potential_sum[0] += attractive_formal(goal_position[0] - position[0])
    potential_sum[1] += attractive_formal(goal_position[1] - position[1])

    resultant_magnitude = np.linalg.norm(potential_sum)
    resultant_angle = np.degrees(np.arctan2(potential_sum[1], potential_sum[0]))

    print(
        f"magnitude: {resultant_magnitude:.2f} | angle:{resultant_angle:.2f} | x: {position[0]:.2f} y: {position[1]:.2f} angel: {position[2]:.2f}"
    )

    resultant_angle = clamp(vector_to_steering(resultant_angle), 30, 160)
    resultant_magnitude = clamp(resultant_magnitude, 0, 40)

    bot.set_motor(0, resultant_magnitude, 0, resultant_magnitude)
    bot.set_pwm_servo(1, resultant_angle)


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


while True:
    start = time.perf_counter()
    point_cloud, repulse_cloud = lidar()

    apf(repulse_cloud)
    deadreckoning()

    end = time.perf_counter()
    print(f"fps {(1 / (end - start))}")


laser.turnOff()
laser.disconnecting()
