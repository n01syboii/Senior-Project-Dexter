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

position = np.zeros(3)
map = np.zeros((480, 270))

bot.get_motor_encoder()
time.sleep(1)
_, prev_left_encoder, _, prev_right_encoder = bot.get_motor_encoder()

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1920, 1080))
clock = pygame.time.Clock()
running = True


def lidar():
    if not laser.doProcessSimple(scan):
        return

    for point in scan.points:
        angle = point.angle
        ran = point.range

        x = math.cos(angle) * ran * 100 + 960
        y = math.sin(angle) * ran * 100 + 540

        pygame.draw.circle(screen, (255, 0, 0), (x, y), 2)


def deadreckoning():
    global prev_left_encoder, prev_right_encoder
    _, current_left_encoder, _, current_right_encoder = bot.get_motor_encoder()

    left_encoder_diff = current_left_encoder - prev_left_encoder
    right_encoder_diff = current_right_encoder - prev_right_encoder
    ave_encoder_diff = (left_encoder_diff + right_encoder_diff) / 2

    prev_left_encoder = current_left_encoder
    prev_right_encoder = current_right_encoder


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("white")

    key = pygame.key.get_pressed()

    _, _, y = bot.get_imu_attitude_data()

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

    bot.set_motor(0, speed, 0, speed)
    bot.set_pwm_servo(1, steering_angle)

    pygame.display.flip()

    clock.tick(1000)  # limits FPS to 60

laser.turnOff()
laser.disconnecting()
pygame.quit()
