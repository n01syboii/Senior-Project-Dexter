import math
import time

from Rosmaster_Lib import Rosmaster

bot = Rosmaster()
bot.create_receive_threading()

encoder_to_meter = 1 / (2 * math.pi * 0.03) * 850


def moveDistance(m):
    _, prev_left_encoder, _, prev_right_encoder = bot.get_motor_encoder()
    while True:
        _, current_left_encoder, _, current_right_encoder = bot.get_motor_encoder()

        left_encoder_diff = current_left_encoder - prev_left_encoder
        right_encoder_diff = current_right_encoder - prev_right_encoder
        ave_encoder_diff = (left_encoder_diff + right_encoder_diff) / 2

        if ave_encoder_diff >= m * encoder_to_meter:
            break

        bot.set_motor(0, 30, 0, 30)

    bot.set_motor(0, 0, 0, 0)
    print(
        "Distance moved in cm = ", (ave_encoder_diff * 100) / encoder_to_meter
    )  # print value in cm


def main():
    bot.get_motor_encoder()
    time.sleep(1)
    moveDistance(0.5)


if __name__ == "__main__":
    main()
