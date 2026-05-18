# rover_keys.py
import curses
import time
from gpiozero import Device, LED, PWMOutputDevice, DigitalOutputDevice
from gpiozero import DistanceSensor
from hal.pin_config import (
    STATUS_LED,
    MOTOR_LEFT_ENA, MOTOR_LEFT_IN1, MOTOR_LEFT_IN2,
    MOTOR_RIGHT_ENB, MOTOR_RIGHT_IN3, MOTOR_RIGHT_IN4,
    ULTRASONIC_TRIG, ULTRASONIC_ECHO,
)

led       = LED(STATUS_LED)
left_ena  = PWMOutputDevice(MOTOR_LEFT_ENA,  initial_value=0)
left_in1  = DigitalOutputDevice(MOTOR_LEFT_IN1,  initial_value=False)
left_in2  = DigitalOutputDevice(MOTOR_LEFT_IN2,  initial_value=False)
right_enb = PWMOutputDevice(MOTOR_RIGHT_ENB, initial_value=0)
right_in3 = DigitalOutputDevice(MOTOR_RIGHT_IN3, initial_value=False)
right_in4 = DigitalOutputDevice(MOTOR_RIGHT_IN4, initial_value=False)
sensor    = DistanceSensor(echo=ULTRASONIC_ECHO, trigger=ULTRASONIC_TRIG)

right_speed = 0.5
left_speed  = 0.5

DRIVE_MODE = 0
MODE_NAMES = {0: "Manual + Brake", 1: "Auto Avoid"}

OBSTACLE_THRESHOLD_M = 0.50
AUTO_BACKUP_SPEED    = 0.1
AUTO_BACKUP_DURATION = 0.8
AUTO_TURN_SPEED      = 0.1
AUTO_TURN_DURATION   = 0.8

def get_distance() -> float:
    d = sensor.distance
    return d if d is not None else float('inf')

def set_motors(left: float, right: float) -> None:
    left_in1.value  = left > 0
    left_in2.value  = left < 0
    left_ena.value  = abs(left)
    right_in3.value = right > 0
    right_in4.value = right < 0
    right_enb.value = abs(right)

def drive_stop()     -> None: set_motors(0.0, 0.0)
def drive_left()     -> None: set_motors( left_speed, -right_speed)
def drive_right()    -> None: set_motors(-left_speed,  right_speed)
def drive_forward()  -> None: set_motors( left_speed,  right_speed)
def drive_backward() -> None: set_motors(-left_speed, -right_speed)

def cleanup() -> None:
    drive_stop()
    for d in (left_ena,left_in1,left_in2,right_enb,right_in3,right_in4,led,sensor):
        d.close()

def auto_avoid_step() -> None:
    dist = get_distance()
    if dist > OBSTACLE_THRESHOLD_M:
        drive_forward()
    else:
        set_motors(-AUTO_BACKUP_SPEED, -AUTO_BACKUP_SPEED)
        time.sleep(AUTO_BACKUP_DURATION)
        set_motors(-AUTO_TURN_SPEED, AUTO_TURN_SPEED)
        time.sleep(AUTO_TURN_DURATION)

def main(stdscr):
    global left_speed, right_speed, DRIVE_MODE

    stdscr.clear()
    stdscr.addstr(2, 0, "Welcome... Ready to drive?")
    for _ in range(3):
        led.on();  time.sleep(0.2)
        led.off(); time.sleep(0.2)

    stdscr.clear()
    stdscr.nodelay(True)
    curses.curs_set(0)

    stdscr.addstr(0, 0,
        "Controls: W/A/S/D (press to move, stays moving)  Space (stop)  1-9/0 (speed)  m (mode)  q (quit)")

    def draw_speed():
        pct = int(left_speed * 100)
        bar = "\u2588" * int(left_speed*20) + "\u2591" * (20-int(left_speed*20))
        stdscr.addstr(4, 0, f"Speed: [{bar}] {pct:3d}%  ")

    def draw_mode():
        stdscr.addstr(5, 0, f"Mode:  {MODE_NAMES[DRIVE_MODE]}           ")

    def draw_distance(dist: float):
        s = " >4.0 m (clear)" if dist == float('inf') else f"{dist:5.2f} m"
        stdscr.addstr(6, 0, f"Dist:  {s}   ")

    def draw_status(msg: str):
        stdscr.addstr(2, 0, f"{msg:<35}")

    draw_speed(); draw_mode()

    # current_action tracks what the rover should keep doing each tick.
    # 'stop' means idle. Pressing a key sets it; Space resets it.
    current_action = 'stop'

    while True:
        # Drain buffer, only care about the most recent key
        key = -1
        while True:
            ch = stdscr.getch()
            if ch == -1:
                break
            key = ch

        # ── Speed keys ────────────────────────────────────────
        if ord('1') <= key <= ord('9'):
            left_speed = right_speed = (key - ord('0')) / 10.0
            draw_speed()
        elif key == ord('0'):
            left_speed = right_speed = 0.99
            draw_speed()

        # ── Mode toggle ───────────────────────────────────────
        elif key == ord('m'):
            DRIVE_MODE = 1 - DRIVE_MODE
            drive_stop()
            current_action = 'stop'
            draw_mode()
            draw_status(f"Switched to {MODE_NAMES[DRIVE_MODE]}")

        # ── Quit ──────────────────────────────────────────────
        elif key == ord('q'):
            break

        # ── Movement keys → set current_action ───────────────
        elif key == ord('w'): current_action = 'forward'
        elif key == ord('s'): current_action = 'backward'
        elif key == ord('a'): current_action = 'left'
        elif key == ord('d'): current_action = 'right'
        elif key == ord(' '):
            current_action = 'stop'
            drive_stop()
            draw_status("Stopped               ")

        # ==========================================================
        # MODE 1 — Autonomous obstacle avoidance
        # ==========================================================
        if DRIVE_MODE == 1:
            auto_avoid_step()
            dist = get_distance()
            draw_distance(dist)
            draw_status("Obstacle! Avoiding..." if dist <= OBSTACLE_THRESHOLD_M
                        else "Auto: driving forward ")

        # ==========================================================
        # MODE 0 — Manual + Brake
        # ==========================================================
        else:
            dist = get_distance()
            draw_distance(dist)
            too_close = dist <= OBSTACLE_THRESHOLD_M

            if current_action == 'forward':
                if too_close:
                    set_motors(0.0, 0.0)
                    draw_status("BLOCKED - obstacle ahead!")
                else:
                    drive_forward()
                    draw_status("Moving forward        ")

            elif current_action == 'backward':
                drive_backward()
                draw_status("Moving backward       ")

            elif current_action == 'left':
                drive_left()
                draw_status("Turning left          ")

            elif current_action == 'right':
                drive_right()
                draw_status("Turning right         ")

            elif current_action == 'stop':
                if too_close:
                    draw_status("Obstacle ahead - turn!")
                else:
                    draw_status("Idle                  ")

        stdscr.refresh()
        time.sleep(0.05)
