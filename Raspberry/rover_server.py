# rover_server_test.py
# Web-based rover control server with USB camera stream.
# Run: python3 rover_server_test.py
# Then open http://<pi-ip>:5000 on any device on the same network.
#
# Dependencies:
#   pip install flask opencv-python --break-system-packages

import time
import threading
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template_string, Response
from gpiozero import LED, PWMOutputDevice, DigitalOutputDevice, DistanceSensor

from hal.pin_config import (
    STATUS_LED,
    MOTOR_LEFT_ENA, MOTOR_LEFT_IN1, MOTOR_LEFT_IN2,
    MOTOR_RIGHT_ENB, MOTOR_RIGHT_IN3, MOTOR_RIGHT_IN4,
    ULTRASONIC_TRIG, ULTRASONIC_ECHO,
)

# ── Hardware setup ────────────────────────────────────────────
led       = LED(STATUS_LED)
left_ena  = PWMOutputDevice(MOTOR_LEFT_ENA,  initial_value=0)
left_in1  = DigitalOutputDevice(MOTOR_LEFT_IN1,  initial_value=False)
left_in2  = DigitalOutputDevice(MOTOR_LEFT_IN2,  initial_value=False)
right_enb = PWMOutputDevice(MOTOR_RIGHT_ENB, initial_value=0)
right_in3 = DigitalOutputDevice(MOTOR_RIGHT_IN3, initial_value=False)
right_in4 = DigitalOutputDevice(MOTOR_RIGHT_IN4, initial_value=False)
sensor    = DistanceSensor(echo=ULTRASONIC_ECHO, trigger=ULTRASONIC_TRIG)

# ── Shared state ──────────────────────────────────────────────
right_speed = 0.5
left_speed  = 0.5
DRIVE_MODE  = 0
# 0 = Manual + Brake
# 1 = Auto Avoid
# 2 = Manual Free
# 3 = Cone Finder (orange color track)

OBSTACLE_THRESHOLD_M = 0.50
AUTO_BACKUP_SPEED    = 0.1
AUTO_BACKUP_DURATION = 0.8
AUTO_TURN_SPEED      = 0.1
AUTO_TURN_DURATION   = 0.8

state_lock = threading.Lock()

# ── USB Camera ────────────────────────────────────────────────
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

_camera      = cv2.VideoCapture(CAMERA_INDEX)
_camera.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
_camera_lock = threading.Lock()

# ── Cone Finder (orange detection) config ─────────────────────
# HSV range for orange. Tweak if lighting makes detection unreliable.
CONE_HSV_LOWER = np.array([ 10, 150, 100])
CONE_HSV_UPPER = np.array([ 25, 255, 255])

# Minimum contour area to count as a real cone (filters out noise).
CONE_MIN_AREA = 1500

# How centered the cone must be (fraction of frame width) before
# the rover drives straight rather than turning to align.
CONE_CENTER_DEADZONE = 0.15

# Ultrasonic distance at which the rover considers itself arrived.
CONE_STOP_DISTANCE_M = 0.40

# All cone-finder speeds are fractions of the manual speed slider (0.0–1.0).
# e.g. CONE_FORWARD_SCALE = 0.7 means the rover approaches at 70% of
# whatever the speed slider is set to.
CONE_FORWARD_SCALE = 0.70   # approach speed scale
CONE_TURN_SCALE    = 0.50   # steering correction scale
CONE_SEARCH_SCALE  = 0.35   # pivot speed while scanning

# How long (seconds) to pivot each tick while searching before re-checking.
CONE_SEARCH_TICK   = 0.08

# Shared detection result — written by generate_frames(), read by cone thread.
_cone_detection = {'found': False, 'cx': 0, 'area': 0, 'frame_w': CAMERA_WIDTH}
_cone_det_lock  = threading.Lock()

# Cone finder state:  'searching' | 'tracking' | 'arrived'
CONE_STATE  = 'searching'
_cone_thread  = None
_cone_running = False


# ── Frame generator ───────────────────────────────────────────
def generate_frames():
    """MJPEG stream. Draws orange detection overlay whenever mode 3 is active."""
    blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype='uint8')
    blank[:] = 30
    while True:
        with _camera_lock:
            ok, frame = _camera.read()
        if not ok:
            frame = blank.copy()

        # Always run detection so the overlay is live in any mode
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, CONE_HSV_LOWER, CONE_HSV_UPPER)
        mask = cv2.erode(mask,  None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        found = False
        cx    = 0
        area  = 0
        if contours:
            c    = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area >= CONE_MIN_AREA:
                found = True
                x, y, w, h = cv2.boundingRect(c)
                cx = x + w // 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 140, 255), 2)
                cv2.circle(frame, (cx, y + h//2), 6, (0, 140, 255), -1)
                cv2.putText(frame, f"Cone ({area:.0f}px)",
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 140, 255), 2)

        # Center guide line
        mid_x = CAMERA_WIDTH // 2
        cv2.line(frame, (mid_x, 0), (mid_x, CAMERA_HEIGHT), (80, 80, 200), 1)

        with _cone_det_lock:
            _cone_detection['found']   = found
            _cone_detection['cx']      = cx
            _cone_detection['area']    = area
            _cone_detection['frame_w'] = CAMERA_WIDTH

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


# ── Motor helpers ─────────────────────────────────────────────
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
    _camera.release()
    for d in (left_ena,left_in1,left_in2,right_enb,right_in3,right_in4,led,sensor):
        d.close()


# ── Auto-avoid thread ─────────────────────────────────────────
_auto_thread  = None
_auto_running = False

def _auto_avoid_loop():
    global _auto_running
    while _auto_running:
        dist = get_distance()
        if dist > OBSTACLE_THRESHOLD_M:
            drive_forward()
            time.sleep(0.05)
        else:
            set_motors(-AUTO_BACKUP_SPEED, -AUTO_BACKUP_SPEED)
            time.sleep(AUTO_BACKUP_DURATION)
            if not _auto_running:
                break
            set_motors(-AUTO_TURN_SPEED, AUTO_TURN_SPEED)
            time.sleep(AUTO_TURN_DURATION)
    drive_stop()

def start_auto():
    global _auto_thread, _auto_running
    _auto_running = True
    _auto_thread  = threading.Thread(target=_auto_avoid_loop, daemon=True)
    _auto_thread.start()

def stop_auto():
    global _auto_running
    _auto_running = False


# ── Cone-finder thread ────────────────────────────────────────
def _cone_loop():
    """
    State machine driven by the speed slider (left_speed):
      searching → pivot using CONE_SEARCH_SCALE * left_speed until cone appears
      tracking  → steer toward cone using CONE_TURN/FORWARD_SCALE * left_speed
                  stop when ultrasonic distance <= CONE_STOP_DISTANCE_M
      arrived   → stop motors, pivot slowly to scan for the next cone;
                  as soon as one is detected, switch back to tracking
    """
    global CONE_STATE, _cone_running

    while _cone_running:
        spd = left_speed   # read the live slider value each tick

        with _cone_det_lock:
            found   = _cone_detection['found']
            cx      = _cone_detection['cx']
            frame_w = _cone_detection['frame_w']

        dist = get_distance()

        # ── arrived: spin in short bursts, ignoring anything still too close ──
        if CONE_STATE == 'arrived':
            # Only accept a detection as a NEW object if the ultrasonic
            # confirms we're no longer right up against something.
            # This filters out the cone we just visited.
            clear_of_last = dist > CONE_STOP_DISTANCE_M
            if found and clear_of_last:
                # New object detected at a safe distance — start tracking
                CONE_STATE = 'tracking'
                drive_stop()
            else:
                # Spin a short burst then re-check
                set_motors(-spd, spd)
                time.sleep(CONE_SEARCH_TICK)
            continue

        # ── searching: no cone visible → pivot ───────────────
        if CONE_STATE == 'searching':
            if found:
                CONE_STATE = 'tracking'
            else:
                scan_spd = spd * CONE_SEARCH_SCALE
                set_motors(-scan_spd, scan_spd)
                time.sleep(CONE_SEARCH_TICK)
            continue

        # ── tracking: cone visible → approach ────────────────
        if CONE_STATE == 'tracking':
            if not found:
                CONE_STATE = 'searching'
                drive_stop()
                time.sleep(0.05)
                continue

            # Arrived when ultrasonic says close enough
            if dist <= CONE_STOP_DISTANCE_M:
                # Back up for 0.5 s to clear the current object
                back_end = time.time() + 0.5
                while time.time() < back_end and _cone_running:
                    spd = left_speed
                    set_motors(-spd, -spd)
                    time.sleep(0.02)
                drive_stop()
                # Turn right for 0.5 s so we face away from the visited object
                turn_end = time.time() + 0.5
                while time.time() < turn_end and _cone_running:
                    spd = left_speed
                    set_motors(spd, -spd)
                    time.sleep(0.02)
                drive_stop()
                CONE_STATE = 'arrived'
                continue

            mid_x    = frame_w / 2
            error    = (cx - mid_x) / mid_x   # -1.0 … +1.0
            deadzone = CONE_CENTER_DEADZONE

            if error < -deadzone:
                t = spd * CONE_TURN_SCALE
                set_motors(t, -t)              # turn left
            elif error > deadzone:
                t = spd * CONE_TURN_SCALE
                set_motors(-t, t)              # turn right
            else:
                f = spd * CONE_FORWARD_SCALE
                set_motors(f, f)               # drive straight

            time.sleep(0.05)

    drive_stop()

def start_cone():
    global _cone_thread, _cone_running, CONE_STATE
    CONE_STATE    = 'searching'
    _cone_running = True
    _cone_thread  = threading.Thread(target=_cone_loop, daemon=True)
    _cone_thread.start()

def stop_cone():
    global _cone_running
    _cone_running = False


# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)
current_action = 'stop'

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Rover Control</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0f1117; color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100svh; padding: 16px; gap: 16px;
  }
  h1 { font-size: 1.3rem; font-weight: 700; letter-spacing: 0.05em; color: #7dd3fc; }

  #camera-wrap {
    width: 100%; max-width: 640px; background: #1e2433;
    border-radius: 12px; overflow: hidden; position: relative; aspect-ratio: 4/3;
  }
  #camera-feed { width: 100%; height: 100%; object-fit: cover; display: block; }
  #cam-badge {
    position: absolute; top: 8px; left: 8px;
    background: rgba(0,0,0,0.55); color: #86efac;
    font-size: 0.7rem; font-weight: 700;
    padding: 3px 8px; border-radius: 20px; letter-spacing: 0.05em;
  }

  #status-bar {
    width: 100%; max-width: 640px; background: #1e2433;
    border-radius: 12px; padding: 12px 16px;
    display: flex; flex-direction: column; gap: 6px; font-size: 0.85rem;
  }
  .stat-row { display: flex; justify-content: space-between; align-items: center; }
  .stat-label { color: #94a3b8; }
  .stat-value { font-weight: 600; }
  #dist-value   { color: #7dd3fc; }
  #action-value { color: #86efac; }
  #obstacle-warning { color: #f87171; font-weight: 700; text-align: center; display: none; }
  #free-warning     { color: #fbbf24; font-weight: 700; text-align: center; display: none; }

  .slider-row {
    width: 100%; max-width: 640px;
    display: flex; align-items: center; gap: 12px; font-size: 0.85rem;
  }
  .slider-row label { color: #94a3b8; white-space: nowrap; }
  input[type=range] { flex: 1; accent-color: #7dd3fc; height: 6px; }
  #speed-display { color: #7dd3fc; font-weight: 700; min-width: 38px; text-align: right; }

  .mode-toggle {
    display: flex; gap: 8px; width: 100%; max-width: 640px; flex-wrap: wrap;
  }
  .mode-btn {
    flex: 1; min-width: 100px; padding: 10px; border-radius: 8px;
    border: 2px solid #334155; background: #1e2433;
    color: #94a3b8; font-size: 0.82rem; font-weight: 600;
    cursor: pointer; transition: all 0.15s;
  }
  .mode-btn.active      { border-color: #7dd3fc; color: #7dd3fc; background: #172033; }
  .mode-btn.active-free { border-color: #fbbf24; color: #fbbf24; background: #1f1800; }
  .mode-btn.active-cone { border-color: #fb923c; color: #fb923c; background: #1f0f00; }
  .mode-btn.active-rgb  { border-color: #4ade80; color: #4ade80; background: #0a1f0a; }

  .dpad {
    display: grid;
    grid-template-columns: repeat(3, 80px);
    grid-template-rows: repeat(3, 80px);
    gap: 8px; user-select: none;
  }
  .btn {
    border-radius: 12px; border: none; font-size: 1.6rem; cursor: pointer;
    background: #1e2433; color: #e2e8f0;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.1s, transform 0.08s;
    -webkit-tap-highlight-color: transparent; touch-action: manipulation;
  }
  .btn:active { background: #2d4a6e; transform: scale(0.93); }
  .btn.stop-btn { background: #2d1f1f; color: #f87171; font-size: 1.1rem; font-weight: 700; }
  .btn.stop-btn:active { background: #5c2626; }
  .btn.disabled-btn { opacity: 0.25; cursor: not-allowed; pointer-events: none; }
  .btn-up    { grid-column: 2; grid-row: 1; }
  .btn-left  { grid-column: 1; grid-row: 2; }
  .btn-stop  { grid-column: 2; grid-row: 2; }
  .btn-right { grid-column: 3; grid-row: 2; }
  .btn-down  { grid-column: 2; grid-row: 3; }

  /* shared autonomous panel */
  .auto-panel {
    display: none; width: 100%; max-width: 640px;
    border: 2px solid #7dd3fc; border-radius: 12px;
    padding: 20px; text-align: center; font-size: 0.95rem;
  }
  .auto-panel .icon { font-size: 2rem; display: block; margin-bottom: 8px; }

  #auto-overlay { background: #172033; color: #7dd3fc; }

  /* cone panel */
  #cone-overlay { background: #1f0f00; border-color: #fb923c; color: #fb923c; }
  #cone-state-badge {
    display: inline-block; margin-top: 10px;
    padding: 4px 16px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.06em;
    background: #2a1400; color: #fb923c; border: 1px solid #fb923c;
  }
  #cone-state-badge.searching { color: #86efac; border-color: #86efac; background: #0d1f0d; }
  #cone-state-badge.tracking  { color: #7dd3fc; border-color: #7dd3fc; background: #0d1a2a; }
  #cone-state-badge.arrived   { color: #fbbf24; border-color: #fbbf24; background: #1f1500; }
  #cone-found-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; background: #334155;
    margin-left: 8px; vertical-align: middle;
    transition: background 0.2s;
  }
  #cone-found-dot.found { background: #fb923c; box-shadow: 0 0 6px #fb923c; }

  .speed-note {
    font-size: 0.75rem; color: #94a3b8; margin-top: 8px;
  }

  /* ── RGB picker mode ── */
  #rgb-overlay { background: #0f1a0f; border-color: #4ade80; color: #4ade80; }
  #rgb-overlay .icon { font-size: 2rem; display: block; margin-bottom: 6px; }
  #rgb-readout {
    display: flex; align-items: center; justify-content: center;
    gap: 12px; margin-top: 10px; flex-wrap: wrap;
  }
  #rgb-swatch {
    width: 48px; height: 48px; border-radius: 8px;
    border: 2px solid #334155; background: #000; flex-shrink: 0;
  }
  #rgb-values {
    font-family: monospace; font-size: 0.95rem;
    line-height: 1.7; text-align: left; color: #e2e8f0;
  }
  .rgb-r { color: #f87171; }
  .rgb-g { color: #86efac; }
  .rgb-b { color: #7dd3fc; }
  #rgb-instructions {
    font-size: 0.78rem; color: #64748b; margin-top: 8px;
  }
  /* crosshair drawn on top of the camera image */
  #rgb-crosshair {
    display: none;
    position: absolute;
    pointer-events: none;
    transform: translate(-50%, -50%);
    width: 20px; height: 20px;
  }
  #rgb-crosshair::before,
  #rgb-crosshair::after {
    content: '';
    position: absolute;
    background: #4ade80;
  }
  #rgb-crosshair::before { width: 2px; height: 100%; left: 50%; top: 0; transform: translateX(-50%); }
  #rgb-crosshair::after  { width: 100%; height: 2px; top: 50%; left: 0; transform: translateY(-50%); }
</style>
</head>
<body>

<h1>🤖 Rover Control</h1>

<div id="camera-wrap">
  <img id="camera-feed" src="/video" alt="Camera feed">
  <div id="cam-badge">● LIVE</div>
</div>

<div id="status-bar">
  <div class="stat-row">
    <span class="stat-label">Distance</span>
    <span class="stat-value" id="dist-value">-- m</span>
  </div>
  <div class="stat-row">
    <span class="stat-label">Action</span>
    <span class="stat-value" id="action-value">Idle</span>
  </div>
  <div id="obstacle-warning">⚠ OBSTACLE — can't go forward!</div>
  <div id="free-warning">⚡ FREE MODE — obstacle sensor disabled</div>
</div>

<div class="slider-row">
  <label>Speed</label>
  <input type="range" id="speed-slider" min="1" max="99" value="50" step="1">
  <span id="speed-display">50%</span>
</div>

<div class="slider-row">
  <label>Stop Distance</label>
  <input type="range" id="stop-dist-slider" min="10" max="100" value="40" step="1">
  <span id="stop-dist-display">0.40 m</span>
</div>

<div class="slider-row">
  <label>Obstacle Brake Distance</label>
  <input type="range" id="obs-dist-slider" min="10" max="150" value="50" step="1">
  <span id="obs-dist-display">0.50 m</span>
</div>

<div class="mode-toggle">
  <button class="mode-btn active" id="btn-m0" onclick="setMode(0)">Manual + Brake</button>
  <button class="mode-btn"        id="btn-m1" onclick="setMode(1)">Auto Avoid</button>
  <button class="mode-btn"        id="btn-m2" onclick="setMode(2)">Manual Free</button>
  <button class="mode-btn"        id="btn-m3" onclick="setMode(3)">🔶 Colored Object Finder</button>
  <button class="mode-btn"        id="btn-m4" onclick="setMode(4)">🎨 RGB Picker</button>
</div>

<!-- Manual d-pad -->
<div class="dpad" id="dpad">
  <button class="btn btn-up"             id="btn-w"   onclick="cmd('forward')">▲</button>
  <button class="btn btn-left"           id="btn-a"   onclick="cmd('left')">◀</button>
  <button class="btn btn-stop stop-btn"  id="btn-spc" onclick="cmd('stop')">■</button>
  <button class="btn btn-right"          id="btn-d"   onclick="cmd('right')">▶</button>
  <button class="btn btn-down"           id="btn-s"   onclick="cmd('backward')">▼</button>
</div>

<!-- Auto-avoid overlay -->
<div class="auto-panel" id="auto-overlay">
  <span class="icon">🔄</span>
  Rover is avoiding obstacles autonomously.<br>Switch to a manual mode to take control.
</div>

<!-- Cone-finder overlay -->
<div class="auto-panel" id="cone-overlay">
  <span class="icon">🔶</span>
  Colored Object Finder active
  <span id="cone-found-dot" title="Orange detected"></span><br>
  <span id="cone-state-badge">SEARCHING</span>
  <div class="speed-note">All speeds controlled by the speed slider above.</div>
</div>

<!-- RGB picker overlay -->
<div class="auto-panel" id="rgb-overlay">
  <span class="icon">🎨</span>
  Hover to sample &mdash; <strong>click</strong> to lock &amp; set as cone target
  <div id="rgb-readout">
    <div id="rgb-swatch"></div>
    <div id="rgb-values">
      <span class="rgb-r">R: --</span><br>
      <span class="rgb-g">G: --</span><br>
      <span class="rgb-b">B: --</span>
    </div>
  </div>
  <div id="rgb-instructions">Move your cursor over the live feed above</div>
  <div id="rgb-locked" style="display:none; margin-top:10px;">
    <div style="font-size:0.8rem; color:#94a3b8; margin-bottom:6px;">Locked sample</div>
    <div style="display:flex; align-items:center; justify-content:center; gap:10px;">
      <div id="rgb-locked-swatch" style="width:36px;height:36px;border-radius:6px;border:2px solid #334155;"></div>
      <span id="rgb-locked-values" style="font-family:monospace; font-size:0.85rem; text-align:left; line-height:1.6;"></span>
    </div>
    <button id="btn-set-color" onclick="setTargetColor()"
      style="margin-top:10px; padding:8px 22px; background:#4ade80; color:#0a1f0a;
             border:none; border-radius:8px; font-weight:700; font-size:0.85rem; cursor:pointer;">
      🎯 Use as target color
    </button>
    <div id="rgb-set-status" style="font-size:0.78rem; color:#4ade80; margin-top:6px; min-height:1em;"></div>
  </div>
</div>
<!-- crosshair sits inside camera-wrap, moved by JS -->
<div id="rgb-crosshair"></div>

<script>
  let currentMode = 0;
  let tooClose    = false;

  async function cmd(action) {
    try {
      const res = await fetch('/command', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cmd: action})
      });
      updateUI(await res.json());
    } catch(e) { console.error(e); }
  }

  const slider       = document.getElementById('speed-slider');
  const speedDisplay = document.getElementById('speed-display');
  slider.addEventListener('input', () => {
    speedDisplay.textContent = slider.value + '%';
    fetch('/speed', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({speed: parseInt(slider.value) / 100})
    });
  });

  // ── Stop Distance slider (Colored Object Finder arrival threshold) ──
  const stopDistSlider  = document.getElementById('stop-dist-slider');
  const stopDistDisplay = document.getElementById('stop-dist-display');
  stopDistSlider.addEventListener('input', () => {
    const m = parseInt(stopDistSlider.value) / 100;
    stopDistDisplay.textContent = m.toFixed(2) + ' m';
    fetch('/set_distances', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({stop_dist: m})
    });
  });

  // ── Obstacle Brake Distance slider (Manual+Brake / Auto Avoid threshold) ──
  const obsDistSlider  = document.getElementById('obs-dist-slider');
  const obsDistDisplay = document.getElementById('obs-dist-display');
  obsDistSlider.addEventListener('input', () => {
    const m = parseInt(obsDistSlider.value) / 100;
    obsDistDisplay.textContent = m.toFixed(2) + ' m';
    fetch('/set_distances', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({obs_dist: m})
    });
  });

  function setMode(m) {
    currentMode = m;
    [0,1,2,3,4].forEach(i => {
      document.getElementById('btn-m'+i).className = 'mode-btn';
    });
    const cls = m === 2 ? 'active-free' : m === 3 ? 'active-cone' : m === 4 ? 'active-rgb' : 'active';
    document.getElementById('btn-m'+m).classList.add(cls);

    const isManual = m === 0 || m === 2;
    document.getElementById('dpad').style.display         = isManual ? 'grid'  : 'none';
    document.getElementById('auto-overlay').style.display = m === 1  ? 'block' : 'none';
    document.getElementById('cone-overlay').style.display = m === 3  ? 'block' : 'none';
    document.getElementById('rgb-overlay').style.display  = m === 4  ? 'block' : 'none';

    // Show/hide the crosshair on the camera wrap
    document.getElementById('rgb-crosshair').style.display = m === 4 ? 'block' : 'none';

    fetch('/mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode: m})
    });
  }

  function updateUI(data) {
    // Distance
    const dist = data.distance;
    document.getElementById('dist-value').textContent =
      dist > 4.0 ? '> 4.0 m' : dist.toFixed(2) + ' m';

    // Warnings
    tooClose = data.too_close;
    const isFree = currentMode === 2;
    document.getElementById('obstacle-warning').style.display =
      (!isFree && currentMode === 0 && tooClose) ? 'block' : 'none';
    document.getElementById('free-warning').style.display =
      isFree ? 'block' : 'none';
    document.getElementById('btn-w').classList.toggle('disabled-btn', !isFree && currentMode === 0 && tooClose);

    // Action
    document.getElementById('action-value').textContent = data.action || 'Idle';

    // Cone finder state
    if (data.cone_state !== undefined) {
      const state  = data.cone_state;
      const badge  = document.getElementById('cone-state-badge');
      const dot    = document.getElementById('cone-found-dot');
      const labels = {searching: 'SEARCHING…', tracking: 'TRACKING', arrived: 'ARRIVED — spinning'};
      badge.textContent = labels[state] || state.toUpperCase();
      badge.className   = state;   // matches CSS classes
      badge.id          = 'cone-state-badge';
      dot.classList.toggle('found', !!data.cone_found);
    }
  }

  // ── RGB picker ────────────────────────────────────────────
  const camWrap   = document.getElementById('camera-wrap');
  const camFeed   = document.getElementById('camera-feed');
  const crosshair = document.getElementById('rgb-crosshair');
  const rgbCanvas = document.createElement('canvas');
  const rgbCtx    = rgbCanvas.getContext('2d', {willReadFrequently: true});

  let lockedR = null, lockedG = null, lockedB = null;

  function samplePixel(e) {
    const rect   = camFeed.getBoundingClientRect();
    const xRel   = e.clientX - rect.left;
    const yRel   = e.clientY - rect.top;
    crosshair.style.left = xRel + 'px';
    crosshair.style.top  = yRel + 'px';
    const scaleX = camFeed.naturalWidth  / rect.width;
    const scaleY = camFeed.naturalHeight / rect.height;
    const px = Math.round(xRel * scaleX);
    const py = Math.round(yRel * scaleY);
    rgbCanvas.width  = camFeed.naturalWidth  || rect.width;
    rgbCanvas.height = camFeed.naturalHeight || rect.height;
    try {
      rgbCtx.drawImage(camFeed, 0, 0, rgbCanvas.width, rgbCanvas.height);
      const pixel = rgbCtx.getImageData(px, py, 1, 1).data;
      return {r: pixel[0], g: pixel[1], b: pixel[2], px, py};
    } catch(e) { return null; }
  }

  camWrap.addEventListener('mousemove', e => {
    if (currentMode !== 4) return;
    const s = samplePixel(e);
    if (!s) return;
    const hex = '#' + [s.r,s.g,s.b].map(v => v.toString(16).padStart(2,'0')).join('');
    document.getElementById('rgb-swatch').style.background = hex;
    document.querySelector('.rgb-r').textContent = 'R: ' + s.r;
    document.querySelector('.rgb-g').textContent = 'G: ' + s.g;
    document.querySelector('.rgb-b').textContent = 'B: ' + s.b;
    document.getElementById('rgb-instructions').textContent =
      'Position: (' + s.px + ', ' + s.py + ')  |  Hex: ' + hex.toUpperCase();
  });

  // Click to lock the sample
  camWrap.addEventListener('click', e => {
    if (currentMode !== 4) return;
    const s = samplePixel(e);
    if (!s) return;
    lockedR = s.r; lockedG = s.g; lockedB = s.b;
    const hex = '#' + [s.r,s.g,s.b].map(v => v.toString(16).padStart(2,'0')).join('');
    document.getElementById('rgb-locked').style.display = 'block';
    document.getElementById('rgb-locked-swatch').style.background = hex;
    document.getElementById('rgb-locked-values').innerHTML =
      '<span class="rgb-r">R: ' + s.r + '</span><br>' +
      '<span class="rgb-g">G: ' + s.g + '</span><br>' +
      '<span class="rgb-b">B: ' + s.b + '</span>';
    document.getElementById('rgb-set-status').textContent = '';
  });

  camWrap.addEventListener('mouseleave', () => {
    if (currentMode !== 4) return;
    crosshair.style.left = '-999px';
  });

  async function setTargetColor() {
    if (lockedR === null) return;
    const btn = document.getElementById('btn-set-color');
    const status = document.getElementById('rgb-set-status');
    btn.disabled = true;
    btn.textContent = 'Updating…';
    try {
      const res = await fetch('/set_color', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({r: lockedR, g: lockedG, b: lockedB})
      });
      const data = await res.json();
      status.textContent = '✓ Target updated — HSV hue ' + data.hue_range[0] + '–' + data.hue_range[1];
      status.style.color = '#4ade80';
    } catch(e) {
      status.textContent = '✗ Failed to update';
      status.style.color = '#f87171';
    }
    btn.disabled = false;
    btn.textContent = '🎯 Use as target color';
  }

  // Position crosshair inside camera-wrap
  camWrap.style.position = 'relative';
  camWrap.appendChild(crosshair);

  const keyMap = {w:'forward', a:'left', s:'backward', d:'right', ' ':'stop'};
  document.addEventListener('keydown', e => {
    if (currentMode === 1 || currentMode === 3) return;
    const action = keyMap[e.key.toLowerCase()];
    if (action) { e.preventDefault(); cmd(action); }
  });

  async function poll() {
    try { updateUI(await (await fetch('/status')).json()); } catch(e) {}
    setTimeout(poll, 200);
  }
  poll();
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────
current_action = 'stop'

@app.route('/')
def index():
    return HTML

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global current_action
    dist      = get_distance()
    too_close = dist <= OBSTACLE_THRESHOLD_M

    if DRIVE_MODE == 0:
        if too_close and current_action == 'forward':
            set_motors(0.0, 0.0)
            current_action = 'blocked'
        if not too_close and current_action == 'blocked':
            current_action = 'stop'

    with _cone_det_lock:
        cone_found = _cone_detection['found']

    return jsonify(
        distance   = round(dist, 2) if dist != float('inf') else 99.0,
        too_close  = too_close,
        mode       = DRIVE_MODE,
        action     = current_action,
        speed      = left_speed,
        cone_state = CONE_STATE,
        cone_found = cone_found,
    )

@app.route('/command', methods=['POST'])
def command():
    global current_action, DRIVE_MODE
    if DRIVE_MODE in (1, 3):
        dist = get_distance()
        return jsonify(distance=round(dist,2), too_close=dist<=OBSTACLE_THRESHOLD_M,
                       action=current_action, cone_state=CONE_STATE)

    data      = request.get_json()
    action    = data.get('cmd', 'stop')
    dist      = get_distance()
    too_close = dist <= OBSTACLE_THRESHOLD_M

    with state_lock:
        if action == 'forward':
            if DRIVE_MODE == 0 and too_close:
                set_motors(0.0, 0.0);  current_action = 'blocked'
            else:
                drive_forward();       current_action = 'forward'
        elif action == 'backward':
            drive_backward();  current_action = 'backward'
        elif action == 'left':
            drive_left();      current_action = 'left'
        elif action == 'right':
            drive_right();     current_action = 'right'
        elif action == 'stop':
            drive_stop();      current_action = 'stop'

    dist = get_distance()
    return jsonify(
        distance  = round(dist, 2) if dist != float('inf') else 99.0,
        too_close = dist <= OBSTACLE_THRESHOLD_M,
        action    = current_action,
        cone_state = CONE_STATE,
    )

@app.route('/speed', methods=['POST'])
def set_speed():
    global left_speed, right_speed
    spd = float(request.get_json().get('speed', 0.5))
    with state_lock:
        left_speed = right_speed = max(0.1, min(1.0, spd))
    return jsonify(speed=left_speed)

@app.route('/mode', methods=['POST'])
def set_mode():
    global DRIVE_MODE, current_action
    m = int(request.get_json().get('mode', 0))
    with state_lock:
        stop_auto()
        stop_cone()
        drive_stop()
        current_action = 'stop'
        DRIVE_MODE = m
        if m == 1:
            current_action = 'auto'
            start_auto()
        elif m == 3:
            current_action = 'cone-finder'
            start_cone()
    return jsonify(mode=DRIVE_MODE)


@app.route('/set_distances', methods=['POST'])
def set_distances():
    """Update stop distance and/or obstacle brake distance live from the sliders."""
    global CONE_STOP_DISTANCE_M, OBSTACLE_THRESHOLD_M
    data = request.get_json()
    with state_lock:
        if 'stop_dist' in data:
            CONE_STOP_DISTANCE_M  = float(data['stop_dist'])
        if 'obs_dist' in data:
            OBSTACLE_THRESHOLD_M  = float(data['obs_dist'])
    return jsonify(
        stop_dist = CONE_STOP_DISTANCE_M,
        obs_dist  = OBSTACLE_THRESHOLD_M,
    )


@app.route('/set_color', methods=['POST'])
def set_color():
    """
    Receives an RGB value from the browser picker, converts it to HSV,
    and updates CONE_HSV_LOWER/UPPER with a tolerance band around that hue.
    Saturation and value floors are kept generous to handle lighting variation.
    """
    global CONE_HSV_LOWER, CONE_HSV_UPPER
    data = request.get_json()
    r, g, b = int(data['r']), int(data['g']), int(data['b'])

    # Convert the single RGB pixel to HSV using OpenCV
    import numpy as np
    pixel = np.uint8([[[b, g, r]]])                     # OpenCV uses BGR
    hsv   = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Build a tolerance band around the sampled hue (+/- 10 degrees)
    HUE_TOL = 10
    h_lo = max(0,   h - HUE_TOL)
    h_hi = min(179, h + HUE_TOL)

    # Keep saturation/value floors high enough to avoid pale or dark noise
    s_lo = max(60,  s - 60)
    v_lo = max(60,  v - 80)

    with state_lock:
        CONE_HSV_LOWER = np.array([h_lo, s_lo, v_lo])
        CONE_HSV_UPPER = np.array([h_hi, 255,  255 ])

    return jsonify(
        h=h, s=s, v=v,
        hue_range=[h_lo, h_hi],
        lower=CONE_HSV_LOWER.tolist(),
        upper=CONE_HSV_UPPER.tolist(),
    )


if __name__ == '__main__':
    try:
        led.on()
        print("Rover server starting on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        stop_auto()
        stop_cone()
        cleanup()
