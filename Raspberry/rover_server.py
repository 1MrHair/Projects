# rover_server_test.py
# Web-based rover control server with USB camera stream.
# Run: python3 rover_server.py
# Then open http://<pi-ip>:5000 on any device on the same network.
#
# Dependencies:
#   pip install flask opencv-python --break-system-packages

import time
import threading
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
MODE_NAMES  = {0: "Manual + Brake", 1: "Auto Avoid"}

OBSTACLE_THRESHOLD_M = 0.50
AUTO_BACKUP_SPEED    = 0.1
AUTO_BACKUP_DURATION = 0.8
AUTO_TURN_SPEED      = 0.1
AUTO_TURN_DURATION   = 0.8

state_lock = threading.Lock()

# ── USB Camera ────────────────────────────────────────────────
# Change CAMERA_INDEX if the wrong device is picked (try 1, 2, …).
# Lower CAMERA_WIDTH/HEIGHT for smoother streaming over slower WiFi.
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

_camera      = cv2.VideoCapture(CAMERA_INDEX)
_camera.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
_camera_lock = threading.Lock()

def generate_frames():
    """Yield a continuous MJPEG stream from the USB camera."""
    import numpy as np
    blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype='uint8')
    blank[:] = 30  # dark grey placeholder when camera is unavailable
    while True:
        with _camera_lock:
            ok, frame = _camera.read()
        if not ok:
            frame = blank
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


# ── Motor / sensor functions ──────────────────────────────────
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


# ── Auto-avoid background thread ──────────────────────────────
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


# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
<title>Rover Control</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #0f1117;
    color: #e2e8f0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100svh;
    padding: 16px;
    gap: 16px;
  }

  h1 {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #7dd3fc;
  }

  /* ── Camera feed ── */
  #camera-wrap {
    width: 100%;
    max-width: 640px;
    background: #1e2433;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
    aspect-ratio: 4/3;
  }
  #camera-feed {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  #cam-badge {
    position: absolute;
    top: 8px; left: 8px;
    background: rgba(0,0,0,0.55);
    color: #86efac;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
    letter-spacing: 0.05em;
  }

  /* ── Status bar ── */
  #status-bar {
    width: 100%;
    max-width: 640px;
    background: #1e2433;
    border-radius: 12px;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 0.85rem;
  }
  .stat-row { display: flex; justify-content: space-between; align-items: center; }
  .stat-label { color: #94a3b8; }
  .stat-value { font-weight: 600; }
  #dist-value   { color: #7dd3fc; }
  #action-value { color: #86efac; }
  #obstacle-warning {
    color: #f87171;
    font-weight: 700;
    text-align: center;
    display: none;
  }

  /* ── Speed slider ── */
  .slider-row {
    width: 100%;
    max-width: 640px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.85rem;
  }
  .slider-row label { color: #94a3b8; white-space: nowrap; }
  input[type=range] { flex: 1; accent-color: #7dd3fc; height: 6px; }
  #speed-display { color: #7dd3fc; font-weight: 700; min-width: 38px; text-align: right; }

  /* ── Mode toggle ── */
  .mode-toggle {
    display: flex;
    gap: 8px;
    width: 100%;
    max-width: 640px;
  }
  .mode-btn {
    flex: 1;
    padding: 10px;
    border-radius: 8px;
    border: 2px solid #334155;
    background: #1e2433;
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }
  .mode-btn.active { border-color: #7dd3fc; color: #7dd3fc; background: #172033; }

  /* ── D-pad ── */
  .dpad {
    display: grid;
    grid-template-columns: repeat(3, 80px);
    grid-template-rows: repeat(3, 80px);
    gap: 8px;
    user-select: none;
  }
  .btn {
    border-radius: 12px;
    border: none;
    font-size: 1.6rem;
    cursor: pointer;
    background: #1e2433;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.1s, transform 0.08s;
    -webkit-tap-highlight-color: transparent;
    touch-action: manipulation;
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

  /* ── Auto overlay ── */
  #auto-overlay {
    display: none;
    width: 100%;
    max-width: 640px;
    background: #172033;
    border: 2px solid #7dd3fc;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    font-size: 0.95rem;
    color: #7dd3fc;
  }
  #auto-overlay span { font-size: 2rem; display: block; margin-bottom: 8px; }
</style>
</head>
<body>

<h1>🤖 Rover Control</h1>

<!-- Camera -->
<div id="camera-wrap">
  <img id="camera-feed" src="/video" alt="Camera feed">
  <div id="cam-badge">● LIVE</div>
</div>

<!-- Status -->
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
</div>

<!-- Speed -->
<div class="slider-row">
  <label>Speed</label>
  <input type="range" id="speed-slider" min="10" max="100" value="50" step="10">
  <span id="speed-display">50%</span>
</div>

<!-- Mode -->
<div class="mode-toggle">
  <button class="mode-btn active" id="btn-manual" onclick="setMode(0)">Manual + Brake</button>
  <button class="mode-btn"        id="btn-auto"   onclick="setMode(1)">Auto Avoid</button>
</div>

<!-- D-pad -->
<div class="dpad" id="dpad">
  <button class="btn btn-up"              id="btn-w" onclick="cmd('forward')">▲</button>
  <button class="btn btn-left"            id="btn-a" onclick="cmd('left')">◀</button>
  <button class="btn btn-stop  stop-btn"  id="btn-spc" onclick="cmd('stop')">■</button>
  <button class="btn btn-right"           id="btn-d" onclick="cmd('right')">▶</button>
  <button class="btn btn-down"            id="btn-s" onclick="cmd('backward')">▼</button>
</div>

<!-- Auto overlay -->
<div id="auto-overlay">
  <span>🔄</span>
  Rover is driving autonomously.<br>Switch to Manual to take control.
</div>

<script>
  let currentMode = 0;
  let tooClose    = false;

  async function cmd(action) {
    try {
      const res  = await fetch('/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd: action })
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
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed: parseInt(slider.value) / 100 })
    });
  });

  function setMode(m) {
    currentMode = m;
    document.getElementById('btn-manual').classList.toggle('active', m === 0);
    document.getElementById('btn-auto').classList.toggle('active',   m === 1);
    document.getElementById('dpad').style.display         = m === 0 ? 'grid' : 'none';
    document.getElementById('auto-overlay').style.display = m === 1 ? 'block' : 'none';
    fetch('/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: m })
    });
  }

  function updateUI(data) {
    const dist = data.distance;
    document.getElementById('dist-value').textContent =
      dist > 4.0 ? '> 4.0 m' : dist.toFixed(2) + ' m';

    tooClose = data.too_close;
    document.getElementById('obstacle-warning').style.display = tooClose ? 'block' : 'none';
    document.getElementById('btn-w').classList.toggle('disabled-btn', tooClose);
    document.getElementById('action-value').textContent = data.action || 'Idle';
  }

  const keyMap = { w:'forward', a:'left', s:'backward', d:'right', ' ':'stop' };
  document.addEventListener('keydown', e => {
    if (currentMode !== 0) return;
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
    """MJPEG stream from the USB camera."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global current_action
    dist      = get_distance()
    too_close = dist <= OBSTACLE_THRESHOLD_M

    # Enforce obstacle limit on every poll tick — if the rover is
    # already moving forward and something enters the threshold zone,
    # cut the motors immediately without waiting for a new /command.
    if too_close and current_action == 'forward':
        set_motors(0.0, 0.0)
        current_action = 'blocked'

    # Clear the blocked state automatically once the path is free again,
    # so the driver can go forward again without re-pressing the button.
    if not too_close and current_action == 'blocked':
        current_action = 'stop'

    return jsonify(
        distance  = round(dist, 2) if dist != float('inf') else 99.0,
        too_close = too_close,
        mode      = DRIVE_MODE,
        action    = current_action,
        speed     = left_speed,
    )

@app.route('/command', methods=['POST'])
def command():
    global current_action, DRIVE_MODE
    if DRIVE_MODE == 1:
        dist = get_distance()
        return jsonify(distance=round(dist,2), too_close=dist<=OBSTACLE_THRESHOLD_M, action='auto')

    data      = request.get_json()
    action    = data.get('cmd', 'stop')
    dist      = get_distance()
    too_close = dist <= OBSTACLE_THRESHOLD_M

    with state_lock:
        if action == 'forward':
            if too_close:
                set_motors(0.0, 0.0)
                current_action = 'blocked'
            else:
                drive_forward()
                current_action = 'forward'
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
        DRIVE_MODE = m
        if m == 1:
            current_action = 'auto'
            start_auto()
        else:
            stop_auto()
            drive_stop()
            current_action = 'stop'
    return jsonify(mode=DRIVE_MODE)


if __name__ == '__main__':
    try:
        led.on()
        print("Rover server starting on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        stop_auto()
        cleanup()
