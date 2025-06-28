import torch
import numpy as np
import cv2
from ultralytics import YOLO
from mss import mss
import math
import threading
import keyboard
import mouse  # Import the mouse library
import time
import ctypes
from ctypes import wintypes

# Initialize the YOLO model with YOLOv10
model = YOLO('yolo10.pt')  # Adjust the model path and variant

# Check CUDA availability
if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

# Constants
CONFIDENCE_THRESHOLD = 0.001
PLAYER_CLASS_ID = 7  # Class ID for 'player'
RESOLUTION_X, RESOLUTION_Y = 2560, 1440
CENTER_X, CENTER_Y = RESOLUTION_X // 2, RESOLUTION_Y // 2

# Exclusion areas loaded from your previously saved data
excluded_areas = [
    ((716, 3), (1930, 87)),
    ((1313, 47), (1313, 47)),
    ((1313, 47), (1312, 46)),
    ((1306, 51), (1306, 51))
]

# Feature toggle for auto-aiming and error handling
auto_aim_active = False
app_paused = False

# Define necessary structures for SendInput
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]

class INPUT(ctypes.Structure):
    class _InputUnion(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_iu",)
    _fields_ = [("type", wintypes.DWORD), ("_iu", _InputUnion)]

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_MOVE_RELATIVE = 0x0001

def send_input(dx, dy):
    """Move the mouse using the Windows API SendInput for relative movement."""
    mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE_RELATIVE, time=0, dwExtraInfo=None)
    inp = INPUT(type=INPUT_MOUSE, mi=mi)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

def get_screen():
    """Capture the screen and resize it to match the expected resolution."""
    with mss() as sct:
        monitor = sct.monitors[1]  # Use the first monitor
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)  # Convert color from BGRA to RGB
        return cv2.resize(frame, (RESOLUTION_X, RESOLUTION_Y))  # Resize the frame

def is_in_excluded_area(x, y):
    """Check if a given point is inside any of the defined exclusion areas."""
    for (x1, y1), (x2, y2) in excluded_areas:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def find_closest_player(detections):
    """Find the closest player based on the bounding boxes and their center coordinates."""
    closest_distance = float('inf')
    closest_bbox = None

    for box in detections:
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        conf = box.conf.item()
        class_id = int(box.cls.item())

        if class_id == PLAYER_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
            bbox_center_x, bbox_center_y = (x_min + x_max) / 2, (y_min + y_max) / 2

            # Skip if the detected player is in an excluded area
            if is_in_excluded_area(bbox_center_x, bbox_center_y):
                continue

            distance = calculate_distance(CENTER_X, CENTER_Y, bbox_center_x, bbox_center_y)

            if distance < closest_distance:
                closest_distance = distance
                closest_bbox = (bbox_center_x, bbox_center_y)

    return closest_bbox

def move_mouse_to_target(target_x, target_y):
    """Move the mouse instantly towards the target."""
    dx = target_x - CENTER_X
    dy = target_y - CENTER_Y

    # Send relative movement immediately
    send_input(int(dx), int(dy))

def perform_aiming_action():
    """Perform a single aiming action when called."""
    frame = get_screen()
    frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255

    if torch.cuda.is_available():
        frame_tensor = frame_tensor.cuda()

    with torch.no_grad():
        results = model(frame_tensor.unsqueeze(0))

    if isinstance(results, list) and len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            closest_player = find_closest_player(result.boxes)

            if closest_player:
                aim_x, aim_y = closest_player
                move_mouse_to_target(aim_x, aim_y)

def detect_and_aim():
    """Capture screen, run detection, and aim at the closest player if auto-aim is active."""
    global auto_aim_active, app_paused

    while True:
        if auto_aim_active and not app_paused:
            perform_aiming_action()

        time.sleep(0.00001)  # Reduced delay for faster updates

def on_right_click():
    """Handler for the right mouse click event."""
    perform_aiming_action()

def toggle_auto_aim():
    """Toggle auto-aim when the F8 hotkey is pressed."""
    global auto_aim_active
    auto_aim_active = not auto_aim_active
    status = "activated" if auto_aim_active else "deactivated"
    print(f"Auto-aim {status}")

def toggle_app_pause():
    """Toggle app pause when the F9 hotkey is pressed."""
    global app_paused
    app_paused = not app_paused
    status = "paused" if app_paused else "resumed"
    print(f"App {status}")

def hotkey_listener():
    """Run hotkey listener in a separate thread."""
    keyboard.add_hotkey('F8', toggle_auto_aim)
    keyboard.add_hotkey('F9', toggle_app_pause)
    mouse.on_right_click(on_right_click)  # Listen for right mouse clicks
    keyboard.wait()  # Keeps the listener running

# Start the hotkey listener in a separate thread
hotkey_thread = threading.Thread(target=hotkey_listener, daemon=True)
hotkey_thread.start()

# Start the detection and aiming loop
try:
    detect_and_aim()
except KeyboardInterrupt:
    print("Program interrupted by user.")
