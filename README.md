# 🎯 YOLOv10 Auto-Aim Bot (Real-Time Detection with Hotkeys)

This project uses a **YOLOv10** model to implement an advanced auto-aiming system. It performs real-time object detection on your screen, identifies specific targets (e.g., "players"), and automatically moves your mouse cursor to aim at the closest one — unless they fall within defined exclusion zones. The app includes hotkeys to toggle auto-aim and pause functionality.

---

## 🔧 Features

- ⚡ Real-time object detection using YOLOv10 (`yolo10.pt`)
- 🖱️ Auto-aims at the nearest detected player (Class ID = 7)
- ❌ Ignores detections in specific "excluded" screen areas
- 🎮 Instant mouse aiming using Windows API
- ⌨️ Hotkey support:
  - `F8` → Toggle auto-aim on/off
  - `F9` → Pause/resume the application
  - Right-click → Perform a single manual aim

---

## 📦 Requirements

- Python 3.8+
- `torch`
- `numpy`
- `opencv-python`
- `ultralytics`
- `mss`
- `keyboard`
- `mouse`

Install all dependencies:

```bash
pip install torch numpy opencv-python ultralytics mss keyboard mouse
```

---

## 🛠 Setup

1. **Model File**: Place your YOLOv10 model (e.g., `yolo10.pt`) in the project directory.
2. **Screen Resolution**: Update the `RESOLUTION_X` and `RESOLUTION_Y` constants if your monitor isn't 2560×1440.
3. **Class ID**: Set `PLAYER_CLASS_ID` to match the class ID of your target object (default: 7).
4. **Exclusion Zones**: Modify the `excluded_areas` list if you want to ignore detections in specific screen regions (like HUDs or overlays).

---

## ▶️ Running the App

```bash
python auto_aim_yolo.py
```

- The app will launch and begin monitoring screen contents.
- Press `F8` to toggle continuous auto-aim.
- Press `F9` to pause/resume the app.
- Right-click to manually aim at the nearest target.

---

## 📌 Behavior Overview

- **Detection**: Uses YOLOv10 to detect objects on your screen.
- **Filtering**: Ignores targets with low confidence or in excluded zones.
- **Targeting**: Calculates the closest valid target to screen center.
- **Aiming**: Uses `ctypes` + Windows API to move the mouse to the target's coordinates.

---

## ⚠️ Disclaimer

This software is intended for **educational and research purposes only**. Using it in multiplayer games or to gain an unfair advantage may violate terms of service and local laws. Use responsibly.

---

## 🧠 Code Overview

```python
# Real-time screen capture and YOLOv10 inference
frame = get_screen()
frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255

# Object detection
results = model(frame_tensor.unsqueeze(0))

# Auto-aim targeting
if auto_aim_active:
    perform_aiming_action()

# Mouse movement via Windows API
send_input(dx, dy)

# Hotkeys
keyboard.add_hotkey('F8', toggle_auto_aim)
keyboard.add_hotkey('F9', toggle_app_pause)
mouse.on_right_click(on_right_click)
```

---

## 📫 Contact

Found an issue or want to improve the project? Feel free to open an issue or submit a pull request.
