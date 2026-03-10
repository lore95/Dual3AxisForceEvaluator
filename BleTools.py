import re
import time
import threading
import csv
import glob
import sys
import os
import signal
from datetime import datetime

import numpy as np
import serial

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

# =========================
# Constants
# =========================

MAX_DEVICES = 4
SAVE_DIR = "Readings"
PLOT_HISTORY = 300
BAUDRATE = 9600
SYNC_RE = re.compile(r'^SYNC:(\d+)\s*$')

# =========================
# Calibration configuration
# =========================
# Placeholder calibration settings.
# Replace these when your calibration data arrives.
#
# Current behavior:
#   force_N = CAL_GAIN[channel] * raw_value + CAL_OFFSET[channel]
#
# For now, these are identity mappings.
CAL_GAIN = {
    "V1": 1.0,
    "V2": 1.0,
    "V3": 1.0,
}

CAL_OFFSET = {
    "V1": 0.0,
    "V2": 0.0,
    "V3": 0.0,
}


def calibrate_channel_force(raw_value, channel_name):
    """
    Convert one raw sensor value into Newtons.

    Placeholder version:
        force_N = gain * raw_value + offset

    Replace this later with your real calibration logic once you receive
    the calibration certificate / calibration file.

    Examples of future replacements:
      - linear fit from calibration CSV
      - piecewise interpolation
      - per-channel polynomial fit
      - TEDS-derived conversion
    """
    gain = CAL_GAIN.get(channel_name, 1.0)
    offset = CAL_OFFSET.get(channel_name, 0.0)
    return float(gain * raw_value + offset)


def load_calibration_from_file(filepath):
    """
    Optional helper for later use.

    Expected CSV example:
        channel,gain,offset
        V1,0.123,0.0
        V2,0.121,-0.01
        V3,0.119,0.005

    You do not need to use this yet. It is here so you can plug in your
    calibration file later without restructuring the program.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Calibration file not found: {filepath}")

    loaded_gain = {}
    loaded_offset = {}

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        required = {"channel", "gain", "offset"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Calibration CSV must contain headers: {sorted(required)}"
            )

        for row in reader:
            ch = row["channel"].strip()
            loaded_gain[ch] = float(row["gain"])
            loaded_offset[ch] = float(row["offset"])

    for ch in ("V1", "V2", "V3"):
        if ch in loaded_gain:
            CAL_GAIN[ch] = loaded_gain[ch]
        if ch in loaded_offset:
            CAL_OFFSET[ch] = loaded_offset[ch]

    print("Loaded calibration:")
    for ch in ("V1", "V2", "V3"):
        print(f"  {ch}: force_N = {CAL_GAIN[ch]} * raw + {CAL_OFFSET[ch]}")


# =========================
# Serial port conf and opening
# =========================

def list_serial_devices():
    candidates = []

    # macOS
    candidates += glob.glob('/dev/tty.usbmodem*')

    # Linux
    candidates += glob.glob('/dev/ttyACM*')
    candidates += glob.glob('/dev/ttyUSB*')

    # Windows
    candidates += [f"COM{i}" for i in range(1, 257)]

    uniq, seen = [], set()
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def try_open_serial(port):
    try:
        return serial.Serial(
            port=port,
            baudrate=BAUDRATE,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1
        )
    except Exception:
        return None


def get_board_time_offset(ser, timeout=2.0):
    """
    Sync one board to the PC clock.

    Protocol:
      - PC sends 's'
      - Board replies: SYNC:<micros_from_micros()>

    Compute:
        offset_s = t_host_recv - board_micros/1e6

    Then:
        host_time ≈ board_time_s + offset_s
    """
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    ser.write(b's')
    ser.flush()

    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if not line:
            continue

        m = SYNC_RE.match(line)
        if m:
            t_host_recv = time.perf_counter()
            board_us = int(m.group(1))
            board_s = board_us / 1_000_000.0
            offset_s = t_host_recv - board_s
            print(f"Offset for board ({ser.port}): host ≈ board + {offset_s:.6f} s")
            return offset_s

    raise TimeoutError(f"No SYNC response from board on {ser.port} within timeout.")


# =========================
# Device context
# =========================

class DeviceContext:
    """
    One device:
      - serial connection
      - reader thread
      - raw/force buffer
      - one subplot with 3 channel-force lines
    """

    def __init__(self, port, ser, ax_channels, offset_s):
        self.port = port
        self.port_sanitized = port.replace("/", "_").replace("\\", "_").replace(":", "_")
        self.ser = ser
        self.offset_s = offset_s

        self.lock = threading.Lock()
        self.buffer = []
        self.index = 0

        self.stop_event = threading.Event()
        self.saved_once = threading.Event()
        self.reader_thread = None

        self.ax_channels = ax_channels
        (self.line_v1,) = self.ax_channels.plot([], [], color='r', label=f"{self.port} V1")
        (self.line_v2,) = self.ax_channels.plot([], [], color='g', label=f"{self.port} V2")
        (self.line_v3,) = self.ax_channels.plot([], [], color='b', label=f"{self.port} V3")

        self.ax_channels.set_title(f"{self.port} — Channel Forces")
        self.ax_channels.set_xlabel("Host Time (s)")
        self.ax_channels.set_ylabel("Force (N)")
        self.ax_channels.grid(True)
        self.ax_channels.legend(loc="upper right")

    def start_reader(self):
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()

    def _read_loop(self):
        pattern = re.compile(
            r'Time:(-?\d+),V1:(-?\d+(?:\.\d+)?),'
            r'V2:(-?\d+(?:\.\d+)?),V3:(-?\d+(?:\.\d+)?)'
        )

        while not self.stop_event.is_set():
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                match = pattern.match(line)
                if not match:
                    continue

                t_us = int(match.group(1))
                t_host = (t_us / 1_000_000.0) + self.offset_s

                v1_raw = float(match.group(2))
                v2_raw = float(match.group(3))
                v3_raw = float(match.group(4))

                f1 = round(calibrate_channel_force(v1_raw, "V1"), 6)
                f2 = round(calibrate_channel_force(v2_raw, "V2"), 6)
                f3 = round(calibrate_channel_force(v3_raw, "V3"), 6)

                with self.lock:
                    self.buffer.append((
                        t_host,
                        self.index,
                        v1_raw, v2_raw, v3_raw,
                        f1, f2, f3
                    ))
                    self.index += 1

            except Exception as e:
                print(f"[{self.port}] Read error: {e}")
                continue

    def update_channels(self):
        with self.lock:
            if len(self.buffer) < 2:
                return

            recent = self.buffer[-PLOT_HISTORY:]
            x = [r[0] for r in recent]
            y1 = [r[5] for r in recent]
            y2 = [r[6] for r in recent]
            y3 = [r[7] for r in recent]

        self.line_v1.set_data(x, y1)
        self.line_v2.set_data(x, y2)
        self.line_v3.set_data(x, y3)

        self.ax_channels.relim()
        self.ax_channels.autoscale_view()

    def get_force_series(self):
        with self.lock:
            if not self.buffer:
                return [], [], []
            f1 = [r[5] for r in self.buffer]
            f2 = [r[6] for r in self.buffer]
            f3 = [r[7] for r in self.buffer]
        return f1, f2, f3

    def finalize_and_save(self):
        if self.saved_once.is_set():
            return
        self.saved_once.set()

        self.stop_event.set()
        try:
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=2)
        except Exception:
            pass

        with self.lock:
            rows = list(self.buffer)

        try:
            os.makedirs(SAVE_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/session_{ts}_{self.port_sanitized}.csv"

            with open(filename, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "host_time_s",
                    "sample_index",
                    "v1_raw", "v2_raw", "v3_raw",
                    "v1_force_N", "v2_force_N", "v3_force_N"
                ])
                for row in rows:
                    t_host, idx, v1_raw, v2_raw, v3_raw, f1, f2, f3 = row
                    w.writerow([t_host, idx, v1_raw, v2_raw, v3_raw, f1, f2, f3])

            print(f"[{self.port}] Saved {len(rows)} rows to {filename}")
        except Exception as e:
            print(f"[{self.port}] Failed to save CSV: {e}")

        try:
            self.ser.close()
        except Exception:
            pass


# =========================
# Global figure & devices
# =========================

DEVICES = []
FIG = None

CENTER_AX = None
CENTER_LINE_V1 = None
CENTER_LINE_V2 = None
CENTER_LINE_V3 = None

BTN_STOP = None
BTN_AX = None
_FINALIZED_ALL = threading.Event()


def get_figure():
    return FIG


# =========================
# Graph UI and setup
# =========================

def setup_devices_and_figure():
    """
    Discover boards, sync them, create the figure layout, and start readers.
    """
    global FIG, DEVICES, BTN_STOP, BTN_AX
    global CENTER_AX, CENTER_LINE_V1, CENTER_LINE_V2, CENTER_LINE_V3

    ports = list_serial_devices()
    opened_pairs = []

    for p in ports:
        if len(opened_pairs) >= MAX_DEVICES:
            break
        ser = try_open_serial(p)
        if ser:
            opened_pairs.append((p, ser))

    if not opened_pairs:
        print("No serial devices found/available.")
        sys.exit(1)

    offset_by_port = {}
    valid_pairs = []

    for port, ser in opened_pairs:
        try:
            offset_s = get_board_time_offset(ser, timeout=2.0)
            offset_by_port[port] = offset_s
            valid_pairs.append((port, ser))
        except TimeoutError as e:
            print(f"[{port}] sync failed: {e}")
            try:
                ser.close()
            except Exception:
                pass

    if not valid_pairs:
        print("No boards successfully synced. Exiting.")
        sys.exit(1)

    FIG = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 3, figure=FIG, width_ratios=[1, 1.4, 1], height_ratios=[1, 1.2, 1])

    CENTER_AX = FIG.add_subplot(gs[:, 1])
    (CENTER_LINE_V1,) = CENTER_AX.plot([], [], color='r', label="Avg V1 across devices")
    (CENTER_LINE_V2,) = CENTER_AX.plot([], [], color='g', label="Avg V2 across devices")
    (CENTER_LINE_V3,) = CENTER_AX.plot([], [], color='b', label="Avg V3 across devices")

    CENTER_AX.set_title("AVERAGE CHANNEL FORCE — ALL DEVICES")
    CENTER_AX.set_xlabel("Sample Index (aligned min length)")
    CENTER_AX.set_ylabel("Force (N)")
    CENTER_AX.grid(True)
    CENTER_AX.legend(loc="upper right")

    corner_slots = [(0, 0), (0, 2), (2, 0), (2, 2)]

    for i, (port, ser) in enumerate(valid_pairs):
        if i >= len(corner_slots):
            break

        r, c = corner_slots[i]
        ax_channels = FIG.add_subplot(gs[r, c])

        ctx = DeviceContext(
            port=port,
            ser=ser,
            ax_channels=ax_channels,
            offset_s=offset_by_port[port]
        )
        DEVICES.append(ctx)
        ctx.start_reader()
        print(f"[{port}] Reader started.")

    BTN_AX = FIG.add_axes([0.40, 0.02, 0.20, 0.06])
    BTN_STOP = Button(BTN_AX, "Stop All")

    def _on_stop_all(_evt=None):
        try:
            BTN_STOP.label.set_text("Stopping…")
            FIG.canvas.draw_idle()
        except Exception:
            pass
        finalize_all_and_exit()

    BTN_STOP.on_clicked(_on_stop_all)

    FIG.canvas.mpl_connect(
        'key_press_event',
        lambda e: finalize_all_and_exit() if e.key == 'escape' else None
    )
    FIG.canvas.mpl_connect('close_event', lambda _evt: finalize_all_and_exit())


# =========================
# Animation update
# =========================

def update_all(_frame):
    """
    Update device plots and center plot.
    Center plot shows average force per channel across devices.
    """
    for ctx in DEVICES:
        ctx.update_channels()

    if not DEVICES:
        return ()

    series_list = []
    for ctx in DEVICES:
        f1, f2, f3 = ctx.get_force_series()
        if f1 and f2 and f3:
            series_list.append((f1, f2, f3))

    if not series_list:
        return ()

    min_len = min(len(s[0]) for s in series_list)
    if min_len < 2:
        return ()

    avg_v1 = [
        float(np.mean([dev[0][i] for dev in series_list]))
        for i in range(min_len)
    ]
    avg_v2 = [
        float(np.mean([dev[1][i] for dev in series_list]))
        for i in range(min_len)
    ]
    avg_v3 = [
        float(np.mean([dev[2][i] for dev in series_list]))
        for i in range(min_len)
    ]

    x = list(range(min_len))

    CENTER_LINE_V1.set_data(x[-PLOT_HISTORY:], avg_v1[-PLOT_HISTORY:])
    CENTER_LINE_V2.set_data(x[-PLOT_HISTORY:], avg_v2[-PLOT_HISTORY:])
    CENTER_LINE_V3.set_data(x[-PLOT_HISTORY:], avg_v3[-PLOT_HISTORY:])

    CENTER_AX.relim()
    CENTER_AX.autoscale_view()

    return ()


# =========================
# Finalize all & exit
# =========================

def finalize_all_and_exit():
    if _FINALIZED_ALL.is_set():
        return
    _FINALIZED_ALL.set()

    for ctx in DEVICES:
        try:
            ctx.finalize_and_save()
        except Exception as e:
            print(f"[{ctx.port}] finalize error: {e}")

    try:
        plt.close(FIG)
    except Exception:
        pass

    try:
        plt.close('all')
    except Exception:
        pass


# =========================
# Graceful stop from outside
# =========================

def _handle_term(_signum, _frame):
    try:
        finalize_all_and_exit()
    finally:
        os._exit(0)


def register_signal_handlers():
    signal.signal(getattr(signal, "SIGTERM", signal.SIGINT), _handle_term)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handle_term)