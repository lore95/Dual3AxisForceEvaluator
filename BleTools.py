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
# Channel / axis mapping
# =========================

CHANNEL_TO_AXIS = {
    "V1": "Fx",
    "V2": "Fy",
    "V3": "Fz",
}

# =========================
# ADC / calibration constants
# =========================
# Assumptions:
# - ADC is 24-bit signed
# - Vref = 1.2 V
# - PGA gain = 128
# - Incoming V1/V2/V3 serial values are raw ADC counts

VREF = 1.2
ADC_GAIN = 128
ADC_BITS = 24
ADC_MAX_SIGNED = (1 << (ADC_BITS - 1)) - 1  # 8388607

AXIS_CAL = {
    "V1": 0.980,
    "V2": 1.004,
    "V3": 0.724,
}


def raw_to_voltage(raw, vref=VREF, gain=ADC_GAIN):
    """
    Convert signed raw ADC count to ADC input voltage in volts.
    """
    signed = int(raw)
    full_scale_input = vref / gain
    return (signed / ADC_MAX_SIGNED) * full_scale_input


def voltage_to_newtons(voltage, channel_name):
    """
    Apply the provided calibration formula:

    V1: ((IN(0))*1000/4000)/((0.980/5000)*10)
    V2: ((IN(1))*1000/4000)/((1.004/5000)*10)
    V3: ((IN(2))*1000/4000)/((0.724/5000)*10)
    """
    k = AXIS_CAL[channel_name]
    return ((voltage * 1000.0 / 4000.0) / ((k / 5000.0) * 10.0))


def raw_to_newtons(raw, channel_name):
    voltage = raw_to_voltage(raw)
    return voltage_to_newtons(voltage, channel_name)


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
      - live plotting buffer
      - recording buffer
      - one subplot with 3 channel-force lines
    """

    def __init__(self, port, ser, ax_channels, offset_s):
        self.port = port
        self.port_sanitized = port.replace("/", "_").replace("\\", "_").replace(":", "_")
        self.ser = ser
        self.offset_s = offset_s

        self.lock = threading.Lock()

        # live data for plotting
        self.live_buffer = []

        # current recording chunk only
        self.record_buffer = []

        self.index = 0
        self.is_recording = False
        self.save_counter = 0

        self.stop_event = threading.Event()
        self.finalized_once = threading.Event()
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

    def start_recording(self):
        with self.lock:
            self.record_buffer = []
            self.is_recording = True

    def stop_recording_and_save(self):
        with self.lock:
            if not self.is_recording:
                return False

            self.is_recording = False
            rows = list(self.record_buffer)
            self.record_buffer = []

        if not rows:
            print(f"[{self.port}] No recorded data to save.")
            return False

        return self._write_rows_to_csv(rows)

    def save_active_recording_if_needed(self):
        with self.lock:
            rows = list(self.record_buffer) if self.is_recording and self.record_buffer else []
            self.is_recording = False
            self.record_buffer = []

        if rows:
            self._write_rows_to_csv(rows)

    def _write_rows_to_csv(self, rows):
        try:
            os.makedirs(SAVE_DIR, exist_ok=True)

            self.save_counter += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/session_{ts}_{self.port_sanitized}_{self.save_counter:03d}.csv"

            with open(filename, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "host_time_s",
                    "sample_index",
                    "v1_raw", "v2_raw", "v3_raw",
                    "v1_voltage_V", "v2_voltage_V", "v3_voltage_V",
                    "v1_newtons", "v2_newtons", "v3_newtons"
                ])
                for row in rows:
                    w.writerow(row)

            print(f"[{self.port}] Saved {len(rows)} rows to {filename}")
            return True

        except Exception as e:
            print(f"[{self.port}] Failed to save CSV: {e}")
            return False

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

                v1_raw = int(float(match.group(2)))
                v2_raw = int(float(match.group(3)))
                v3_raw = int(float(match.group(4)))

                v1_voltage = raw_to_voltage(v1_raw)
                v2_voltage = raw_to_voltage(v2_raw)
                v3_voltage = raw_to_voltage(v3_raw)

                v1_newtons = round(raw_to_newtons(v1_raw, "V1"), 6)
                v2_newtons = round(raw_to_newtons(v2_raw, "V2"), 6)
                v3_newtons = round(raw_to_newtons(v3_raw, "V3"), 6)

                row = (
                    t_host,          # [0]
                    self.index,      # [1]
                    v1_raw,          # [2]
                    v2_raw,          # [3]
                    v3_raw,          # [4]
                    v1_voltage,      # [5]
                    v2_voltage,      # [6]
                    v3_voltage,      # [7]
                    v1_newtons,      # [8]
                    v2_newtons,      # [9]
                    v3_newtons       # [10]
                )

                with self.lock:
                    self.live_buffer.append(row)

                    if len(self.live_buffer) > PLOT_HISTORY:
                        self.live_buffer = self.live_buffer[-PLOT_HISTORY:]

                    if self.is_recording:
                        self.record_buffer.append(row)

                    self.index += 1

            except Exception as e:
                print(f"[{self.port}] Read error: {e}")
                continue

    def update_channels(self):
        with self.lock:
            if len(self.live_buffer) < 2:
                return

            recent = self.live_buffer[-PLOT_HISTORY:]
            x = [r[0] for r in recent]
            y1 = [r[8] for r in recent]
            y2 = [r[9] for r in recent]
            y3 = [r[10] for r in recent]

        self.line_v1.set_data(x, y1)
        self.line_v2.set_data(x, y2)
        self.line_v3.set_data(x, y3)

        self.ax_channels.relim()
        self.ax_channels.autoscale_view()

    def get_force_series(self):
        with self.lock:
            if not self.live_buffer:
                return [], [], []
            f1 = [r[8] for r in self.live_buffer]
            f2 = [r[9] for r in self.live_buffer]
            f3 = [r[10] for r in self.live_buffer]
        return f1, f2, f3

    def finalize_and_close(self):
        if self.finalized_once.is_set():
            return
        self.finalized_once.set()

        # save unfinished active recording, if any
        self.save_active_recording_if_needed()

        self.stop_event.set()
        try:
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=2)
        except Exception:
            pass

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

BTN_RECORD = None
BTN_AX = None
_RECORDING_ACTIVE = False
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
    global FIG, DEVICES, BTN_RECORD, BTN_AX
    global CENTER_AX, CENTER_LINE_V1, CENTER_LINE_V2, CENTER_LINE_V3
    global _RECORDING_ACTIVE

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

    BTN_AX = FIG.add_axes([0.36, 0.02, 0.28, 0.06])
    BTN_RECORD = Button(BTN_AX, "Start Recording")
    _RECORDING_ACTIVE = False

    def _on_record_toggle(_evt=None):
        global _RECORDING_ACTIVE

        if not _RECORDING_ACTIVE:
            for ctx in DEVICES:
                ctx.start_recording()

            _RECORDING_ACTIVE = True

            try:
                BTN_RECORD.label.set_text("Stop Recording")
                FIG.canvas.draw_idle()
            except Exception:
                pass

            print("Recording started.")

        else:
            any_saved = False
            for ctx in DEVICES:
                saved = ctx.stop_recording_and_save()
                any_saved = any_saved or saved

            _RECORDING_ACTIVE = False

            try:
                BTN_RECORD.label.set_text("Start Recording")
                FIG.canvas.draw_idle()
            except Exception:
                pass

            if any_saved:
                print("Recording stopped and saved.")
            else:
                print("Recording stopped.")

    BTN_RECORD.on_clicked(_on_record_toggle)

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
            ctx.finalize_and_close()
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


# =========================
# Main
# =========================

if __name__ == "__main__":
    register_signal_handlers()
    setup_devices_and_figure()

    timer = FIG.canvas.new_timer(interval=100)
    timer.add_callback(update_all, None)
    timer.start()

    try:
        plt.show()
    finally:
        finalize_all_and_exit()