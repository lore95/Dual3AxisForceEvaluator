import os
import re
import csv
import time
import asyncio
from datetime import date
from typing import Optional, Callable, Awaitable, Dict, Any, Tuple

import numpy as np
from bleak import BleakClient
from bleak.exc import BleakError, BleakDBusError

from calibration import right_sensor_counts_to_newtons, adc_to_sensor_voltage

LINE_RE = re.compile(
    r"Time:(-?\d+),V1:(-?\d+(?:\.\d+)?),V2:(-?\d+(?:\.\d+)?),V3:(-?\d+(?:\.\d+)?)"
)

ZERO_CAPTURE_SECONDS = 1.0
PLOT_HISTORY = 300
VEXC = 10


class AsyncSensorReader:
    """
    BLE sensor reader with 3-channel support:
      - parses Time,V1,V2,V3 notifications
      - calculates zero offsets
      - converts to voltage + force
      - stores data in a layout compatible with the serial pipeline
    """

    def __init__(
        self,
        ble_address: str,
        tx_uuid: str,
        ble_loop: asyncio.AbstractEventLoop,
        *,
        prompt_save_cb: Optional[Callable[[str, Optional[str]], Awaitable[bool]]] = None,
    ):
        self.ble_address = ble_address
        self.tx_uuid = tx_uuid
        self.ble_loop = ble_loop

        self.client: Optional[BleakClient] = None
        self.is_connected = False
        self.is_reading = False

        self.disconnect_error = False
        self.state_change_cb: Optional[Callable[[], None]] = None

        self._intentional_disconnect = False
        self._disconnect_lock = asyncio.Lock()

        self.prompt_save_cb = prompt_save_cb
        self.name: Optional[str] = None

        self._pending_meta: Dict[str, Any] = {
            "athlete_id": "UNKNOWN",
            "distance_cm": 0.0,
            "weight_kg": 0,
        }

        # Row format:
        # (
        #   host_time_s, sample_index,
        #   v1_raw, v2_raw, v3_raw,
        #   v1_voltage_V, v2_voltage_V, v3_voltage_V,
        #   v1_newtons, v2_newtons, v3_newtons
        # )
        self.collected_rows: list[Tuple] = []
        self.live_rows: list[Tuple] = []
        self.sample_index = 0

        self.zero_capture_start_host: Optional[float] = None
        self.zero_capture_done = False
        self.zero_samples: list[Tuple[int, int, int]] = []
        self.zero_offsets = (0.0, 0.0, 0.0)

    # ---------------- notifications ----------------

    def notification_handler(self, sender: int, data: bytearray):
        line = data.decode("utf-8", errors="ignore").strip()
        m = LINE_RE.match(line)
        if not m:
            return

        t_us = int(m.group(1))
        t_host = t_us / 1_000_000.0

        v1_raw = int(float(m.group(2)))
        v2_raw = int(float(m.group(3)))
        v3_raw = int(float(m.group(4)))

        if self.zero_capture_start_host is None:
            self.zero_capture_start_host = t_host

        if not self.zero_capture_done:
            self.zero_samples.append((v1_raw, v2_raw, v3_raw))
            elapsed = t_host - self.zero_capture_start_host

            if elapsed >= ZERO_CAPTURE_SECONDS and self.zero_samples:
                arr = np.array(self.zero_samples, dtype=np.float64)
                self.zero_offsets = (
                    float(np.mean(arr[:, 0])),
                    float(np.mean(arr[:, 1])),
                    float(np.mean(arr[:, 2])),
                )
                self.zero_capture_done = True
                print(
                    f"[BLE {self.ble_address}] Zero offsets captured: "
                    f"V1={self.zero_offsets[0]:.3f}, "
                    f"V2={self.zero_offsets[1]:.3f}, "
                    f"V3={self.zero_offsets[2]:.3f}"
                )
            return

        v1_voltage_V = adc_to_sensor_voltage(v1_raw, self.zero_offsets[0])
        v2_voltage_V = adc_to_sensor_voltage(v2_raw, self.zero_offsets[1])
        v3_voltage_V = adc_to_sensor_voltage(v3_raw, self.zero_offsets[2])

        v1_newtons, v2_newtons, v3_newtons = right_sensor_counts_to_newtons(
            v1_raw, v2_raw, v3_raw, self.zero_offsets, VEXC
        )

        row = (
            t_host,
            self.sample_index,
            v1_raw,
            v2_raw,
            v3_raw,
            v1_voltage_V,
            v2_voltage_V,
            v3_voltage_V,
            v1_newtons,
            v2_newtons,
            v3_newtons,
        )

        self.live_rows.append(row)
        if len(self.live_rows) > PLOT_HISTORY:
            self.live_rows = self.live_rows[-PLOT_HISTORY:]

        if self.is_reading:
            self.collected_rows.append(row)

        self.sample_index += 1

    # ---------------- disconnect handling ----------------

    def _on_disconnect(self, _client: BleakClient):
        if self._intentional_disconnect:
            return

        print(f"[BLE {self.ble_address}] Device disconnected unexpectedly.")
        self.disconnect_error = True
        self.is_reading = False
        self.is_connected = False
        self._notify_state_change()

        try:
            asyncio.run_coroutine_threadsafe(self._handle_disconnect(), self.ble_loop)
        except Exception as e:
            print(f"[BLE {self.ble_address}] Failed to schedule disconnect handler: {e}")

    async def _handle_disconnect(self):
        async with self._disconnect_lock:
            had_data = bool(self.collected_rows)

            if self.client:
                try:
                    await self.client.disconnect()
                except Exception:
                    pass
                self.client = None

            save = True
            if self.disconnect_error and self.prompt_save_cb is not None and had_data:
                try:
                    save = await self.prompt_save_cb(self.ble_address, self.name)
                except Exception as e:
                    print(f"[BLE {self.ble_address}] prompt_save_cb failed: {e}")
                    save = True

            if (not had_data) or (not save):
                self._clear_buffers()
                return

            meta = dict(self._pending_meta)
            athlete_id = meta.get("athlete_id", "UNKNOWN")
            distance_cm = float(meta.get("distance_cm", 0.0))
            weight_kg = int(meta.get("weight_kg", 0))

            try:
                filename = await asyncio.to_thread(
                    self._save_data,
                    self.collected_rows,
                    athlete_id,
                    distance_cm,
                    weight_kg,
                )
                print(f"[BLE {self.ble_address}] Saved partial data: {filename}")
            finally:
                self._clear_buffers()

    def _clear_buffers(self):
        self.collected_rows.clear()
        self.live_rows.clear()
        self.sample_index = 0
        self.zero_capture_start_host = None
        self.zero_capture_done = False
        self.zero_samples.clear()
        self.zero_offsets = (0.0, 0.0, 0.0)
        self._pending_meta = {"athlete_id": "UNKNOWN", "distance_cm": 0.0, "weight_kg": 0}

    def _notify_state_change(self):
        cb = self.state_change_cb
        if cb:
            try:
                cb()
            except Exception:
                pass

    # ---------------- connect / disconnect ----------------

    async def connect_device(self) -> bool:
        self._intentional_disconnect = False
        self.disconnect_error = False
        self._notify_state_change()

        if self.client:
            try:
                await self.client.disconnect()
            except Exception:
                pass
            self.client = None

        print(f"[BLE {self.ble_address}] Attempting connection...")
        try:
            self.client = BleakClient(
                self.ble_address,
                timeout=20.0,
                disconnected_callback=self._on_disconnect,
            )
            await self.client.connect()

            if not self.client.is_connected:
                print(f"[BLE {self.ble_address}] Failed to connect.")
                self.is_connected = False
                self._notify_state_change()
                return False

            await self.client.start_notify(self.tx_uuid, self.notification_handler)

            self.is_connected = True
            self.disconnect_error = False
            self._notify_state_change()
            print(f"[BLE {self.ble_address}] Connected. Notifications activated.")
            return True

        except (BleakError, BleakDBusError) as e:
            print(f"[BLE {self.ble_address}] Connection error: {e}")
        except Exception as e:
            print(f"[BLE {self.ble_address}] Unexpected error: {e}")

        self.is_connected = False
        self.disconnect_error = False
        self.client = None
        self._notify_state_change()
        return False

    async def disconnect_device(self) -> bool:
        self._intentional_disconnect = True
        self.is_reading = False

        try:
            if self.client:
                try:
                    await self.client.stop_notify(self.tx_uuid)
                except Exception:
                    pass
                try:
                    await self.client.disconnect()
                except Exception:
                    pass
                self.client = None

            self.is_connected = False
            self.disconnect_error = False
            self._notify_state_change()
            print(f"[BLE {self.ble_address}] Explicitly disconnected.")
            return True
        finally:
            self._intentional_disconnect = False

    close = disconnect_device

    # ---------------- reading control ----------------

    async def start_reading(
        self,
        athlete_id: str = "UNKNOWN",
        distance_cm: float = 0.0,
        weight_kg: int = 0,
        direction: int = 0,
    ) -> bool:
        if not (self.client and self.client.is_connected):
            return False

        self.collected_rows.clear()
        self.sample_index = 0
        self.zero_capture_start_host = None
        self.zero_capture_done = False
        self.zero_samples.clear()
        self.zero_offsets = (0.0, 0.0, 0.0)

        self._pending_meta = {
            "athlete_id": str(athlete_id or "UNKNOWN"),
            "distance_cm": float(distance_cm),
            "weight_kg": int(weight_kg),
        }

        if direction == 0:
            self.is_reading = True

        return True

    async def stop_reading(
        self,
        athlete_id: str = "UNKNOWN",
        distance_cm: float = 0.0,
        weight_kg: int = 0,
    ):
        self.is_reading = False
        print(f"[BLE {self.ble_address}] Data logging stopped. Saving data...")

        filename = await asyncio.to_thread(
            self._save_data,
            self.collected_rows,
            athlete_id,
            distance_cm,
            weight_kg,
        )
        self._clear_buffers()
        return filename

    # ---------------- save ----------------

    def _save_data(self, rows, athlete_id: str, distance_cm: float, weight_kg: int):
        os.makedirs("readings", exist_ok=True)

        today_str = date.today().isoformat()
        athlete_id = (athlete_id or "").strip()
        save_dir = os.path.join("readings", f"{today_str}_{athlete_id}" if athlete_id else today_str)
        os.makedirs(save_dir, exist_ok=True)

        timestamp_s = int(time.time())
        dist_str = f"{int(distance_cm)}cm"
        weight_str = f"{int(weight_kg)}kg"
        filename = os.path.join(save_dir, f"{timestamp_s}_{dist_str}_{weight_str}_grip_data.csv")

        if not rows:
            print("[SAVE] No data to save.")
            return None

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "host_time_s",
                "sample_index",
                "v1_raw", "v2_raw", "v3_raw",
                "v1_voltage_V", "v2_voltage_V", "v3_voltage_V",
                "v1_newtons", "v2_newtons", "v3_newtons",
            ])
            writer.writerows(rows)

        print(f"[SAVE] Saved {len(rows)} samples to {filename}")
        return filename