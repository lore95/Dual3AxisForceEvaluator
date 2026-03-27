import asyncio
import threading
from typing import Dict, Optional, List, Tuple, Callable, Awaitable

from bleak import BleakScanner

from Controller.serial_reader import AsyncSensorReader

TX_UUID_DEFAULT = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify


class BLELoopThread:
    """Runs a dedicated asyncio loop in a background thread."""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro):
        """Schedule a coroutine onto the BLE loop, returns concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self):
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass

        try:
            self._thread.join(timeout=2)
        except Exception:
            pass


class BLEManager:
    def __init__(self, ble_loop: asyncio.AbstractEventLoop):
        self.ble_loop = ble_loop
        self.readers: Dict[str, AsyncSensorReader] = {}
        self.on_state_change: Optional[Callable[[], None]] = None

        # UI will set this: async prompt(addr, name)->bool
        self.prompt_save_cb: Optional[Callable[[str, Optional[str]], Awaitable[bool]]] = None

    async def scan_devices(self, timeout: float = 6.0) -> List[Tuple[str, str]]:
        devices = await BleakScanner.discover(timeout=timeout)
        out: List[Tuple[str, str]] = []

        for d in devices:
            out.append((d.address, d.name or "Unknown"))

        out.sort(key=lambda x: (x[1] or "").lower())
        return out

    async def connect(self, address: str, *, tx_uuid: str = TX_UUID_DEFAULT) -> bool:
        if address in self.readers and self.readers[address].is_connected:
            return True

        reader = self.readers.get(address)
        if reader is None:
            async def _reader_prompt(addr: str, name: Optional[str]):
                if self.prompt_save_cb:
                    return await self.prompt_save_cb(addr, name)
                return True

            reader = AsyncSensorReader(
                ble_address=address,
                tx_uuid=tx_uuid,
                ble_loop=self.ble_loop,
                prompt_save_cb=_reader_prompt,
            )
            self.readers[address] = reader

            reader.state_change_cb = lambda: (self.on_state_change() if self.on_state_change else None)

        return await reader.connect_device()

    async def disconnect(self, address: str) -> bool:
        reader = self.readers.get(address)
        if not reader:
            return True
        return await reader.disconnect_device()

    async def disconnect_all(self):
        for addr in list(self.readers.keys()):
            try:
                await self.disconnect(addr)
            except Exception:
                pass