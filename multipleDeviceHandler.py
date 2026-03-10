# main.py

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from BleTools import (
    setup_devices_and_figure,
    update_all,
    finalize_all_and_exit,
    register_signal_handlers,
    get_figure,
)

if __name__ == "__main__":
    # Install signal handlers (Ctrl+C, SIGTERM, etc.)
    register_signal_handlers()

    # Set up devices, calibration, and figure layout
    setup_devices_and_figure()

    # Get the figure created by the tools module
    fig = get_figure()

    # Start animation
    anim = FuncAnimation(fig, update_all, interval=50, blit=False)

    plt.show()

    # Safety net
    finalize_all_and_exit()
    print("All data collection stopped.")