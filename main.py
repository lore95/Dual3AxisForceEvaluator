from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from app import (
    setup_devices_and_figure,
    update_all,
    finalize_all_and_exit,
    register_signal_handlers,
    get_figure,
)

if __name__ == "__main__":
    register_signal_handlers()
    setup_devices_and_figure()
    fig = get_figure()

    anim = FuncAnimation(fig, update_all, interval=50, blit=False)
    plt.show()

    finalize_all_and_exit()
    print("All data collection stopped.")