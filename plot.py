import csv
import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt


def select_csv_file(initial_folder="Readings"):
    root = tk.Tk()
    root.withdraw()  # hide main tkinter window

    initial_dir = os.path.abspath(initial_folder)

    file_path = filedialog.askopenfilename(
        title="Select a CSV file to plot",
        initialdir=initial_dir,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    root.destroy()
    return file_path


def plot_csv(csv_file):
    index = []

    v1_raw = []
    v2_raw = []
    v3_raw = []

    v1_force = []
    v2_force = []
    v3_force = []

    with open(csv_file, "r", newline="") as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            index.append(i)

            v1_raw.append(float(row["v1_raw"]))
            v2_raw.append(float(row["v2_raw"]))
            v3_raw.append(float(row["v3_raw"]))

            v1_force.append(float(row["v1_force_N"]))
            v2_force.append(float(row["v2_force_N"]))
            v3_force.append(float(row["v3_force_N"]))

    plt.figure(figsize=(10, 6))
    plt.plot(index, v1_force, label="V1 Force")
    plt.plot(index, v2_force, label="V2 Force")
    plt.plot(index, v3_force, label="V3 Force")

    plt.title(f"Sensor Forces\n{os.path.basename(csv_file)}")
    plt.xlabel("Sample Index")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    csv_file = select_csv_file("Readings")

    if not csv_file:
        print("No file selected.")
        return

    plot_csv(csv_file)


if __name__ == "__main__":
    main()