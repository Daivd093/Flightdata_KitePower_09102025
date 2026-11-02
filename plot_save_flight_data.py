#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11 15:00:00 2024

Simple visualization toolchain for experimental data.

@author: Oriol Cayon
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import os
import zipfile

def plot_kite_data_with_sliders(flight_data, variables, labels, subplot_labels):
    """
    Plot kite flight data with:
      - cycle selection slider
      - time slider for manually moving through each cycle
      - elevation and azimuth sliders for the 3D view
      - synchronized red markers on the 3D trajectory and time-series plots
    """

    unique_cycles = np.sort(flight_data["cycle"].unique())
    n = len(variables)

    def slice_variable(variable, mask):
        """Apply the cycle mask to either a pandas Series or NumPy array."""
        if isinstance(variable, pd.Series):
            return variable.loc[mask].to_numpy()

        mask_array = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
        return np.asarray(variable)[mask_array]

    def load_cycle(cycle_index):
        """Extract all data belonging to one cycle."""
        selected_cycle = unique_cycles[int(cycle_index)]
        mask = flight_data["cycle"] == selected_cycle

        t = flight_data.loc[mask, "time"].to_numpy()
        x = flight_data.loc[mask, "kite_pos_east"].to_numpy()
        y = flight_data.loc[mask, "kite_pos_north"].to_numpy()
        z = flight_data.loc[mask, "kite_height"].to_numpy()

        cycle_variables = []

        for variable_group in variables:
            cycle_variables.append(
                [
                    slice_variable(variable, mask)
                    for variable in variable_group
                ]
            )

        return {
            "cycle": selected_cycle,
            "t": t,
            "x": x,
            "y": y,
            "z": z,
            "variables": cycle_variables,
            "number_of_points": len(t),
        }

    def set_3d_limits(axis, x, y, z):
        """Set 3D limits with padding."""
        if len(x) == 0:
            return

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        z_range = z.max() - z.min()

        x_padding = max(0.05 * x_range, 1e-6)
        y_padding = max(0.05 * y_range, 1e-6)
        z_padding = max(0.05 * z_range, 1e-6)

        axis.set_xlim(x.min() - x_padding, x.max() + x_padding)
        axis.set_ylim(y.min() - y_padding, y.max() + y_padding)
        axis.set_zlim(z.min() - z_padding, z.max() + z_padding)

    def set_y_limits(axis, data_group):
        """Set suitable vertical limits for one time-series subplot."""
        valid_arrays = [
            np.asarray(data)
            for data in data_group
            if len(data) > 0
        ]

        if not valid_arrays:
            return

        all_values = np.concatenate(valid_arrays)
        all_values = all_values[np.isfinite(all_values)]

        if len(all_values) == 0:
            return

        minimum = np.min(all_values)
        maximum = np.max(all_values)

        if np.isclose(minimum, maximum):
            padding = max(0.05 * abs(minimum), 1.0)
        else:
            padding = 0.08 * (maximum - minimum)

        axis.set_ylim(minimum - padding, maximum + padding)

    # Initial cycle
    state = load_cycle(0)

    if state["number_of_points"] == 0:
        raise ValueError("The selected cycle contains no data points.")

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8))

    # Additional bottom space for four sliders
    plt.subplots_adjust(
        bottom=0.25,
        left=0.07,
        right=0.97,
        top=0.95
    )

    gs = gridspec.GridSpec(
        n,
        2,
        width_ratios=[1, 1],
        height_ratios=[0.5 for _ in range(n)]
    )

    # ------------------------------------------------------------------
    # 3D trajectory
    # ------------------------------------------------------------------
    ax_3d = fig.add_subplot(gs[0:min(3, n), 0], projection="3d")

    trajectory_line, = ax_3d.plot(
        state["x"],
        state["y"],
        state["z"],
        label="Trajectory"
    )

    trajectory_point, = ax_3d.plot(
        [state["x"][0]],
        [state["y"][0]],
        [state["z"][0]],
        "ro",
        markersize=8,
        label="Selected point"
    )

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("Kite Trajectory")
    ax_3d.view_init(elev=30, azim=30)
    ax_3d.legend()

    set_3d_limits(
        ax_3d,
        state["x"],
        state["y"],
        state["z"]
    )

    # ------------------------------------------------------------------
    # Time-series plots
    # ------------------------------------------------------------------
    axes_right = []
    line_groups = []
    marker_groups = []
    time_cursors = []

    for i in range(n):
        axis = fig.add_subplot(gs[i, 1])
        axes_right.append(axis)

        subplot_lines = []
        subplot_markers = []

        for j, y_data in enumerate(state["variables"][i]):
            current_label = labels[i][j]

            if current_label is None:
                line, = axis.plot(state["t"], y_data)
            else:
                line, = axis.plot(
                    state["t"],
                    y_data,
                    label=current_label
                )

            marker, = axis.plot(
                [state["t"][0]],
                [y_data[0]],
                "ro",
                markersize=6,
                zorder=5
            )

            subplot_lines.append(line)
            subplot_markers.append(marker)

        time_cursor = axis.axvline(
            state["t"][0],
            color="red",
            linestyle="--",
            linewidth=1.2,
            alpha=0.6
        )

        axis.set_ylabel(subplot_labels[i])
        axis.grid(True)

        if i == n - 1:
            axis.set_xlabel("Time")

        if any(label is not None for label in labels[i]):
            axis.legend()

        axis.set_xlim(state["t"].min(), state["t"].max())
        set_y_limits(axis, state["variables"][i])

        line_groups.append(subplot_lines)
        marker_groups.append(subplot_markers)
        time_cursors.append(time_cursor)

    # ------------------------------------------------------------------
    # Sliders
    # ------------------------------------------------------------------

    # Time slider
    time_slider_axis = fig.add_axes(
        [0.10, 0.04, 0.30, 0.03],
        facecolor="lightgoldenrodyellow"
    )

    maximum_time_index = max(state["number_of_points"] - 1, 1)

    time_slider = Slider(
        time_slider_axis,
        "Time",
        0,
        maximum_time_index,
        valinit=0,
        valstep=1
    )

    # Cycle slider
    cycle_slider_axis = fig.add_axes(
        [0.10, 0.08, 0.30, 0.03],
        facecolor="lightgoldenrodyellow"
    )

    cycle_slider = Slider(
        cycle_slider_axis,
        "Cycle",
        0,
        len(unique_cycles) - 1,
        valinit=0,
        valstep=1
    )

    # Elevation slider
    elevation_slider_axis = fig.add_axes(
        [0.10, 0.12, 0.30, 0.03],
        facecolor="lightgoldenrodyellow"
    )

    elevation_slider = Slider(
        elevation_slider_axis,
        "Elevation",
        0,
        90,
        valinit=30
    )

    # Azimuth slider
    azimuth_slider_axis = fig.add_axes(
        [0.10, 0.16, 0.30, 0.03],
        facecolor="lightgoldenrodyellow"
    )

    azimuth_slider = Slider(
        azimuth_slider_axis,
        "Azimuth",
        0,
        360,
        valinit=30
    )

    # ------------------------------------------------------------------
    # Current-point update
    # ------------------------------------------------------------------
    def update_selected_point(point_index):
        """Move every red marker to the selected data point."""
        if state["number_of_points"] == 0:
            return

        point_index = int(round(point_index))
        point_index = np.clip(
            point_index,
            0,
            state["number_of_points"] - 1
        )

        current_time = state["t"][point_index]

        # Move point on 3D trajectory
        trajectory_point.set_data(
            [state["x"][point_index]],
            [state["y"][point_index]]
        )
        trajectory_point.set_3d_properties(
            [state["z"][point_index]]
        )

        # Move points and vertical cursor on right-hand plots
        for i in range(n):
            time_cursors[i].set_xdata(
                [current_time, current_time]
            )

            for j, marker in enumerate(marker_groups[i]):
                current_value = state["variables"][i][j][point_index]

                marker.set_data(
                    [current_time],
                    [current_value]
                )

        # Show elapsed time in the current cycle
        elapsed_time = current_time - state["t"][0]
        time_slider.valtext.set_text(f"{elapsed_time:.2f} s")

        fig.canvas.draw_idle()

    def update_time(value):
        update_selected_point(value)

    # ------------------------------------------------------------------
    # Cycle update
    # ------------------------------------------------------------------
    def update_cycle(value):
        nonlocal state

        cycle_index = int(round(cycle_slider.val))
        state = load_cycle(cycle_index)

        if state["number_of_points"] == 0:
            return

        # Update 3D trajectory
        trajectory_line.set_data(state["x"], state["y"])
        trajectory_line.set_3d_properties(state["z"])

        set_3d_limits(
            ax_3d,
            state["x"],
            state["y"],
            state["z"]
        )

        # Update time-series data
        for i, axis in enumerate(axes_right):
            for j, line in enumerate(line_groups[i]):
                line.set_data(
                    state["t"],
                    state["variables"][i][j]
                )

            axis.set_xlim(
                state["t"].min(),
                state["t"].max()
            )

            set_y_limits(
                axis,
                state["variables"][i]
            )

        # Update the time-slider range
        new_maximum = max(state["number_of_points"] - 1, 1)

        time_slider.eventson = False

        time_slider.valmin = 0
        time_slider.valmax = new_maximum
        time_slider.valstep = 1
        time_slider.ax.set_xlim(0, new_maximum)
        time_slider.set_val(0)

        time_slider.eventson = True

        # Move all markers to the first point of the new cycle
        update_selected_point(0)

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 3D view update
    # ------------------------------------------------------------------
    def update_view(value):
        ax_3d.view_init(
            elev=elevation_slider.val,
            azim=azimuth_slider.val
        )
        fig.canvas.draw_idle()

    # Connect callbacks
    time_slider.on_changed(update_time)
    cycle_slider.on_changed(update_cycle)
    elevation_slider.on_changed(update_view)
    azimuth_slider.on_changed(update_view)

    # Show initial elapsed time
    update_selected_point(0)

    plt.show()

def ensure_csv_file(filename):
    """
    Ensure the CSV file is available by checking its existence
    and unzipping it if necessary.

    Parameters:
        csv_filename (str): The name of the CSV file to check.
        zip_filename (str): The name of the ZIP file containing the CSV.
    """
    csv_filename = filename + ".csv"
    zip_filename = filename + ".zip"
    # Check if the CSV file exists
    if not os.path.exists(csv_filename):
        print(f"{csv_filename} not found. Checking for {zip_filename}...")

        # Check if the ZIP file exists
        if os.path.exists(zip_filename):
            print(f"{zip_filename} found. Extracting...")
            with zipfile.ZipFile(zip_filename, "r") as zip_ref:
                zip_ref.extractall(".")  # Extract to the current directory
            print(f"Extracted {csv_filename}.")
        else:
            raise FileNotFoundError(
                f"Neither {csv_filename} nor {zip_filename} exists in the directory."
            )
    else:
        print(f"{csv_filename} is already present.")


# Ensure the CSV file is available
ensure_csv_file("log_2025-10-09_58-33-00")
# Load flight data
df = pd.read_csv("log_2025-10-09_58-33-00.csv")


# Plotting to see which part of the data is actually flight data
kite_height = df["kite_height"]
min_height = 20*np.ones(kite_height.shape)
cycle_100 = 100*df["cycle"]
x = df.index.to_numpy()
plt.plot(x,np.transpose([kite_height,min_height,cycle_100]))
plt.legend(["kite height","min flight height", "100 * cycle_num"])
plt.title("Whole data and flight data")
plt.show()


df["flight"] = (kite_height > min_height).astype(int)

df[["ground_tether_reelout_speed","ground_mech_power","ground_tether_force","kite_height"]].isna().any()

flight_phase_mapping = {"unknown": 0, "pp-ro": 1, "pp-ri": -1, "pp-riro": -0.5, "pp-rori": 0.5}
df["flight_phase_num"]= df["flight_phase"].map(flight_phase_mapping)
# flight_phase_index already exists, and has {"unknown": NaN, "pp-ro": 1, "pp-ri": 2, "pp-riro": 3, "pp-rori": 4}

for q in ["ground_tether_reelout_speed", "ground_tether_length", "ground_tether_force", "ground_mech_energy", "ground_mech_power"]:
    plt.plot(x,np.transpose([df[q],df["flight_phase_num"]*max(abs(df[q])),df["flight"]*max(abs(df[q])),df["cycle"]*max(abs(df[q]))/3]))
    plt.legend([q,"flight_phase={unknown=0,pp-ro=100%,pp-ri=-100%,pp-riro=-50%,pp-rori=50%}","flight","cycle"])
    plt.title(q+" and some extra info")
    plt.show()


# There are 2 points where kite_height is NaN, one is missing on cycle 2, so flight is 0 on one point it shouldn't. And the other one is before the kite starts flying.

# Fixing "flight"
df.loc[df["kite_height"].isna() & (df["cycle"] == 2), "flight"] = 1

# Interpolating "kite_flight" just in case.
df["kite_height"] = df["kite_height"].interpolate(method="linear")

from datetime import datetime, UTC

flight_data = df[df["flight"] == 1]

# Convert timestamp to datetime object of initial and final time
datetime_ini = datetime.fromtimestamp(flight_data["time"].iloc[0], UTC)
datetime_end = datetime.fromtimestamp(flight_data["time"].iloc[-1], UTC)

print(f"Initial time: {datetime_ini}")
print(f"Final time: {datetime_end}")

# Select variables to plot
tether_force = flight_data["ground_tether_force"]
kite_speed_0 = np.sqrt(
    flight_data["kite_0_vx"] ** 2
    + flight_data["kite_0_vy"] ** 2
    + flight_data["kite_0_vz"] ** 2
)
kite_speed_1 = np.sqrt(
    flight_data["kite_1_vx"] ** 2
    + flight_data["kite_1_vy"] ** 2
    + flight_data["kite_1_vz"] ** 2
)


# Select only 4 lidar speed variables, max, min and 2 in between

reelout_speed = flight_data["ground_tether_reelout_speed"]
power = flight_data["ground_mech_power"]
heading = flight_data["kite_heading"] * 180 / np.pi
course = flight_data["kite_course"] * 180 / np.pi
depower_setting = flight_data["kite_actual_depower"]
steering_setting = flight_data["kite_actual_steering"]
va = flight_data["airspeed_apparent_windspeed"]
cycle_num = flight_data["cycle"]
lidar_speeds = []
heights = []
for column in flight_data.columns:
    if "m Wind Speed (m/s)" in column:
        lidar_speeds.append(flight_data[column])
        heights.append("".join(filter(str.isdigit, column)))
# Select only 4 lidar speed variables, max, min and 2 in between
lidar_speeds = lidar_speeds[0:-1:3]
heights = heights[0:-1:3]


variables = [
    [tether_force * reelout_speed*9.80665, power],
    [power],
    [tether_force*9.80665],
    [reelout_speed],
]
labels = [
    ["F*v", "Pm"],
    [None],
    [None],
    [None],
]

subplot_labels = [
    "Power (W)",
    "Ground Mech Power (W)",
    "Tether Force (N)",
    "Reelout Speed (m/s)",
]
# Call the function
plot_kite_data_with_sliders(flight_data, variables, labels, subplot_labels)


# ------------------------------------------------------------------

# Saving on reduced CSV files for the electrical testbench:
  
# Columns to keep in all saved files
columns_to_save = [
    "time",
    "cycle",
    "flight_phase",
    "flight_phase_index",
    "kite_height",
    "ground_tether_length",
    "ground_tether_force_N",        # after multiplying ground_tether_force by 9.80665
    "ground_tether_reelout_speed",
    "ground_tether_power_calc",     # calculated from multiplying ground_tether_force_N*ground_tether_reelout_speed
    "ground_mech_power",            # measured mech power
    "ground_mech_energy"            # measured mech energy
]
 
# Adding the calculated columns to df:
df["ground_tether_force_N"] = df["ground_tether_force"] * 9.80665
df["ground_tether_power_calc"] = df["ground_tether_force_N"] * df["ground_tether_reelout_speed"]

# 1. Full flight log, selected columns, original data
df_full = df[columns_to_save]
df_full.to_csv("full_flightlog.csv", index=False)
print("Saved: full_flightlog.csv")
 
# 2. Clean cycles only (flight_phase_index not NaN), original data
#       These are the 4 full cycles without unknown flight phases,
#       excluding takeoff, landing and the partial cycles around them.
clean_cycles_mask = df["flight_phase_index"].notna()
df_clean = df.loc[clean_cycles_mask, columns_to_save]
df_clean.to_csv("clean_cycles.csv", index=False)
print("Saved: clean_cycles.csv")


# Saving also a normalized version of both dataframes

num_cols_not_to_normalize = ["time", "cycle", "flight_phase", "flight_phase_index"]

numeric_cols = df_full.select_dtypes(include=np.number).columns
numeric_cols_to_norm = numeric_cols.difference(num_cols_not_to_normalize)
max_vals = df_full[numeric_cols_to_norm].abs().max()
max_vals = max_vals.replace(0, 1) # Avoiding division by 0, just in case

# Saving the original normalization values
max_vals.to_csv("normalization_values.csv", header=["max_value"])

def normalize(data,norm_cols):
    normalized = data.copy()
    normalized[norm_cols] = data[norm_cols] / max_vals
    return normalized

# 3. Clean cycles only, normalize
df_clean_normalized = normalize(df_clean,numeric_cols_to_norm)
df_clean_normalized.to_csv("clean_cycles_normalized.csv", index=False)
print("Saved: clean_cycles_normalized.csv")
 
# --- 4. Full flight log, normalized (for future use) ---
df_full_normalized = normalize(df_full,numeric_cols_to_norm)
df_full_normalized.to_csv("full_flightlog_normalized.csv", index=False)
print("Saved: full_flightlog_normalized.csv")
 