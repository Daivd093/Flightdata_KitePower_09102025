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

'''
def plot_kite_data_with_sliders(flight_data, variables, labels, subplot_labels):
    """
    Plot kite flight data with interactive sliders for cycle selection and 3D view control.

    Parameters:
        flight_data (pd.DataFrame): The dataset containing kite flight data.
        selected_vars (list of str): The variables to plot in time-series format.
    """
    # Extract unique cycles
    unique_cycles = flight_data["cycle"].unique()

    # Initial cycle
    current_cycle = unique_cycles[0]

    # Filter data for the initial cycle
    mask = flight_data["cycle"] == current_cycle

    # Extract time and position data
    t = flight_data[mask]["time"].values
    x = flight_data[mask]["kite_pos_east"].values
    y = flight_data[mask]["kite_pos_north"].values
    z = flight_data[mask]["kite_height"].values

    # Setup figure and layout
    n = len(variables)
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(
        n, 2, width_ratios=[1, 1], height_ratios=[0.5 for _ in range(n)]
    )

    # 3D plot of the trajectory
    ax_3d = fig.add_subplot(gs[0:3, 0], projection="3d")
    (trajectory_line,) = ax_3d.plot(x, y, z, label="Trajectory")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("Kite Trajectory")
    ax_3d.legend()

    # Time series plots for each selected variable
    ax_vars = []
    for i, variable in enumerate(variables):
        ax = fig.add_subplot(gs[i, 1:])
        ax_subvars = []
        for j, var in enumerate(variable):
            (line,) = ax.plot(t, var[mask], label=labels[i][j])
            ax_subvars.append((line))

        ax.set_ylabel(subplot_labels[i])
        if i == len(variables) - 1:
            ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True)
        ax_vars.append(ax_subvars)

    # Cycle slider
    slider_ax_cycle = fig.add_axes(
        [0.1, 0.05, 0.3, 0.03], facecolor="lightgoldenrodyellow"
    )
    cycle_slider = Slider(
        slider_ax_cycle, "Cycle", 0, len(unique_cycles) - 1, valinit=0, valstep=1
    )

    # Slider for elevation angle
    slider_ax_elev = fig.add_axes(
        [0.1, 0.1, 0.3, 0.03], facecolor="lightgoldenrodyellow"
    )
    elev_slider = Slider(
        slider_ax_elev, "Elevation", 0, 90, valinit=30, orientation="horizontal"
    )

    # Slider for azimuth angle
    slider_ax_azim = fig.add_axes(
        [0.1, 0.15, 0.3, 0.03], facecolor="lightgoldenrodyellow"
    )
    azim_slider = Slider(
        slider_ax_azim, "Azimuth", 0, 360, valinit=30, orientation="horizontal"
    )

    # Update function for cycle slider
    def update_cycle(val):
        nonlocal flight_data, t, x, y, z, variables
        cycle_idx = int(cycle_slider.val)
        selected_cycle = unique_cycles[cycle_idx]
        mask = flight_data["cycle"] == selected_cycle

        # Update trajectory data
        t = flight_data[mask]["time"].values
        x = flight_data[mask]["kite_pos_east"].values
        y = flight_data[mask]["kite_pos_north"].values
        z = flight_data[mask]["kite_height"].values
        trajectory_line.set_data(x, y)
        trajectory_line.set_3d_properties(z)
        trajectory_line.axes.set_xlim(x.min(), x.max())
        trajectory_line.axes.set_ylim(y.min(), y.max())
        trajectory_line.axes.set_zlim(z.min(), z.max())

        # Update variables
        for (line), variable in zip(ax_vars, variables):
            for j, var in enumerate(variable):
                line[j].set_data(t, var[mask])
            line[j].axes.set_xlim(t.min(), t.max())

        # Redraw the figure
        fig.canvas.draw_idle()

    # Update function for 3D view sliders
    def update_view(val):
        ax_3d.view_init(elev=elev_slider.val, azim=azim_slider.val)
        fig.canvas.draw_idle()

    # Connect sliders to update functions
    cycle_slider.on_changed(update_cycle)
    elev_slider.on_changed(update_view)
    azim_slider.on_changed(update_view)

    plt.show()
'''

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

flight_data = df[df["kite_height"] > 20]

from datetime import datetime, UTC

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

'''
import numpy as np

g0 = 9.80665  # N per kgf

power_from_tether = (
    flight_data["ground_tether_force"]
    * g0
    * flight_data["ground_tether_reelout_speed"]
)

measured_power = flight_data["ground_mech_power"]

valid = (
    np.isfinite(power_from_tether)
    & np.isfinite(measured_power)
    & (np.abs(power_from_tether) > 100)
)

ratio = measured_power[valid] / power_from_tether[valid]

print("Median power ratio:", np.median(ratio))
print(
    "Correlation:",
    np.corrcoef(
        measured_power[valid],
        power_from_tether[valid],
    )[0, 1],
)



import numpy as np
import pandas as pd

g0 = 9.80665

force_N = flight_data["ground_tether_force"] * g0
speed = flight_data["ground_tether_reelout_speed"]
power_tether = force_N * speed
power_mech = flight_data["ground_mech_power"]

analysis = pd.DataFrame({
    "speed": speed,
    "power_tether": power_tether,
    "power_mech": power_mech,
}).replace([np.inf, -np.inf], np.nan).dropna()

# Remove low-power points, where ratios become unstable
analysis = analysis[
    (analysis["power_tether"].abs() > 500)
    & (analysis["power_mech"].abs() > 100)
]

for phase, mask in {
    "reel-out": analysis["speed"] > 0,
    "reel-in": analysis["speed"] < 0,
}.items():

    subset = analysis.loc[mask]

    if len(subset) < 2:
        continue

    correlation = subset["power_mech"].corr(subset["power_tether"])

    ratio = (
        subset["power_mech"]
        / subset["power_tether"]
    ).median()

    # Fit P_mech = slope * P_tether + intercept
    slope, intercept = np.polyfit(
        subset["power_tether"],
        subset["power_mech"],
        1,
    )

    print(f"\n{phase}")
    print(f"Samples: {len(subset)}")
    print(f"Correlation: {correlation:.3f}")
    print(f"Median ratio: {ratio:.3f}")
    print(f"Slope: {slope:.3f}")
    print(f"Intercept: {intercept:.1f} W")


import numpy as np

x = analysis["power_tether"].to_numpy()
y = analysis["power_mech"].to_numpy()

x = (x - x.mean()) / x.std()
y = (y - y.mean()) / y.std()

max_lag_samples = 100
lags = range(-max_lag_samples, max_lag_samples + 1)

correlations = []

for lag in lags:
    if lag < 0:
        corr = np.corrcoef(x[-lag:], y[:lag])[0, 1]
    elif lag > 0:
        corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
    else:
        corr = np.corrcoef(x, y)[0, 1]

    correlations.append(corr)

best_index = int(np.nanargmax(correlations))
best_lag = list(lags)[best_index]
best_corr = correlations[best_index]

sampling_period_seconds = 0.1
best_lag_seconds = best_lag * sampling_period_seconds

print("\nBest lag in samples:", best_lag)
print("Best lag in seconds:", best_lag_seconds)
print("Best correlation:", best_corr)



import numpy as np

sampling_period = 0.1
max_lag_seconds = 5
max_lag_samples = round(max_lag_seconds / sampling_period)


def find_best_lag(x, y, max_lag):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []

    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(x[-lag:], y[:lag])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]

        correlations.append(corr)

    correlations = np.asarray(correlations)

    best_index = np.nanargmax(correlations)

    return (
        lags[best_index],
        correlations[best_index],
        lags,
        correlations,
    )


for phase, mask in {
    "reel-out": analysis["speed"] > 0,
    "reel-in": analysis["speed"] < 0,
}.items():

    subset = analysis.loc[mask]

    best_lag, best_corr, _, _ = find_best_lag(
        subset["power_tether"],
        subset["power_mech"],
        max_lag_samples,
    )

    print(f"\n{phase}")
    print(f"Best lag: {best_lag} samples")
    print(f"Best lag: {best_lag * sampling_period:.2f} s")
    print(f"Best correlation: {best_corr:.3f}")




dt = 0.1

for phase, mask in {
    "reel-out": analysis["speed"] > 0,
    "reel-in": analysis["speed"] < 0,
}.items():

    subset = analysis.loc[mask]

    energy_tether_J = np.trapezoid(
        subset["power_tether"],
        dx=dt,
    )

    energy_mech_J = np.trapezoid(
        subset["power_mech"],
        dx=dt,
    )

    print(f"\n{phase}")
    print(f"Tether energy: {energy_tether_J / 3.6e6:.3f} kWh")
    print(f"Mechanical energy: {energy_mech_J / 3.6e6:.3f} kWh")

    if abs(energy_tether_J) > 0:
        print(
            "Energy ratio:",
            energy_mech_J / energy_tether_J,
        )


dt = 0.1

power_tether = (
    flight_data["ground_tether_force"]
    * 9.80665
    * flight_data["ground_tether_reelout_speed"]
)

power_mech = flight_data["ground_mech_power"]
speed = flight_data["ground_tether_reelout_speed"]

valid = (
    np.isfinite(power_tether)
    & np.isfinite(power_mech)
    & np.isfinite(speed)
)

for phase, phase_mask in {
    "reel-out": speed > 0,
    "reel-in": speed < 0,
}.items():

    mask = valid & phase_mask

    energy_tether_J = np.sum(power_tether[mask]) * dt
    energy_mech_J = np.sum(power_mech[mask]) * dt

    print(f"\n{phase}")
    print(f"Tether energy: {energy_tether_J / 3.6e6:.3f} kWh")
    print(f"Mechanical energy: {energy_mech_J / 3.6e6:.3f} kWh")
    print(
        "Energy ratio:",
        energy_mech_J / energy_tether_J,
    )

'''