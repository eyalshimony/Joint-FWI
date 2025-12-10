import data
import os
import numpy as np
import concurrent.futures
import collections
import csv
import sys
import fileinput
import seismograms_handler
import pandas
from objects import CMTSolution
import obspy
from shutil import copyfile
import pickle
import re
import shutil

global sortedStations
global sortedFNs
global comp_seis_path
global fib_seis_path
global seismograms
global noise_seis_path
global projects_base_path
global project_name
global base_data_path
global specfem_base

projects_base_path = "/DATA/eyal/specfem2d/"
specfem_base = "/DATA/eyal/specfem3d/"
project_name = ""
base_data_path = "/DATA/eyal/"


def read_green_functions(name):
    green_functions = []
    for i in range(1, 4):
        strain_seismograms = read_SU_seismograms(name + f"/green{i}", "r")
        green_functions.append(seismograms_handler.calculate_DAS_seismograms(strain_seismograms))
    return green_functions


def change_source_coordinates(source_file_path: str, new_xs: float, new_zs: float):
    """
    Changes the 'xs' and 'zs' values in a SOURCE file to the given new coordinates.
    The function preserves comments and general formatting of the original lines.

    Args:
        source_file_path (str): The full path to the SOURCE file.
        new_xs (float): The new x-coordinate value for 'xs'.
        new_zs (float): The new z-coordinate value for 'zs'.
    """
    if not os.path.exists(source_file_path):
        print(f"Error: SOURCE file not found at '{source_file_path}'", file=os.sys.stderr)
        return

    lines = []
    try:
        with open(source_file_path, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error reading SOURCE file '{source_file_path}': {e}", file=os.sys.stderr)
        return

    modified_lines = []
    xs_found = False
    zs_found = False

    for line in lines:
        stripped_line = line.strip()

        # Check for 'xs' line
        if stripped_line.startswith("xs "):
            if not xs_found:  # Only modify the first occurrence
                # Split the line to preserve leading whitespace and comments
                parts = line.split('=')
                if len(parts) > 1:
                    # Reconstruct the line with the new xs value, preserving the comment
                    # Use ljust to maintain column alignment for the value
                    new_value_str = f"{new_xs:.1f}".ljust(15)  # Adjust ljust width if needed
                    modified_lines.append(f"{parts[0]}= {new_value_str} # source location x in meters\n")
                else:  # Fallback if line format is unexpected
                    modified_lines.append(line)
                xs_found = True
            else:
                modified_lines.append(line)  # Append subsequent 'xs' if they exist (unlikely for this file)

        # Check for 'zs' line
        elif stripped_line.startswith("zs "):
            if not zs_found:  # Only modify the first occurrence
                # Split the line to preserve leading whitespace and comments
                parts = line.split('=')
                if len(parts) > 1:
                    # Reconstruct the line with the new zs value, preserving the comment
                    new_value_str = f"{new_zs:.1f}".ljust(15)  # Adjust ljust width if needed
                    modified_lines.append(
                        f"{parts[0]}= {new_value_str} # source location z in meters (zs is ignored if source_surf is set to true, it is replaced with the topography height)\n")
                else:  # Fallback if line format is unexpected
                    modified_lines.append(line)
                zs_found = True
            else:
                modified_lines.append(line)  # Append subsequent 'zs' if they exist

        # For all other lines, append them as they are
        else:
            modified_lines.append(line)

    try:
        with open(source_file_path, 'w') as f:
            f.writelines(modified_lines)
        print(f"Successfully updated 'xs' to {new_xs:.1f} and 'zs' to {new_zs:.1f} in '{source_file_path}'")
    except IOError as e:
        print(f"Error writing to SOURCE file '{source_file_path}': {e}", file=os.sys.stderr)


def write_STATIONS(stations, path):
    with open(path + 'STATIONS', 'w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerows([(stationData.station, stationData.network, stationData.latitude, stationData.longitude,
                      stationData.elevation, stationData.burial) for stationData in stations])


def write_source_file(solution, name):
    """
    Writes the SOURCE file for a seismic simulation based on provided parameters.
    """

    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name

    # Define the fixed parameters based on the example file
    source_surf = '.false.'
    source_type = 2
    time_function_type = 5
    name_of_source_file = 'DATA/stf' # Only used if time_function_type = 8
    burst_band_width = 0.0         # Only used if time_function_type = 9
    anglesource = 0.0
    factor = 1.0
    vx = 0.0
    vz = 0.0

    # Construct the content of the SOURCE file
    # Use f-strings for easy formatting, maintaining the structure and comments
    source_content = f"""\
## Source 1
source_surf                               = {source_surf}      # source inside the medium, or source automatically moved exactly at the surface by the solver
xs                                        = {solution.xs:.1f}        # source location x in meters
zs                                        = {solution.zs:.1f}        # source location z in meters (zs is ignored if source_surf is set to true, it is replaced with the topography height)
## Source type parameters:
#  1 = elastic force or acoustic pressure
#  2 = moment tensor
# or Initial field type (when initialfield set in Par_file):
# For a plane wave including converted and reflected waves at the free surface:
#  1 = P wave,
#  2 = S wave,
#  3 = Rayleigh wave
# For a plane wave without converted nor reflected waves at the free surface, i.e. with the incident wave only:
#  4 = P wave,
#  5 = S wave
# For initial mode displacement:
#  6 = mode (2,3) of a rectangular membrane
source_type                               = {source_type}
# Source time function:
# In the case of a source located in an acoustic medium,
# to get pressure for a Ricker in the seismograms, here we need to select a Gaussian for the potential Chi
# used as a source, rather than a Ricker, because pressure = - Chi_dot_dot.
# This is true both when USE_TRICK_FOR_BETTER_PRESSURE is set to .true. or to .false.
# Options:
#  1 = second derivative of a Gaussian (a.k.a. Ricker),
#  2 = first derivative of a Gaussian,
#  3 = Gaussian,
#  4 = Dirac,
#  5 = Heaviside (4 and 5 will produce noisy recordings because of frequencies above the mesh resolution limit),
#  6 = ocean acoustics type I,
#  7 = ocean acoustics type II,
#  8 = external source time function = 8 (source read from file),
#  9 = burst,
# 10 = Sinus source time function,
# 11 = Marmousi Ormsby wavelet
time_function_type = {time_function_type}
# If time_function_type == 8, enter below the custom source file to read (two columns file with time and amplitude) :
# (For the moment dt must be equal to the dt of the simulation. File name cannot exceed 150 characters)
# IMPORTANT: do NOT put quote signs around the file name, just put the file name itself otherwise the run will stop
name_of_source_file = {name_of_source_file}  # Only for option 8 : file containing the source wavelet
burst_band_width                          = {burst_band_width:.1f}      # Only for option 9 : band width of the burst
f0                                        = {solution.f0:.1f}      # dominant source frequency (Hz) if not Dirac or Heaviside
tshift                                    = {solution.tshift:.4f}        # time shift when multi sources (if one source, must be zero)
## Force source
# angle of the source (for a force only); for a plane wave, this is the incidence angle; for moment tensor sources this is unused
anglesource                               = {anglesource:.1f}
## Moment tensor
# The components of a moment tensor source must be given in N.m, not in dyne.cm as in the DATA/CMTSOLUTION source file of the 3D version of the code.
Mxx = {solution.Mxx:.3e}     # Mxx component (for a moment tensor source only)
Mzz = {solution.Mzz:.3e}     # Mzz component (for a moment tensor source only)
Mxz = {solution.Mxz:.3e}     # Mxz component (for a moment tensor source only)
## Amplification (factor to amplify source time function)
factor                                    = {factor:.1f}    # amplification factor
## Moving source parameters
vx                                        = {vx:.1f}        # Horizontal source velocity (m/s)
vz                                        = {vz:.1f}        # Vertical source velocity (m/s)
"""

    # Write the content to the file named SOURCE
    file_path = path + '/DATA/SOURCE'
    with open(file_path, 'w') as f:
        f.write(source_content)


# --- Main Workflow Function ---
def manage_parallel_simulations(name, base_dir="."):
    global projects_base_path
    global project_name
    """
    Manages the setup, execution, and restoration of parallel simulations.
    """
    base_dir = projects_base_path + project_name + name
    print("--- Starting Simulation Workflow Management (Parent Script Mode) ---")

    original_run_count = 20
    green_per_run = 3
    total_simulations = original_run_count * green_per_run  # 20 * 3 = 60

    # Define new naming scheme for original 'run' folders to avoid conflicts
    # e.g., run0001 -> run0101, run0020 -> run0120
    original_to_temp_offset = 100

    # Store paths for restoration
    original_run_paths = [os.path.join(base_dir, f"run{i:04d}") for i in range(1, original_run_count + 1)]
    temp_run_paths = [os.path.join(base_dir, f"run{i + original_to_temp_offset:04d}") for i in
                      range(1, original_run_count + 1)]

    # Store mapping for moving green folders back during restoration
    # new_run_name (e.g., "run0001") -> (temp_parent_run_name (e.g., "run0101"), original_green_name (e.g., "green1"))
    new_sim_to_original_green_map = {}  # This will store the *forward* mapping

    # Backup original run_this_example.sh content (from parent directory)
    master_script_path = os.path.join(base_dir, "run_this_example.sh")
    original_master_script_content = None

    if os.path.exists(master_script_path):
        with open(master_script_path, 'r') as f:
            original_master_script_content = f.read()
        print(f"Backed up original 'run_this_example.sh' content from '{master_script_path}'")
    else:
        print(
            f"Error: 'run_this_example.sh' not found at '{master_script_path}'. Cannot proceed without it for restoration.",
            file=sys.stderr)
        return

    try:
        # --- Phase 1: Rename current run0001-run0020 to run0101-run0120 ---
        print("\n--- Phase 1: Renaming original 'runXXXX' directories ---")
        for i in range(original_run_count):
            old_path = original_run_paths[i]
            new_path = temp_run_paths[i]
            if os.path.isdir(old_path):
                os.rename(old_path, new_path)
                print(f"  Renamed '{os.path.basename(old_path)}' to '{os.path.basename(new_path)}'")
            else:
                print(f"  Warning: Original directory '{old_path}' not found. Skipping rename.", file=sys.stderr)

        # --- Phase 2: Move 'greenX' folders to top-level 'runYYYY' ---
        print("\n--- Phase 2: Moving 'greenX' folders to top-level 'runYYYY' ---")
        current_new_run_idx = 1
        green_subfolders = ["green1", "green2", "green3"]

        for i_orig_run in range(1, original_run_count + 1):
            temp_parent_run_name = f"run{i_orig_run + original_to_temp_offset:04d}"
            temp_parent_run_path = os.path.join(base_dir, temp_parent_run_name)

            if not os.path.isdir(temp_parent_run_path):
                print(
                    f"  Warning: Temporary run directory '{temp_parent_run_path}' not found. Skipping its green folders.")
                continue

            for green_name in green_subfolders:
                green_source_path = os.path.join(temp_parent_run_path, green_name)
                new_top_level_run_name = f"run{current_new_run_idx:04d}"
                new_top_level_run_path = os.path.join(base_dir, new_top_level_run_name)

                if os.path.isdir(green_source_path):
                    # Move the green folder to the top level and rename it
                    shutil.move(green_source_path, new_top_level_run_path)
                    print(f"  Moved '{os.path.basename(green_source_path)}' to '{new_top_level_run_name}'")

                    # Store mapping for restoration: new_top_level_run_name -> (temp_parent_run_name, green_name)
                    new_sim_to_original_green_map[new_top_level_run_name] = (temp_parent_run_name, green_name)
                    current_new_run_idx += 1
                else:
                    print(f"  Warning: Green directory '{green_source_path}' not found. Skipping move.",
                          file=sys.stderr)

        # --- Phase 3: Modify the master run_this_example.sh script ---
        print("\n--- Phase 3: Modifying master 'run_this_example.sh' for 60 runs ---")
        modified_script_content = original_master_script_content

        # Replace 'start=1' - typically already 1, but for robustness
        modified_script_content = re.sub(r'start=\s*\d+', 'start=1', modified_script_content)

        # Replace 'end=20' with 'end=60'
        modified_script_content = re.sub(r'end=\s*\d+', f'end={total_simulations}', modified_script_content)

        # Replace 'MPIPROC=$(( 20*$NPROC ))' with 'MPIPROC=$(( 60*$NPROC ))'
        modified_script_content = re.sub(
            r'MPIPROC=\$\(\(\s*\d+\*\$NPROC\s*\)\)',
            f'MPIPROC=$(( {total_simulations}*$NPROC ))',
            modified_script_content
        )

        # Remove the internal 'mkdir' and 'rm -rf' loops within the script
        # These operations were part of the original script that set up individual runs
        # We need to remove the loop that sets up the individual directories if the master script
        # is now just orchestrating the already prepared run directories (which are the moved greens).
        # We target the specific 'for' loop for cleanup based on its content.
        loop_pattern = r'for i in \$\(seq -f "run%04g" \$start \$end\); do\s*\n\s*# Create OUTPUT_FILES directory\s*\n\s*mkdir -p "\$i/OUTPUT_FILES"\s*\n\s*# Remove any existing files inside OUTPUT_FILES\s*\n\s*rm -rf "\$i/OUTPUT_FILES"/\*\s*\n\s*# Copy setup files\s*\n\s*cp "\$i/DATA/Par_file" "\$i/OUTPUT_FILES/"\s*\n\s*cp "\$i/DATA/SOURCE" "\$i/OUTPUT_FILES/"\s*\n\s*done'

        modified_script_content = re.sub(
            loop_pattern,
            '',  # Replace with empty string
            modified_script_content,
            flags=re.DOTALL  # Allow . to match newlines
        )
        # Also remove any leftover newlines if the above results in too many
        modified_script_content = re.sub(r'\n{2,}', '\n\n',
                                         modified_script_content)  # Reduce multiple newlines to max two

        with open(master_script_path, 'w') as f:
            f.write(modified_script_content)
        print(f"  Modified master 'run_this_example.sh' at '{master_script_path}'")

        # --- Phase 4: Run the simulation ---
        print("\n--- Phase 4: Running the simulations (calling run_modelling) ---")
        # Change to the base directory before running the modelling function
        # This assumes run_modelling executes 'run_this_example.sh' from base_dir
        original_cwd = os.getcwd()
        os.chdir(base_dir)
        run_modelling_clean(name)  # Call your simulation function
        os.chdir(original_cwd)  # Change back to original working directory
    except Exception as e:
        print(f"\nAn error occurred during workflow execution: {e}", file=sys.stderr)
        print("Attempting to restore original state...", file=sys.stderr)
    finally:
        print("\n--- Phase 5: Restoring original state ---")

        # Move the new top-level run folders back into their original green structure
        # This is the reverse of Phase 2
        print("  Moving new top-level 'runYYYY' directories back to their 'greenX' origins...")
        for new_run_name, (temp_parent_run_name, green_name) in new_sim_to_original_green_map.items():
            new_top_level_run_path = os.path.join(base_dir, new_run_name)
            original_parent_run_path = os.path.join(base_dir, temp_parent_run_name)
            destination_green_path = os.path.join(original_parent_run_path, green_name)

            # Ensure the parent directory (e.g., run0101) exists before moving the green folder back into it
            os.makedirs(original_parent_run_path, exist_ok=True)  # Will not create if exists

            if os.path.isdir(new_top_level_run_path):
                try:
                    shutil.move(new_top_level_run_path, destination_green_path)
                    print(
                        f"    Moved '{os.path.basename(new_top_level_run_path)}' back to '{os.path.relpath(destination_green_path, base_dir)}'")
                except Exception as e:
                    print(f"    Error moving '{new_top_level_run_path}' back to '{destination_green_path}': {e}",
                          file=sys.stderr)
            else:
                print(
                    f"    Warning: '{new_top_level_run_path}' not found for restoration (might have been processed or moved by run_modelling). Skipping move back.",
                    file=sys.stderr)

        # Restore the original 'run0001'-'run0020' names from 'run0101'-'run0120'
        print("  Renaming temporary 'run01XX' directories back to original 'run00XX' names...")
        # Iterate backwards to avoid conflicts if paths overlap (though not strictly necessary here)
        for i in range(original_run_count - 1, -1, -1):
            temp_path = temp_run_paths[i]
            original_path = original_run_paths[i]
            if os.path.isdir(temp_path):  # Only rename if the temporary directory still exists
                try:
                    os.rename(temp_path, original_path)
                    print(f"    Renamed '{os.path.basename(temp_path)}' back to '{os.path.basename(original_path)}'")
                except Exception as e:
                    print(f"    Error renaming '{temp_path}' back to '{original_path}': {e}", file=sys.stderr)
            # else:
            # This case is normal if the directory was already processed or missing
            # print(f"    Warning: Temporary directory '{temp_path}' not found during final rename. Skipping.", file=sys.stderr)

        # Restore the master run_this_example.sh script content (in base_dir)
        print("  Restoring original master 'run_this_example.sh'...")
        if master_script_path and original_master_script_content is not None:
            try:
                with open(master_script_path, 'w') as f:
                    f.write(original_master_script_content)
                print(f"    Restored '{os.path.basename(master_script_path)}'")
            except Exception as e:
                print(f"    Error restoring '{master_script_path}': {e}", file=sys.stderr)
        else:
            print(f"    No original content for '{master_script_path}' to restore (might have been missing initially).")

    print("\n--- Simulation Workflow Management Complete (and restored) ---")


def run_modelling_clean(name):
    global projects_base_path
    global project_name

    curr_dir = os.getcwd()
    os.chdir(projects_base_path + project_name + name)
    os.system("./run_this_example.sh")
    os.chdir(curr_dir)


def run_modelling(name):
    global projects_base_path
    global project_name

    curr_dir = os.getcwd()
    os.chdir(projects_base_path + project_name + name)
    os.system("./change_simulation_type.pl -f")
    for line in fileinput.input(projects_base_path + project_name + name + "/DATA/Par_file", inplace=1):
        if "SAVE_FORWARD" in line:
            line = line.replace("true", "false")
        if "APPROXIMATE_HESS_KL" in line:
            line = line.replace("true", "false")
        if "UNDO_ATTENUATION_AND_OR_PML" in line:
            line = line.replace("true", "false")
        sys.stdout.write(line)
    os.system("./run_this_example.sh")
    os.chdir(curr_dir)


def run_modelling_for_FWI_multi(names):
    global projects_base_path
    global project_name

    curr_dir = os.getcwd()

    name = names[0].split("/")[0]
    os.chdir(projects_base_path + project_name + name)
    os.system("./change_simulation_type.pl -f")
    for line in fileinput.input(projects_base_path + project_name + name + "/DATA/Par_file", inplace=1):
        if "SAVE_FORWARD" in line:
            line = line.replace("false", "true")
        if "APPROXIMATE_HESS_KL" in line:
            line = line.replace("true", "false")
        if "seismotype" in line:
            line = line.replace("1", "9")
        sys.stdout.write(line)
    os.chdir(projects_base_path + project_name + name)
    os.system("./run_this_example.sh")
    os.chdir(curr_dir)


def write_adjoint_sources_to_SU_files(adjoint_sources, name):
    global projects_base_path
    global project_name

    adjoint_sources.sort(keys=['station'])
    for tr in adjoint_sources:
        tr.data = tr.data.astype(np.float32)
    adjoint_sources_x = adjoint_sources.select(channel="X")
    adjoint_sources_z = adjoint_sources.select(channel="Z")
    adjoint_sources_x.write(projects_base_path + project_name + name + "/SEM/Ux_file_single.su.adj", format="SU",
                            byteorder="<")
    adjoint_sources_z.write(projects_base_path + project_name + name + "/SEM/Uz_file_single.su.adj", format="SU",
                            byteorder="<")


def fill_trace_headers_partial(file, stf, path, time_shift):
    global projects_base_path
    global project_name

    curr_seis = obspy.read(path + file, format="SU", unpack_trace_headers=True)
    stations = data.get_stations()
    for trace in curr_seis:
        station = stations[trace.stats.su['trace_header']['trace_sequence_number_within_line'] - 1]
        trace.stats.network = station.network
        trace.stats.station = station.station
        trace.stats.starttime = stf.stats.starttime + time_shift
        trace.stats.channel = file[1].capitalize()
        trace.stats.location = ";long=" + str(station.longitude) + ";bur=" + str(station.burial)
    return curr_seis


def read_SU_seismograms(name, seismogram_type, time_shift=0):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/OUTPUT_FILES/"
    files = os.listdir(path)
    seismograms = obspy.Stream()
    stf = read_source_time_function_orig(name)
    n_workers = 10
    filtered_files = [file for file in files if file.endswith(".su") and f"_{seismogram_type}" in file]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(fill_trace_headers_partial, file, stf, path, time_shift)
                   for file in filtered_files]
        for future in futures:
            seismograms += future.result()
    seismograms.sort(keys=['channel'])
    seismograms.sort(keys=['station'])
    return seismograms


def read_observed_seismograms_from_pickle(name):
    global base_data_path
    global project_name
    global noise_seis_path

    with open(f"{projects_base_path}{project_name}{name.split('/')[0]}/observed_seismograms{name[-4:]}.pk", "rb") as f:
        seismograms = pickle.load(f)
    return seismograms


def run_structural_adjoint_modelling_multi(names):
    global projects_base_path
    global project_name

    curr_dir = os.getcwd()

    name = names[0].split("/")[0]
    os.chdir(projects_base_path + project_name + name)
    os.system("./change_simulation_type.pl -b")
    for line in fileinput.input(projects_base_path + project_name + name + "/DATA/Par_file", inplace=1):
        if "APPROXIMATE_HESS_KL" in line:
            line = line.replace("false", "true")
        if "UNDO_ATTENUATION_AND_OR_PML" in line:
            line = line.replace("false", "true")
        if "SAVE_FORWARD" in line:
            line = line.replace("true", "false")
        if "seismotype" in line:
            line = line.replace("9", "1")
        sys.stdout.write(line)
    os.chdir(projects_base_path + project_name + name)
    os.system("./run_pure_solver.sh")
    os.chdir(curr_dir)


def read_source_location_kernels(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/OUTPUT_FILES/"
    with open(path + "src_frechet.000001", "r") as f:
        kernels = f.read().splitlines()
    return float(kernels[3]), float(kernels[4])


def read_adjoint_strain_tensor_at_source(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/OUTPUT_FILES/"
    files = os.listdir(path)
    adjoint_strain_tensor = obspy.Stream()
    for fn in files:
        if "NT.S" and ".sem" in fn:
            fn_parts = fn.split(".")
            if fn_parts[2][0] == "S":
                with open(path + fn, 'r') as f:
                    data = f.readlines()
                    data = [line.split() for line in data]
                    data = [list(map(float, i)) for i in data]
                    data = np.asarray(data)
                    dt = data[1, 0] - data[0, 0]
                    srate = 1 / dt
                    data = obspy.Trace(data[:, 1],
                                       {'sampling_rate': srate, 'delta': dt, 'network': "ET", 'station': name,
                                        'channel': fn_parts[2][1:3], 'npts': len(data[:, 1]),
                                        'starttime': obspy.UTCDateTime(data[0, 0]),
                                        'endtime': obspy.UTCDateTime(data[-1, 0])})
                    adjoint_strain_tensor.append(data)

    return adjoint_strain_tensor


def run_green_modelling(name):
    for i in range(3):
        run_modelling(name + "/green" + str(i+1))


def run_green_modelling_parallel(name):
    names = []
    for i in range(6):
        names.append(name + "/green" + str(i+1))
        for line in fileinput.input(projects_base_path + project_name + names[i] + "/DATA/Par_file", inplace=1):
            if "NUMBER_OF_SIMULTANEOUS_RUNS" in line:
                line = line.replace("2", "1")
            sys.stdout.write(line)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        collections.deque(executor.map(run_modelling, names), maxlen=0)


def read_source_time_function(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/OUTPUT_FILES/plot_source_time_function.txt"
    with open(path, "r") as f:
        lines = f.readlines()
        line_parts = [line.split() for line in lines]
        line_parts = [list(map(float, i)) for i in line_parts]
        data = np.asarray(line_parts)
        dt = data[1,0] - data[0,0]
        start_time = obspy.UTCDateTime(data[0,0])
        end_time = obspy.UTCDateTime(data[-1, 0])
        return obspy.Trace(data[:, 1], {'sampling_rate': 1/dt, 'delta': dt, 'station': name, 'npts': len(lines),
                                        'starttime': start_time,'endtime': end_time})


def read_source_time_function_orig(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/DATA/stf"
    with open(path, "r") as f:
        lines = f.readlines()
        line_parts = [line.split() for line in lines]
        line_parts = [list(map(float, i)) for i in line_parts]
        data = np.asarray(line_parts)
        dt = data[1,0] - data[0,0]
        start_time = obspy.UTCDateTime(data[0,0])
        end_time = obspy.UTCDateTime(data[-1, 0])
        return obspy.Trace(data[:, 1], {'sampling_rate': 1/dt, 'delta': dt, 'station': name, 'npts': len(lines),
                                        'starttime': start_time,'endtime': end_time})


def copy_source_time_function(name):
    global projects_base_path
    global project_name

    in_path = projects_base_path + project_name + name + "/OUTPUT_FILES/plot_source_time_function.txt"
    out_path = projects_base_path + project_name + name + "/plot_source_time_function.txt"
    copyfile(in_path, out_path)


def read_tomographic_models(name, it_num):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/DATA/"
    fns = os.listdir(path)
    if it_num < 0:
        fn = "tomo_file.xyz"
    else:
        fn = f"tomo_file_it{it_num}.xyz"

    tomo_models = pandas.read_csv(path + fn, skiprows=3, delimiter=" ", usecols=range(5)).values
    return tomo_models


def write_raw_model(model, name, it_num=-1):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/DATA/"
    xs = list(set(model[:, 0]))
    zs = list(set(model[:, 1]))
    xs.sort()
    zs.sort()
    min_x = min(xs)
    max_x = max(xs)
    min_z = min(zs)
    max_z = max(zs)
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]
    nx = len(xs)
    nz = len(zs)
    min_vp = min(model[:, 2])
    min_vs = min(model[:, 3])
    min_rho = min(model[:, 4])
    max_vp = max(model[:, 2])
    max_vs = max(model[:, 3])
    max_rho = max(model[:, 4])
    text_to_write = str(min_x) + " " + str(min_z) + " " + str(max_x) + " " + str(max_z) + str("\n") + str(dx) + " " + \
                    str(dz) + "\n" + str(nx) + " " + str(nz) + "\n" + str(min_vp) + " " + str(max_vp) + " " + \
                    str(min_vs) + " " + str(max_vs) + " " + str(min_rho) + " " + str(max_rho)
    fn = "tomo_file.xyz"
    np.savetxt(path + fn, model, header=text_to_write, comments='')
    if it_num >= 0:
        fn = "tomo_file_it" + str(it_num) + ".xyz"
        np.savetxt(path + fn, model, header=text_to_write, comments='')


def write_tomographic_file_xyz(name, vp, vs, rho, x0, z0, dx, dz):
    end_x = x0 + dx * (np.ma.size(vp, 0) - 1)
    end_z = z0 + dz * (np.ma.size(vp, 1) - 1)
    nx = np.ma.size(vp, 0)
    nz = np.ma.size(vp, 1)
    vp_min = np.min(vp)
    vp_max = np.max(vp)
    vs_min = np.min(vs)
    vs_max = np.max(vs)
    rho_min = np.min(rho)
    rho_max = np.max(rho)

    text_to_write = str(x0) + " " + str(z0) + " " + str(end_x) + " " + str(end_z) + \
                    str("\n") + str(dx) + " " + str(dz) + "\n" + str(nx) + " " + \
                    str(nz) + "\n" + str(vp_min) + " " + str(vp_max) + " " + str(vs_min) + " " + str(vs_max) + " " + \
                    str(rho_min) + " " + str(rho_max) + "\n"
    for i in range(np.ma.size(vp, 1)):
        for k in range(np.ma.size(vp, 0)):
            text_to_write += f"{x0 + k * dx} {z0 + i * dz} {vp[k, i]} {vs[k, i]} {rho[k, i]}\n"

    with open(projects_base_path + project_name + name + "/DATA/tomo_file" + ".xyz", "w") as f:
        f.write(text_to_write)


def write_iteration_model(name, it_num):
    global projects_base_path
    global project_name

    curr_path = projects_base_path + project_name + name + "/DATA/"
    dir_files = os.listdir(curr_path)
    rel_files = [fn for fn in dir_files if "tomo_file" in fn and "_it" not in fn]
    copyfile(curr_path + "tomo_file.xyz", curr_path + f"tomo_file_it{it_num}.xyz")


def models_to_1d_vector(model):
    orig_model_list = []
    orig_model_list.extend(list(model[:, 2]))
    orig_model_list.extend(list(model[:, 3]))
    orig_model_list.extend(list(model[:, 4]))
    orig_model = np.asarray(orig_model_list)
    return orig_model


def write_kernel(name, kernel, kernel_name):
    global projects_base_path
    global project_name

    if not os.path.isdir(projects_base_path + project_name + name + "/KERNELS/"):
        os.mkdir(projects_base_path + project_name + name + "/KERNELS/")
    with open(projects_base_path + project_name + name + "/KERNELS/" + kernel_name + ".pk", "wb") as f:
        pickle.dump(kernel, f)


def get_run_folders(directory):
    # Regular expression pattern: "run" followed by exactly four digits
    pattern = re.compile(r'^run\d{4}$')

    # List all items in the directory and filter for matching folders
    run_folders = [
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name)) and pattern.match(name)
    ]

    return run_folders


def get_single_kernel(name, folder):
    kernel = combine_kernel("/".join([name, folder]))
    for j in range(2, 5):
        rehsaped_kernel = np.reshape(kernel[:, j], (1089, 1089))
        rehsaped_kernel[:13, :] = 0
        rehsaped_kernel[-13:, :] = 0
        rehsaped_kernel[:, :13] = 0
        kernel[:, j] = rehsaped_kernel.flatten()
    return kernel


def get_kernels(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name
    run_folders = get_run_folders(path)
    run_folders.sort()

    averaged_kernels_all_sources = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = []
        for i in range(len(run_folders)):
            futures.append(executor.submit(get_single_kernel, name, run_folders[i]))
        results = [future.result() for future in futures]
        for kernel in results:
            averaged_kernels_all_sources.append(kernel)
    return averaged_kernels_all_sources


def get_single_hessian(name, folder):
    hessian = combine_hessian("/".join([name, folder]))
    for j in range(2, 3):
        rehsaped_kernel = np.reshape(hessian[:, j], (1089, 1089))
        rehsaped_kernel[:13, :] = 0
        rehsaped_kernel[-13:, :] = 0
        rehsaped_kernel[:, :13] = 0
        hessian[:, j] = rehsaped_kernel.flatten()
    return np.abs(hessian)


def get_hessians(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name
    run_folders = get_run_folders(path)
    run_folders.sort()

    averaged_hessian_all_sources = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = []
        for i in range(len(run_folders)):
            futures.append(executor.submit(get_single_hessian, name, run_folders[i]))
        results = [future.result() for future in futures]
        for hessian in results:
            averaged_hessian_all_sources.append(hessian)
    return averaged_hessian_all_sources


def combine_kernel(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/OUTPUT_FILES/"
    fns = os.listdir(path)
    fns_filtered = [fn for fn in fns if "alpha" in fn]

    kernels_list_form = []
    for fn in fns_filtered:
        kernels_list_form.extend(np.loadtxt(path + fn))
    kernels_list_form = np.array(kernels_list_form)

    xz, indices, inverse = np.unique(kernels_list_form[:, :2], axis=0, return_index=True, return_inverse=True)
    sums = np.zeros((len(xz), 3))  # Columns for A, B, C
    counts = np.zeros(len(xz))
    np.add.at(sums, inverse, kernels_list_form[:, 2:])  # Summing A, B, C for each unique (x, y)
    np.add.at(counts, inverse, 1)
    averaged_values = sums / counts[:, None]
    averaged_kernels = np.hstack((xz, averaged_values))

    averaged_kernels = averaged_kernels[averaged_kernels[:, 1].argsort()]
    averaged_kernels = averaged_kernels[averaged_kernels[:, 0].argsort(kind='mergesort')]

    return averaged_kernels


def combine_hessian(name):
    global projects_base_path
    global project_name

    path = projects_base_path + project_name + name + "/OUTPUT_FILES/"
    fns = os.listdir(path)
    fns_filtered = [fn for fn in fns if "Hessian2" in fn]
    kernels_list_form = []
    for fn in fns_filtered:
        kernels_list_form.extend(np.loadtxt(path + fn))
    kernels_list_form = np.array(kernels_list_form)

    xz, indices, inverse = np.unique(kernels_list_form[:, :2], axis=0, return_index=True, return_inverse=True)
    sums = np.zeros((len(xz), 1))  # Columns for A, B, C
    counts = np.zeros(len(xz))
    np.add.at(sums, inverse, kernels_list_form[:, 2:])  # Summing A, B, C for each unique (x, y)
    np.add.at(counts, inverse, 1)
    averaged_values = sums / counts[:, None]
    averaged_kernels = np.hstack((xz, averaged_values))

    averaged_kernels = averaged_kernels[averaged_kernels[:, 1].argsort()]
    averaged_kernels = averaged_kernels[averaged_kernels[:, 0].argsort(kind='mergesort')]

    return averaged_kernels


def parse_source_file(file_path, event_name):
    # Initialize an empty dictionary to hold the values
    data = {}

    # Regular expressions to capture parameter values
    pattern = re.compile(r'(\w+)\s*=\s*([-\d.Ee+]+)')

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                key, value = match.groups()
                try:
                    data[key] = float(value)
                except ValueError:
                    pass  # Skip if the value isn't a valid float

    # Extract values and apply the multiplication with the factor for moment tensor components
    factor = data.get('factor', 1.0)
    cmt_solution = CMTSolution(
        event_name=event_name,
        xs=data.get('xs', 0.0),
        zs=data.get('zs', 0.0),
        f0=data.get('f0', 0.0),
        tshift=data.get('tshift', 0.0),
        Mxx=data.get('Mxx', 0.0) * factor,
        Mzz=data.get('Mzz', 0.0) * factor,
        Mxz=data.get('Mxz', 0.0) * factor
    )

    return cmt_solution