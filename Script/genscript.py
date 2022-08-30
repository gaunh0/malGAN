import numpy as np
import PyInstaller.__main__

get_computer_name = """
# Get computer name
print("Computer Name:", os.getenv("COMPUTERNAME", "Unknown"))
"""

download_file = """
# Download file
print("IP information stored at", urlretrieve("https://ipinfo.io")[0])
"""

get_free_disk_space = """
# Get free disk space
free_bytes = ctypes.c_ulonglong(0)
ctypes.windll.kernel32.GetDiskFreeSpaceExW(
    ctypes.c_wchar_p(str(WINDOWS_PATH)), None, None, ctypes.pointer(free_bytes)
)
free_gb = free_bytes.value / 1024**3
print("Free disk space:", free_gb, "GB")
"""

remove_directory = """
# Remove directory
Path(tempfile.mkdtemp()).rmdir()
"""

get_temp_dir = """
# Get Temp Path
print("Temp Path:", tempfile.gettempdir())
"""

get_username = """
# Get Username
print("Username:", os.getenv("username", "Unknown"))
"""

get_file_info = """
# Get file info
print("explorer.exe info:", next(WINDOWS_PATH.glob("explorer.exe")).stat())
"""

get_system_directory = """
# Get system directory
(WINDOWS_PATH / "System32").exists()
"""

copy_file = """
# Copy file
copyfile(str(DUMMY_FILE), tempfile.mktemp())
"""

create_directory = """
# Create directory
tempfile.mkdtemp()
"""

terminate_process = """
# Terminate process
subprocess.run(["taskkill", "/F", "/im", "chrome.exe"])
"""

delete_file = """
# Delete file
f, p = tempfile.mkstemp()
open(f).close()
Path(p).unlink()
"""

set_file_time = """
# Change file access time
DUMMY_FILE.touch()
"""

get_time = """
# Get time
print("Current time:", time.strftime("%X %z"))
"""

get_short_path_name = """
# Get short path name
DUMMY_FILE.relative_to(Path("C:"))
"""

read_file = """
# Read file
DUMMY_FILE.read_bytes()
"""

write_file = """
# Write file
Path(os.environ["appdata"] + "/Microsoft/Windows/Start Menu/Programs/Startup/malgan.txt").write_text("Malgan")
"""

write_console = """
# Write to console
print("Malgan")
"""

wait_for_enter = """
# Wait for user
input("Press Enter to exit. ")
"""

array_map = {  # maps array indices to code requiring specific Windows APIs
    0: terminate_process,
    1: get_file_info,
    8: write_console,
    11: get_short_path_name,
    20: get_temp_dir,
    22: get_file_info,
    23: get_file_info,
    28: get_file_info,
    34: create_directory,
    37: get_system_directory,
    42: get_file_info,
    44: get_file_info,
    49: get_time,
    57: delete_file,
    58: get_file_info,
    59: write_file,
    60: read_file,
    63: get_file_info,
    75: write_file,
    83: get_computer_name,
    84: get_file_info,
    91: read_file,
    92: read_file,
    93: get_system_directory,
    95: get_system_directory,
    113: get_file_info,
    117: get_computer_name,
    121: get_time,
    127: get_file_info,
    125: copy_file,
    134: write_console,
    140: get_system_directory,
    149: get_username,
    153: get_file_info,
    156: get_username,
    161: set_file_time,
    162: copy_file,
    165: copy_file,
    167: get_username,
    168: get_username,
    178: remove_directory,
    183: get_free_disk_space,
    184: remove_directory,
    210: get_system_directory,
    214: download_file,
    215: get_free_disk_space,
    242: create_directory,
    248: download_file,
    254: delete_file,
    259: get_file_info
}

get_command = np.vectorize(lambda i: array_map.get(i, ""))


def generate(array):
    import time
    import os.path

    apis, = np.where(array == 1)
    commands = get_command(apis) if len(apis) else []
    commands = np.unique(commands)
    commands = "".join(commands)
    commands += wait_for_enter

    skeleton = __file__.replace("genscript", "skeleton")

    with open(skeleton) as skel, open(time.strftime("gen_%Y%m%d%H%M%S.py"), "w") as gen:
        gen.write(skel.read() + commands)
        exe = gen.name.replace(".py", ".exe")
        PyInstaller.__main__.run([
            "--clean",
            "--onefile",
            # "--noconsole",  # Uncomment to run malware silently
            "--log-level=ERROR",
            "--name=" + exe,
            gen.name
        ])
        print("EXE can be found at", os.path.join("dist", exe))

        print("Done!")


if __name__ == "__main__":
    # Generate array with all the commands, for testing
    array = np.array([int(bool(i in array_map)) for i in range(max(array_map) + 1)])
    generate(array)
