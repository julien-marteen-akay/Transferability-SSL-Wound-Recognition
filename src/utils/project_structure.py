import os
from pathlib import Path
from datetime import datetime


def extend_directory(path):
    """
    Creates directories (also nested if needed) without overwriting already existing directories

    Args:
      path: The path to the folder, which shall be created and also accepts nested dictionaries, which haven't been created yet.
    
    Returns:
      path: Just for convenience the path is returned again.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_current_datetime():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_experiment_folder(path, custom_name=False):
    """
    Creates a folder named experiment_ and also attaches the current datetime to it, inside the folder specified via path,
    which also gets created if it doesn't exist (not existing nested dictionaries are also okay)

    Returns:
      The full path to the newly created folder
    """
    time_stamp = get_current_datetime()
    custom_name = f"experiment_{time_stamp}" if not custom_name else f"{custom_name}_{time_stamp}"
    path = os.path.join(path, custom_name)
    extend_directory(path)
    return path
