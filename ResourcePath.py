import os
import sys

def get_executable_dir() -> str:
    """Get the directory where the executable is located (for persistent data)"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.abspath(".")

def resource_path(relative_path: str) -> str:
    """ Get absolute path to resource, works for dev and for PyInstaller """
    # If path is already absolute, return it as-is
    if os.path.isabs(relative_path):
        return relative_path
    
    # Check if this is persistent data (identities, Images) or a subdirectory
    path_lower = relative_path.replace("\\", "/").lower()
    if "identities" in path_lower or "images" in path_lower:
        # Persistent data should be stored next to the executable
        base_path = get_executable_dir()
    else:
        # Packaged resources use MEIPASS
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)