import os

def create_directory(directory_path: str) -> bool:
    """Create a directory if it does not exist.

    Args:
        directory_path (str): The path of the directory to create.

    Returns:
        bool: True if the directory was created, False if it already existed.
    
    Raises:
        OSError: If an error occurs while creating the directory.
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except OSError as e:
        print(f"Error creating directory: {e}")
        return False
    
def get_last_element_from_path(file_path: str) -> str:
    """Extract the last element from a given file path.

    Args:
        file_path (str): The file path from which to extract the last element.

    Returns:
        str: The last element of the path, or an empty string if the path is empty.
    """
    return os.path.basename(file_path)