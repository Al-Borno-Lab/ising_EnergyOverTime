import os

def find_file_recursive(root_folder, file_name, current_depth=0, max_depth=None, verbose=True):
    """
    Recursively search for a file in all subdirectories
    
    Args:
        root_folder (str): Path to start searching from
        file_name (str): Name of the file to look for
        current_depth (int): Current recursion depth (for display purposes)
        max_depth (int): Maximum depth to search (None for unlimited)
        verbose (bool): Whether to print detailed output
    
    Returns:
        list: List of paths where the file was found
    """
    found_paths = []
    indent = "  " * current_depth
    
    # Check if we've reached max depth
    if max_depth is not None and current_depth > max_depth:
        return found_paths
    
    # Normalize the path to handle different path separators
    root_folder = os.path.normpath(root_folder)
    
    # Check if the folder exists and is accessible
    if not os.path.exists(root_folder):
        if verbose:
            print(f"{indent}❌ Folder does not exist: {root_folder}")
        return found_paths
    
    if not os.path.isdir(root_folder):
        if verbose:
            print(f"{indent}❌ Not a directory: {root_folder}")
        return found_paths
    
    try:
        items = os.listdir(root_folder)
    except PermissionError:
        if verbose:
            print(f"{indent}❌ Permission denied: {root_folder}")
        return found_paths
    except Exception as e:
        if verbose:
            print(f"{indent}❌ Error accessing {root_folder}: {e}")
        return found_paths
    
    if verbose:
        print(f"{indent}📁 Searching in: {root_folder}")
    
    # Check if the file exists in current directory
    file_path = os.path.join(root_folder, file_name)
    if os.path.isfile(file_path):  # Use isfile instead of exists to ensure it's a file
        if verbose:
            print(f"{indent}  ✅ Found '{file_name}'")
        found_paths.append(file_path)
    else:
        if verbose:
            print(f"{indent}  ❌ '{file_name}' not found")
    
    # Recursively search in subdirectories
    subdirs = []
    for item in items:
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    
    if verbose and subdirs:
        print(f"{indent}  📂 Found {len(subdirs)} subdirectories")
    
    for subdir in subdirs:
        try:
            subfolder_results = find_file_recursive(subdir, file_name, current_depth + 1, max_depth, verbose)
            found_paths.extend(subfolder_results)
        except Exception as e:
            if verbose:
                print(f"{indent}  ❌ Error processing {subdir}: {e}")
    
    return found_paths

def extract_results_folder(file_path):
    """
    Extract the folder name that ends with '_results' from a file path
    
    Args:
        file_path (str): Full path to a file
        
    Returns:
        str: The folder name ending with '_results', or None if not found
    """
    # Split the path into components
    path_parts = file_path.split(os.sep)
    
    # Look for a folder that ends with '_results'
    for part in path_parts:
        if part.endswith('_results'):
            return part
    
    return None

def ensure_directory_exists(directory_path):
    """
    Check if a directory exists and create it (including all parent directories) if it doesn't.
    
    Args:
        directory_path (str): Path to the directory to check/create
    
    Returns:
        bool: True if directory exists or was created successfully, False if creation failed
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory '{directory_path}': {e}")
        return False

