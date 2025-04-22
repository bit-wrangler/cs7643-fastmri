import os

def delete_zone_identifiers(path):
    """
    Deletes Zone.Identifier files recursively within the given path.

    Args:
        path (str): The root directory path to search within.
    """
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(":Zone.Identifier"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    target_directory = input("Enter the directory path to clean: ")
    if os.path.isdir(target_directory):
        delete_zone_identifiers(target_directory)
    else:
        print("Invalid directory path.")