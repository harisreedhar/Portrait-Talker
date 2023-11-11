import os
import shutil

image_extensions = [
    ".bmp",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".tiff",
    ".tif",
    ".webp",
    ".pic",
    ".ico",
]

video_extensions = [
    ".avi",
    ".mkv",
    ".mp4",
    ".mov",
    ".wmv",
    ".flv",
    ".3gp",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".asf",
    ".ts",
    ".vob",
    ".webm",
    ".gif",
]

def is_file(file_path):
    if not file_path:
        return False
    return os.path.isfile(file_path)

def is_image(file_path):
    if is_file(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in image_extensions
    return False

def is_video(file_path):
    if is_file(file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in video_extensions
    return False

def create_directory(directory_path, remove_existing=True):
    if os.path.exists(directory_path) and remove_existing:
        shutil.rmtree(directory_path)

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
        return directory_path
    else:
        counter = 1
        while True:
            new_directory_path = f"{directory_path}_{counter}"
            if not os.path.exists(new_directory_path):
                os.mkdir(new_directory_path)
                return new_directory_path
            counter += 1