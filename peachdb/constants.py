import os

_USER_DIR = os.path.expanduser("~/")
_DISK_CACHE_DIR = os.path.join(_USER_DIR, ".peachdb")
os.makedirs(_DISK_CACHE_DIR, exist_ok=True)

SHELVE_DB = f"{_DISK_CACHE_DIR}/db"
BLOB_STORE = f"{_DISK_CACHE_DIR}/blobs"
