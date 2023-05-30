import os

_USER_DIR = os.path.expanduser("~/")
_DISK_CACHE_DIR = os.path.join(_USER_DIR, ".peachdb")
os.makedirs(_DISK_CACHE_DIR, exist_ok=True)

SHELVE_DB = f"{_DISK_CACHE_DIR}/db"
BLOB_STORE = f"{_DISK_CACHE_DIR}/blobs"
CACHED_REQUIREMENTS_TXT = f"{_DISK_CACHE_DIR}/requirements.txt"

GIT_REQUIREMENTS_TXT = "https://raw.githubusercontent.com/peach-db/peachdb/master/requirements.txt"
