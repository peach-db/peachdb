import subprocess

from peachdb.constants import _DISK_CACHE_DIR


def sync_cache_dir_s3():
    subprocess.Popen(["aws", "s3", "sync", _DISK_CACHE_DIR, "s3://metavoice-vector-db/peachdb-cache/"])
