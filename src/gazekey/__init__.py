from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gazekey")
except PackageNotFoundError:
    __version__ = "unknown"
