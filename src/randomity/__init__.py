from . import generate
from . import evaluate
from . import _utils

import importlib.metadata

try:
    __version__ = importlib.metadata.version("randomity")
except importlib.metadata.PackageNotFoundError:
    # running from source without an installed distribution (e.g. tests / CI)
    __version__ = "0.0.0+source"

def version():
    return __version__