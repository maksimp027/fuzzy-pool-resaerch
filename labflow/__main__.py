"""Allow running Labflow via `python -m labflow`."""

import sys

from labflow.cli import main

if __name__ == "__main__":
    sys.exit(main())
