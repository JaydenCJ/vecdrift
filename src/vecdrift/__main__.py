"""Allow ``python -m vecdrift`` as an alias for the ``vecdrift`` script."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
