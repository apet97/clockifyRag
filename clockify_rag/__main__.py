"""Entry point for running clockify_rag as a module.

Supports: python -m clockify_rag [sanity_check]
"""

import sys

if __name__ == "__main__":
    # Default to sanity_check if no args
    from . import sanity_check

    sys.exit(sanity_check.main())
