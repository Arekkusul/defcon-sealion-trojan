"""Make the project root importable so `import sovereign` and
`from scripts import ...` work when tests run from anywhere."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
