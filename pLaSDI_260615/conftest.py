# -*- coding: utf-8 -*-
"""
conftest.py - pytest configuration.
Adds the project root to sys.path so src and config.py can be imported.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
