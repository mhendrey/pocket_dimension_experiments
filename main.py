"""
Legacy compatibility wrapper for main.py

This file maintains backward compatibility with existing scripts and tests.
New code should use the src/ module structure instead.
"""

from __future__ import annotations

import json
from pathlib import Path




















def main() -> None:
    """
    Legacy compatibility wrapper.
    
    This function delegates to the new src/experiment module for actual execution,
    maintaining backward compatibility with existing scripts and tests.
    """
    from src.experiment import run_experiment
    
    print("\nRunning full experiment with all 500K documents...")
    summary = run_experiment(Path("artifacts_full"), d=128)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
