"""
test_integration.py — Integration tests for AeroNet Lite
==========================================================
Verifies that the main simulation orchestrator runs end-to-end.

Run with:
    python -m pytest tests/test_integration.py -v

Author: Member 1 -- The Architect
"""

import sys
import os
import matplotlib
matplotlib.use("Agg") # Headless mode

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.main import run_simulation

def test_simulation_run_no_crash():
    """Smoke test: ensure the 20-step simulation runs without exceptions."""
    try:
        run_simulation()
        success = True
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        success = False
    
    assert success is True

if __name__ == "__main__":
    test_simulation_run_no_crash()
    print("Integration test passed!")
