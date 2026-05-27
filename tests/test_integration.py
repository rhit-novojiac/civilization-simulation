import subprocess
import os
import pytest

def test_full_simulation_100_ticks():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Run main.py with --fresh and --max-ticks 100
    # We do not use --log so it runs headless and faster
    result = subprocess.run(
        ["venv\\Scripts\\python.exe", "main.py", "--fresh", "--max-ticks", "100"],
        cwd=root_dir,
        capture_output=True,
        text=True
    )
    
    # Assert successful exit code
    assert result.returncode == 0, f"Simulation failed with error:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}"
