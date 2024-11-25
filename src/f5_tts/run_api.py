import os
import sys
from pathlib import Path

# Debug prints
print("Current directory:", os.getcwd())
print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
print("sys.path:", sys.path)

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from f5_tts.infer.api import app
    import uvicorn
    
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=7860)
except ImportError as e:
    print(f"Import Error: {e}")
    print("\nDirectory contents:")
    for root, dirs, files in os.walk('.'):
        print(f"\nDirectory: {root}")
        print("Files:", files)
        print("Subdirs:", dirs) 