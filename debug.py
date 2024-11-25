import os
import sys

print("=== Current Directory ===")
print(os.getcwd())

print("\n=== Python Path ===")
print('\n'.join(sys.path))

print("\n=== Directory Contents ===")
for root, dirs, files in os.walk('.'):
    print(f"\nDirectory: {root}")
    print("Files:", files)
    print("Subdirs:", dirs) 