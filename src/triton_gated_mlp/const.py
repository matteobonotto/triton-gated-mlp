from pathlib import Path
import os

ROOT_PATH = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)))
).parent.parent  # pathlib.Path().cwd()
