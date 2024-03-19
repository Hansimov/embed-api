from pathlib import Path
from tclogger import OSEnver

secrets_path = Path(__file__).parent / "secrets.json"
ENVS = OSEnver(secrets_path)
