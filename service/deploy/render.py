"""Generate reviewable macOS launchd/Caddy files; never install or start services."""

import argparse
import getpass
import os
import plistlib
import re
from pathlib import Path

from dotenv import load_dotenv

from service.config import ROOT


def render(domain: str, root: Path, models: Path, user: str, home: Path, output: Path) -> None:
    if not re.fullmatch(r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}", domain):
        raise ValueError("Use a public DNS hostname, without scheme, port or path")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
        raise ValueError("Invalid macOS account name")
    root, models, home, output = (p.expanduser().resolve() for p in (root, models, home, output))
    if not output.is_relative_to(root / "runs" / "jobs"):
        raise ValueError("Generated files must stay inside ignored runs/jobs")
    output.mkdir(parents=True, exist_ok=True)
    api = {
        "Label": "kz.fortebank.api", "UserName": user, "WorkingDirectory": str(root),
        "ProgramArguments": ["/usr/bin/caffeinate", "-s", str(root / ".venv/bin/python"),
                             "-m", "uvicorn", "service.main:app",
                             "--host", "127.0.0.1", "--port", "8000", "--workers", "1"],
        "EnvironmentVariables": {
            "HOME": str(home), "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONUNBUFFERED": "1", "VISION_ENGINE": "real", "MODELS_DIR": str(models),
            "SERVICE_WORKERS": "2", "SERVICE_CAPACITY": "8", "SERVICE_TIMEOUT_SEC": "270",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        },
        "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 10, "ExitTimeOut": 45,
        "StandardOutPath": str(output / "api.log"), "StandardErrorPath": str(output / "api-error.log"),
    }
    caddy = {
        "Label": "kz.fortebank.https",
        "ProgramArguments": ["/opt/homebrew/bin/caddy", "run", "--config",
                             "/Library/Application Support/ForteBank/Caddyfile", "--adapter", "caddyfile"],
        "EnvironmentVariables": {"HOME": "/var/db/fortebank-caddy"},
        "RunAtLoad": True, "KeepAlive": True, "ThrottleInterval": 10,
        "StandardOutPath": "/var/log/fortebank-https.log", "StandardErrorPath": "/var/log/fortebank-https-error.log",
    }
    for name, value in [("kz.fortebank.api.plist", api), ("kz.fortebank.https.plist", caddy)]:
        (output / name).write_bytes(plistlib.dumps(value))
    (output / "Caddyfile").write_text(
        "{\n    admin off\n}\n\n" + domain + " {\n    reverse_proxy 127.0.0.1:8000\n}\n", encoding="utf-8",
    )


def main() -> None:
    load_dotenv(ROOT / ".env.local", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--models", type=Path, default=Path(os.getenv("MODELS_DIR", ROOT.parent / "models")))
    args = parser.parse_args()
    output = ROOT / "runs/jobs/deploy"
    render(args.domain, ROOT, args.models, getpass.getuser(), Path.home(), output)
    print(f"Prepared files in {output}. Review and install using service/deploy/README.md; nothing was started.")


if __name__ == "__main__":
    main()
