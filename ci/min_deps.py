"""Write the lower bound of every dependency in pyproject.toml as an exact pin, for the hatch-test "min" environment.

Run from the repository root: ``python ci/min_deps.py``. Writes ``ci/min-deps.txt``, which is generated and not committed.
"""

import re
import tomllib
from pathlib import Path

SKIP_EXTRAS = {"dev", "doc", "test"}  # tools for development, not dependencies of donordata
# name, optional [extras], ">=" lower bound, optional further specifiers, optional environment marker
LOWER_BOUND = re.compile(r"^\s*([\w.-]+)\s*(?:\[[^\]]*\])?\s*>=\s*([^,;\s]+)[^;]*(;.*)?$")

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
requirements = list(project["dependencies"])
for extra, extra_requirements in project.get("optional-dependencies", {}).items():
    if extra not in SKIP_EXTRAS:
        requirements += extra_requirements

pins = set()
for requirement in requirements:
    if match := LOWER_BOUND.match(requirement):
        name, version, marker = match.groups()
        pins.add(f"{name}=={version}{marker or ''}")

Path("ci/min-deps.txt").write_text("\n".join(sorted(pins)) + "\n")
