"""Generates region_routing_check.ipynb from validate_region_routing.py.

`main` already has the adapters and shared code this needs (curve_n122, table_ft,
run_finetune.py, vision/ocr.py, vision/layout.py, all committed via the merged Step 28 +
Stage C PR) -- but validate_region_routing.py itself lives on this branch until its own PR
merges, so the notebook still embeds it via `%%writefile` (same reasoning as
KAGGLE/kaggle.ipynb and KAGGLE/stage_c/'s generators). Safe to drop the embed cell once
this file is on `main`.

Regenerate after changing validate_region_routing.py:
    python KAGGLE/region_routing_check/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SCRIPT = (HERE / "validate_region_routing.py").read_text(encoding="utf-8")

REPO_URL = "https://github.com/anuronmaitro/doc-agent-1.git"


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def build_cells() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        md("""# MathScholar -- region-routing validation (curve_n122 + table_ft on table crops only)

One-off check, not a pipeline deliverable: does routing `table_ft` ONLY to detected
table-region crops (leaving `curve_n122` for the rest of the page) get the table-page
gains without the whole-page regressions v6 showed on `as_p0334`/`as_p0441`/`as_p0528`?

`main` already has everything (Step 28 + Stage C merged via PR #30) -- this just clones
main and runs `KAGGLE/region_routing_check/validate_region_routing.py`, no embedding
needed.""")
    )

    cells.append(code(f"""REPO_URL = "{REPO_URL}"
BRANCH = "main"
"""))

    cells.append(md("""## 1. Clone the repo and install pinned dependencies"""))
    cells.append(code("""import os
import subprocess

if not os.path.exists("/kaggle/working/repo"):
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, "/kaggle/working/repo"], check=True)
%cd /kaggle/working/repo
!pip install -q --no-cache-dir -r requirements.lock
import pkg_resources  # noqa: F401

print("pkg_resources OK")
"""))

    cells.append(
        md(
            """## 2. Materialize the val page images (needed for layout.detect() + crop generation)"""
        )
    )
    cells.append(code("""!ANNOT=1 bash scripts/get_data.sh
"""))

    cells.append(md("""## 3. Write the validation script

Embedded verbatim (not cloned) because this file isn't on `main` yet -- see this
notebook's own generator docstring."""))
    cells.append(code('import os\n\nos.makedirs("KAGGLE/region_routing_check", exist_ok=True)\n'))
    cells.append(
        code("%%writefile KAGGLE/region_routing_check/validate_region_routing.py\n" + SCRIPT)
    )

    cells.append(md("""## 4. Run the validation"""))
    cells.append(code("""!python KAGGLE/region_routing_check/validate_region_routing.py
"""))

    return cells


def main() -> None:
    nb = {
        "cells": build_cells(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "region_routing_check.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
