"""Enable ``python -m filmneg`` — launches the GUI.

The GUI is WIP. Until then this prints a friendly pointer instead of crashing with a traceback, so the
module always resolves.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from filmneg.gui.app import main as gui_main
    except ImportError as exc:
        print(
            "The filmneg GUI is not available yet "
            f"({exc}).\n"
            'Install the GUI extras with:  pip install -e ".[gui]"\n'
            "For headless processing, use:  filmneg-batch --help",
            file=sys.stderr,
        )
        return 1
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
