"""filmneg — invert film negatives (color, B&W) from JPEG, RAW, and TIFF.

The package is split into two layers with a strict boundary:

* ``filmneg.engine`` — pure numpy image processing. Never imports Qt.
* ``filmneg.gui``    — PySide6 desktop app built on top of the engine.

``filmneg.batch`` is a headless CLI that drives the engine without the GUI.
"""

__version__ = "0.1.0"
