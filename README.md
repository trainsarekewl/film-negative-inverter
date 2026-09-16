# Film Negative Inverter

Converts scanned/photographed film negatives (JPEG, RAW, TIFF) into positive
images, working in linear light with a density/log-based inversion model
(à la darktable's *negadoctor*).

## Status

Early scaffolding — see [plan.md](plan.md) for the milestone roadmap.

## Development setup

```sh
pip install -e ".[gui,dust,build,dev]"
pytest
python -m filmneg
```

## Layout

- `src/filmneg/engine/` — pure numpy image processing (no Qt imports, ever).
- `src/filmneg/gui/` — PySide6 desktop app.
- `src/filmneg/batch.py` — headless batch CLI.
- `tests/` — pytest suite, including the engine/GUI boundary check.
