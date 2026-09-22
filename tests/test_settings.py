from __future__ import annotations

import json

from filmneg.engine.settings import FilmSettings, sidecar_path


def test_default_round_trips_through_json_string():
    settings = FilmSettings()
    assert FilmSettings.from_json(settings.to_json()) == settings


def test_round_trips_through_a_file(tmp_path):
    settings = FilmSettings(film_type="bw_neg")
    path = tmp_path / "photo.filmneg.json"
    settings.to_json(path)
    assert FilmSettings.from_json(path) == settings


def test_sidecar_path_convention():
    assert sidecar_path("photo.tif") == sidecar_path("photo.tif").with_name("photo.filmneg.json")
    assert str(sidecar_path("roll/photo.tif")).endswith("photo.filmneg.json")


def test_missing_keys_fall_back_to_defaults():
    partial = json.dumps({"film_type": "slide"})
    settings = FilmSettings.from_json(partial)
    assert settings.film_type == "slide"
    assert settings.tone == FilmSettings().tone
    assert settings.invert == FilmSettings().invert


def test_unknown_keys_are_ignored():
    payload = json.dumps({"film_type": "slide", "some_future_field": 42})
    settings = FilmSettings.from_json(payload)
    assert settings.film_type == "slide"


def test_nested_partial_settings_fill_missing_subfields():
    payload = json.dumps({"invert": {"mode": "linear"}})
    settings = FilmSettings.from_json(payload)
    assert settings.invert.mode == "linear"
    assert settings.invert.dmax == FilmSettings().invert.dmax


def test_lists_coerce_back_to_tuples():
    payload = json.dumps({"base_rgb": [0.9, 0.5, 0.3], "geometry": {"crop": [0, 0, 1, 1]}})
    settings = FilmSettings.from_json(payload)
    assert settings.base_rgb == (0.9, 0.5, 0.3)
    assert isinstance(settings.base_rgb, tuple)
    assert settings.geometry.crop == (0, 0, 1, 1)
    assert isinstance(settings.geometry.crop, tuple)
