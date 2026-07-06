from types import SimpleNamespace

import pandas as pd

from lib.load_params import MapInfo
from lib.map_builder import MapBuilder


def _builder(tmp_path, *, span_km_east=100, span_km_north=100, min_depth=12):
    return MapBuilder(
        map_info=MapInfo(
            sw_lat=47.7,
            sw_lon=-67,
            span_km_east=span_km_east,
            span_km_north=span_km_north,
            resolution_km=1.0,
        ),
        ship=SimpleNamespace(info=SimpleNamespace(min_depth=min_depth)),
        map_dir=tmp_path,
    )


def test_depth_metadata_matches_current_map_params(tmp_path):
    builder = _builder(tmp_path)
    builder._write_depth_metadata("topo")

    assert builder._depth_cache_current("topo")


def test_depth_metadata_rejects_changed_map_params(tmp_path):
    builder = _builder(tmp_path)
    builder._write_depth_metadata("topo")
    changed_builder = _builder(tmp_path, span_km_east=300)

    assert not changed_builder._depth_cache_current("topo")


def test_depth_cache_accepts_legacy_map_params_csv(tmp_path):
    builder = _builder(tmp_path)
    pd.DataFrame([builder._map_params_metadata()]).to_csv(builder.map_params_path, index=False)

    assert builder._depth_cache_current("topo")


def test_navigability_metadata_rejects_changed_min_depth(tmp_path):
    builder = _builder(tmp_path, min_depth=12)
    builder._write_navigability_metadata()
    changed_builder = _builder(tmp_path, min_depth=14)

    assert not changed_builder._navigability_cache_current()
