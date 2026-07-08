import pandas as pd
from types import SimpleNamespace

from lib.map_builder import Corner, Set, SetEditor, normalize_zero_based_set_tables, validate_zero_based_set_tables


def test_normalize_zero_based_set_tables_compacts_deleted_set_gap():
    corners_df = pd.DataFrame({
        "corner_id": [0, 1, 2, 3, 4, 5, 6, 7],
        "x": [0, 1, 1, 0, 2, 2, 3, 3],
        "y": [0, 0, 1, 1, 0, 1, 1, 0],
    })
    sets_df = pd.DataFrame({
        "set_id": [0, 0, 0, 0, 2, 2, 2, 2],
        "order": [0, 1, 2, 3, 0, 1, 2, 3],
        "corner_id": [0, 1, 2, 3, 4, 5, 6, 7],
    })

    normalized_corners, normalized_sets = normalize_zero_based_set_tables(corners_df, sets_df)

    validate_zero_based_set_tables(normalized_corners, normalized_sets)
    assert normalized_sets["set_id"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_normalize_zero_based_set_tables_compacts_corner_gaps_and_drops_unused():
    corners_df = pd.DataFrame({
        "corner_id": [4, 10, 11, 12, 99],
        "x": [0, 1, 1, 0, 50],
        "y": [0, 0, 1, 1, 50],
    })
    sets_df = pd.DataFrame({
        "set_id": [7, 7, 7, 7],
        "order": [0, 1, 2, 3],
        "corner_id": [4, 10, 11, 12],
    })

    normalized_corners, normalized_sets = normalize_zero_based_set_tables(corners_df, sets_df)

    validate_zero_based_set_tables(normalized_corners, normalized_sets)
    assert normalized_corners["corner_id"].tolist() == [0, 1, 2, 3]
    assert normalized_sets["corner_id"].tolist() == [0, 1, 2, 3]


def test_set_editor_compacts_live_ids_after_delete_and_add_sequence():
    corners = [
        SimpleNamespace(id=10, sets={0}, artist=SimpleNamespace(remove=lambda: None)),
        SimpleNamespace(id=11, sets={0}, artist=SimpleNamespace(remove=lambda: None)),
        SimpleNamespace(id=20, sets={2}, artist=SimpleNamespace(remove=lambda: None)),
        SimpleNamespace(id=21, sets={2}, artist=SimpleNamespace(remove=lambda: None)),
        SimpleNamespace(id=30, sets={3}, artist=SimpleNamespace(remove=lambda: None)),
        SimpleNamespace(id=31, sets={3}, artist=SimpleNamespace(remove=lambda: None)),
    ]
    sets = [
        SimpleNamespace(id=0, corners=corners[0:2]),
        SimpleNamespace(id=2, corners=corners[2:4]),
        SimpleNamespace(id=3, corners=corners[4:6]),
    ]
    editor = SetEditor.__new__(SetEditor)
    editor.corners = corners
    editor.sets = sets

    editor._compact_ids()

    assert [z.id for z in editor.sets] == [0, 1, 2]
    assert [c.id for c in editor.corners] == [0, 1, 2, 3, 4, 5]
    assert Set._next_id == 3
    assert Corner._next_id == 6
