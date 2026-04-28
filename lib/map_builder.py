# lib/map_builder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import io
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from pyproj import CRS, Transformer

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PatchPolygon
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

from lib.load_params import MapInfo, Ship
from lib.paths import (
    DEPTH_GRID,
    NAVIGABILITY_MAP,
    CORNERS,
    ZONES,
    ZONE_INEQ,
    TRANSITION_INEQ,
    ADJ,
)


# ======================================================================
# Generic grid helpers
# ======================================================================

def gmrt_geotiff_url(west, south, east, north, mresolution=250, layer="topo"):
    return (
        "https://www.gmrt.org/services/GridServer?"
        f"west={west}&south={south}&east={east}&north={north}"
        f"&layer={layer}&format=geotiff&mresolution={int(mresolution)}"
    )


def grid_from_depth_df(depth_df: pd.DataFrame):
    df = depth_df.copy()

    xcol = "x" if "x" in df.columns else "x_km"
    ycol = "y" if "y" in df.columns else "y_km"

    if xcol not in df.columns or ycol not in df.columns:
        raise ValueError("Expected columns x_km,y_km or x,y in depth dataframe.")
    if "depth_m" not in df.columns:
        raise ValueError("Expected column depth_m in depth dataframe.")

    df[xcol] = pd.to_numeric(df[xcol], errors="coerce").round(6)
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce").round(6)
    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    df = df.dropna(subset=[xcol, ycol])

    df = df.groupby([ycol, xcol], as_index=False, sort=True)["depth_m"].mean()

    x = np.sort(df[xcol].unique())
    y = np.sort(df[ycol].unique())

    grid = df.pivot(index=ycol, columns=xcol, values="depth_m").reindex(index=y, columns=x)
    Z = grid.to_numpy(dtype=float)

    return x, y, Z


def polygon_area_ccw(points_xy: np.ndarray) -> float:
    xs = points_xy[:, 0]
    ys = points_xy[:, 1]
    return 0.5 * np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)


def order_polygon_points(pts):
    pts = np.asarray(pts, dtype=float)
    ctr = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0])
    return np.argsort(angles)


# ======================================================================
# Zone artifact builders
# ======================================================================

def validate_zero_based_zone_tables(corners_df: pd.DataFrame, zones_df: pd.DataFrame):
    required_c = {"corner_id", "x", "y"}
    required_z = {"zone_id", "order", "corner_id"}

    if not required_c.issubset(corners_df.columns):
        raise ValueError(f"corners_df missing columns: {required_c - set(corners_df.columns)}")
    if not required_z.issubset(zones_df.columns):
        raise ValueError(f"zones_df missing columns: {required_z - set(zones_df.columns)}")

    corners_df["corner_id"] = corners_df["corner_id"].astype(int)
    zones_df[["zone_id", "order", "corner_id"]] = zones_df[
        ["zone_id", "order", "corner_id"]
    ].astype(int)

    zone_ids = sorted(zones_df["zone_id"].unique())
    if zone_ids != list(range(len(zone_ids))):
        raise ValueError(f"zone_id must be contiguous zero-based IDs. Got {zone_ids}")

    corner_ids = sorted(corners_df["corner_id"].unique())
    if corner_ids != list(range(len(corner_ids))):
        raise ValueError(f"corner_id must be contiguous zero-based IDs. Got {corner_ids}")

    for zid, group in zones_df.groupby("zone_id"):
        orders = sorted(group["order"].astype(int).tolist())
        if orders != [0, 1, 2, 3]:
            raise ValueError(f"Zone {zid} must have orders [0,1,2,3]. Got {orders}")


def get_zone_ccw_corner_ids(zid: int, zones_df: pd.DataFrame, corners_df: pd.DataFrame):
    sub = zones_df[zones_df["zone_id"] == zid].copy()
    sub = sub.set_index("order").sort_index()

    if set(sub.index.tolist()) != {0, 1, 2, 3}:
        raise ValueError(f"Zone {zid} must have orders 0..3 exactly.")

    cids = [int(sub.loc[o, "corner_id"]) for o in [0, 1, 2, 3]]

    xy = corners_df.set_index("corner_id")[["x", "y"]]
    pts = xy.loc[cids].to_numpy(float)

    if polygon_area_ccw(pts) < 0:
        cids = cids[::-1]

    return cids


def compute_lambda_array_zero_based(
    corners_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    normalize: bool = True,
) -> np.ndarray:
    validate_zero_based_zone_tables(corners_df, zones_df)

    corner_map = corners_df.set_index("corner_id")[["x", "y"]].to_dict(orient="index")
    zone_ids = sorted(zones_df["zone_id"].unique())
    Z = len(zone_ids)

    lambda_array = np.zeros((3, 4, Z), dtype=float)

    for z in zone_ids:
        cids = get_zone_ccw_corner_ids(z, zones_df, corners_df)
        pts = np.array(
            [(float(corner_map[cid]["x"]), float(corner_map[cid]["y"])) for cid in cids],
            dtype=float,
        )

        for e in range(4):
            i = e
            j = (e + 1) % 4

            xi, yi = pts[i]
            xj, yj = pts[j]

            dx = xj - xi
            dy = yj - yi

            # Format used by optimizer:
            # a_y * y + a_x * x + c >= 0
            a_y = dx
            a_x = -dy
            c = -(a_y * yi + a_x * xi)

            if normalize:
                norm = np.hypot(a_x, a_y)
                if norm > 0:
                    a_y /= norm
                    a_x /= norm
                    c /= norm

            lambda_array[0, e, z] = a_y
            lambda_array[1, e, z] = a_x
            lambda_array[2, e, z] = c

    return lambda_array


def build_adjacency_zero_based(zones_df: pd.DataFrame) -> np.ndarray:
    zone_ids = sorted(zones_df["zone_id"].unique())
    if zone_ids != list(range(len(zone_ids))):
        raise ValueError(f"zone_id must be contiguous zero-based IDs. Got {zone_ids}")

    Z = len(zone_ids)
    adj = np.zeros((Z, Z), dtype=int)
    np.fill_diagonal(adj, 1)

    corners_per_zone = {
        int(zid): set(group["corner_id"].astype(int).tolist())
        for zid, group in zones_df.groupby("zone_id")
    }

    for i in range(Z):
        for j in range(i + 1, Z):
            if len(corners_per_zone[i] & corners_per_zone[j]) >= 2:
                adj[i, j] = 1
                adj[j, i] = 1

    return adj


def shared_edge_corner_pair(zid_i: int, zid_j: int, zones_df: pd.DataFrame):
    ci = set(zones_df[zones_df["zone_id"] == zid_i]["corner_id"].astype(int).tolist())
    cj = set(zones_df[zones_df["zone_id"] == zid_j]["corner_id"].astype(int).tolist())
    shared = sorted(ci & cj)

    if len(shared) != 2:
        return None

    return int(shared[0]), int(shared[1])


def edge_indices_adjacent_to_shared(ccw_corners, shared_pair):
    shared_set = set(shared_pair)
    e_shared = None

    for e in range(4):
        a = ccw_corners[e]
        b = ccw_corners[(e + 1) % 4]
        if {a, b} == shared_set:
            e_shared = e
            break

    if e_shared is None:
        return None

    return [(e_shared - 1) % 4, (e_shared + 1) % 4]


def build_transition_arrays_zero_based(
    lambda_array: np.ndarray,
    zones_df: pd.DataFrame,
    corners_df: pd.DataFrame,
):
    validate_zero_based_zone_tables(corners_df, zones_df)

    adj = build_adjacency_zero_based(zones_df)
    Z = adj.shape[0]

    if lambda_array.shape != (3, 4, Z):
        raise ValueError(f"lambda_array must have shape (3,4,{Z}). Got {lambda_array.shape}")

    ccw_by_zone = {
        z: get_zone_ccw_corner_ids(z, zones_df, corners_df)
        for z in range(Z)
    }

    trans_from = np.full((2, 3, Z, Z), np.nan, dtype=float)
    trans_to = np.full((2, 3, Z, Z), np.nan, dtype=float)

    for i in range(Z):
        for j in range(Z):
            if i == j or adj[i, j] != 1:
                continue

            shared = shared_edge_corner_pair(i, j, zones_df)
            if shared is None:
                continue

            edges_i = edge_indices_adjacent_to_shared(ccw_by_zone[i], shared)
            edges_j = edge_indices_adjacent_to_shared(ccw_by_zone[j], shared)

            if edges_i is None or edges_j is None:
                continue

            for k, e in enumerate(edges_i):
                trans_from[k, 0, i, j] = float(lambda_array[0, e, i])
                trans_from[k, 1, i, j] = float(lambda_array[1, e, i])
                trans_from[k, 2, i, j] = float(lambda_array[2, e, i])

            for k, e in enumerate(edges_j):
                trans_to[k, 0, i, j] = float(lambda_array[0, e, j])
                trans_to[k, 1, i, j] = float(lambda_array[1, e, j])
                trans_to[k, 2, i, j] = float(lambda_array[2, e, j])

    return trans_from, trans_to, adj


# ======================================================================
# UI data models
# ======================================================================

class Corner:
    _next_id = 0

    def __init__(self, x, y, ax, color="tab:cyan", id_override=None):
        if id_override is None:
            self.id = Corner._next_id
            Corner._next_id += 1
        else:
            self.id = int(id_override)
            Corner._next_id = max(Corner._next_id, self.id + 1)

        self.x = float(x)
        self.y = float(y)
        self.artist = ax.plot(
            [self.x], [self.y],
            "o", ms=8, mfc=color, mec="k", picker=6, zorder=5
        )[0]
        self.highlighted = False
        self.zones = set()

    def set_xy(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.artist.set_data([self.x], [self.y])

    def set_highlight(self, on):
        self.highlighted = bool(on)
        if on:
            self.artist.set_markersize(10)
            self.artist.set_mec("yellow")
            self.artist.set_linewidth(2.0)
        else:
            self.artist.set_markersize(8)
            self.artist.set_mec("k")
            self.artist.set_linewidth(1.0)

    def get_xy(self):
        return np.array([self.x, self.y], dtype=float)


class Zone:
    _next_id = 0

    def __init__(self, corners, ax, facecolor="tab:orange", alpha=0.3, id_override=None):
        if id_override is None:
            self.id = Zone._next_id
            Zone._next_id += 1
        else:
            self.id = int(id_override)
            Zone._next_id = max(Zone._next_id, self.id + 1)

        self.corners = list(corners)

        for c in self.corners:
            c.zones.add(self.id)

        xy = np.array([c.get_xy() for c in self.corners])
        self.patch = PatchPolygon(
            xy,
            closed=True,
            facecolor=facecolor,
            edgecolor="k",
            alpha=alpha,
            lw=2,
            zorder=3,
        )
        self.patch.set_picker(True)
        ax.add_patch(self.patch)

    def get_xy(self):
        return np.array([c.get_xy() for c in self.corners])

    def update_patch(self):
        self.patch.set_xy(self.get_xy())

    def edges_as_corner_pairs(self):
        out = []
        n = len(self.corners)
        for i in range(n):
            out.append((self.corners[i], self.corners[(i + 1) % n]))
        return out


def edge_exists_in_any_zone(c1, c2, zones):
    for z in zones:
        for a, b in z.edges_as_corner_pairs():
            if (a is c1 and b is c2) or (a is c2 and b is c1):
                return True
    return False


def status(ax, text):
    ax.set_title(text, fontsize=10)


# ======================================================================
# Zone editor
# ======================================================================

class ZoneEditor:
    def __init__(
        self,
        nav,
        artifact_callback,
        pixel_extent=None,
        corners_path=CORNERS,
        zones_path=ZONES,
        cmap="Greys",
        origin="lower",
    ):
        self.nav = nav
        self.pixel_extent = pixel_extent
        self.artifact_callback = artifact_callback
        self.corners_path = Path(corners_path)
        self.zones_path = Path(zones_path)

        self.corners = []
        self.zones = []

        self.mode = "idle"
        self.temp_new_corners = []
        self.selected_shared = []
        self.dragging_corner = None

        self.fig = plt.figure(figsize=(9, 7))
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[1.0, 0.12], figure=self.fig)

        self.ax = self.fig.add_subplot(gs[0])
        self.ax.imshow(
            self.nav,
            cmap=cmap,
            origin=origin,
            extent=self.pixel_extent,
            interpolation="nearest",
        )
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("X (km)")
        self.ax.set_ylabel("Y (km)")
        status(self.ax, "Click 'New Zone'. Drag corners. 'Save All' exports CSVs and artifacts.")

        controls = self.fig.add_subplot(gs[1])
        controls.axis("off")
        sub = gs[1].subgridspec(1, 8, wspace=0.02)

        ax_new = self.fig.add_subplot(sub[0, 0])
        ax_next = self.fig.add_subplot(sub[0, 1])
        ax_finish = self.fig.add_subplot(sub[0, 2])
        ax_cancel = self.fig.add_subplot(sub[0, 3])
        ax_delete = self.fig.add_subplot(sub[0, 4])
        ax_save = self.fig.add_subplot(sub[0, 5])
        ax_import = self.fig.add_subplot(sub[0, 6])
        ax_done = self.fig.add_subplot(sub[0, 7])

        self.btn_new = Button(ax_new, "New Zone")
        self.btn_next = Button(ax_next, "Next Step")
        self.btn_finish = Button(ax_finish, "Finish Zone")
        self.btn_cancel = Button(ax_cancel, "Cancel")
        self.btn_delete = Button(ax_delete, "Delete Zone")
        self.btn_save = Button(ax_save, "Save All")
        self.btn_import = Button(ax_import, "Import CSV")
        self.btn_done = Button(ax_done, "Done")

        self.btn_new.on_clicked(self.on_new_zone)
        self.btn_next.on_clicked(self.on_next_step)
        self.btn_finish.on_clicked(self.on_finish_zone)
        self.btn_cancel.on_clicked(self.on_cancel)
        self.btn_delete.on_clicked(self.on_delete_zone)
        self.btn_save.on_clicked(self.on_save)
        self.btn_import.on_clicked(self.on_import)
        self.btn_done.on_clicked(self.on_done)

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def _clear_all(self):
        for z in list(self.zones):
            try:
                z.patch.remove()
            except Exception:
                pass
        self.zones.clear()

        for c in list(self.corners):
            try:
                c.artist.remove()
            except Exception:
                pass
        self.corners.clear()

        self.temp_new_corners.clear()
        self.selected_shared.clear()
        self.dragging_corner = None
        self.mode = "idle"

        Corner._next_id = 0
        Zone._next_id = 0

        self.fig.canvas.draw_idle()

    def on_import(self, event):
        try:
            self.import_from_csv(self.corners_path, self.zones_path)
            status(self.ax, f"Imported {self.corners_path.name} and {self.zones_path.name}.")
        except Exception as e:
            status(self.ax, f"Import failed: {e}")
        self.fig.canvas.draw_idle()

    def import_from_csv(self, corners_path, zones_path):
        corners_path = Path(corners_path)
        zones_path = Path(zones_path)

        df_corners = pd.read_csv(corners_path)
        df_zones = pd.read_csv(zones_path)

        validate_zero_based_zone_tables(df_corners, df_zones)

        self._clear_all()

        id_to_corner = {}
        for _, row in df_corners.sort_values("corner_id").iterrows():
            cid = int(row["corner_id"])
            c = Corner(float(row["x"]), float(row["y"]), self.ax, id_override=cid)
            id_to_corner[cid] = c
            self.corners.append(c)

        for zid, group in df_zones.sort_values(["zone_id", "order"]).groupby("zone_id"):
            corner_ids = group.sort_values("order")["corner_id"].astype(int).tolist()
            corners = [id_to_corner[cid] for cid in corner_ids]
            z = Zone(corners, self.ax, id_override=int(zid))
            z.update_patch()
            self.zones.append(z)

        if self.corners:
            Corner._next_id = max(c.id for c in self.corners) + 1
        if self.zones:
            Zone._next_id = max(z.id for z in self.zones) + 1

    def clear_temp_state(self):
        self.temp_new_corners.clear()

    def total_current_corners(self):
        return len(self.selected_shared) + len(self.temp_new_corners)

    def nearest_corner(self, x, y, tol_px=8):
        if not self.corners:
            return None

        trans = self.ax.transData.transform
        target_px = trans((x, y))

        best = None
        best_d2 = tol_px * tol_px

        for c in self.corners:
            px = trans((c.x, c.y))
            d2 = (px[0] - target_px[0]) ** 2 + (px[1] - target_px[1]) ** 2
            if d2 <= best_d2:
                best = c
                best_d2 = d2

        return best

    def on_new_zone(self, event):
        if self.mode in ("placing_first", "select_shared", "placing_remaining"):
            status(self.ax, "Already adding a zone. Finish/Next/Cancel first.")
            self.fig.canvas.draw_idle()
            return

        self.clear_temp_state()
        for c in self.selected_shared:
            c.set_highlight(False)
        self.selected_shared.clear()

        if len(self.zones) == 0:
            self.mode = "placing_first"
            status(self.ax, "First zone: click 4 corners, then Finish Zone.")
        else:
            self.mode = "select_shared"
            status(self.ax, "Select 2-4 shared corners, then Next Step.")

        self.fig.canvas.draw_idle()

    def on_next_step(self, event):
        if self.mode != "select_shared":
            status(self.ax, "Use New Zone first.")
            self.fig.canvas.draw_idle()
            return

        if not (2 <= len(self.selected_shared) <= 4):
            status(self.ax, f"Select 2-4 shared corners. Current: {len(self.selected_shared)}.")
            self.fig.canvas.draw_idle()
            return

        if len(self.selected_shared) == 2:
            c1, c2 = self.selected_shared
            if not edge_exists_in_any_zone(c1, c2, self.zones):
                status(self.ax, "The selected 2 corners are not an existing edge.")
                self.fig.canvas.draw_idle()
                return

        if len(self.selected_shared) == 4:
            status(self.ax, "All 4 corners selected. You can Finish Zone.")
        else:
            self.mode = "placing_remaining"
            need = 4 - len(self.selected_shared)
            status(self.ax, f"Place/select {need} remaining corner(s).")

        self.fig.canvas.draw_idle()

    def on_finish_zone(self, event):
        if self.mode not in ("placing_first", "placing_remaining", "select_shared"):
            status(self.ax, "Nothing to finish.")
            self.fig.canvas.draw_idle()
            return

        current = self.selected_shared + self.temp_new_corners

        unique = []
        seen = set()
        for c in current:
            if c.id not in seen:
                unique.append(c)
                seen.add(c.id)

        if len(unique) != 4:
            status(self.ax, f"Need 4 unique corners. Current: {len(unique)}.")
            self.fig.canvas.draw_idle()
            return

        pts = [c.get_xy() for c in unique]
        order = order_polygon_points(pts)
        ordered_corners = [unique[i] for i in order]

        z = Zone(ordered_corners, self.ax)
        self.zones.append(z)
        z.update_patch()

        for c in self.selected_shared:
            c.set_highlight(False)

        self.clear_temp_state()
        self.selected_shared.clear()
        self.mode = "idle"

        status(self.ax, f"Zone {z.id} created.")
        self.fig.canvas.draw_idle()

    def on_cancel(self, event):
        for c in list(self.temp_new_corners):
            if len(c.zones) == 0:
                try:
                    c.artist.remove()
                except Exception:
                    pass
                if c in self.corners:
                    self.corners.remove(c)

        self.clear_temp_state()

        for c in self.selected_shared:
            c.set_highlight(False)
        self.selected_shared.clear()

        self.mode = "idle"
        status(self.ax, "Cancelled.")
        self.fig.canvas.draw_idle()

    def on_delete_zone(self, event):
        self.mode = "delete_zone"
        status(self.ax, "Delete mode: click a zone.")
        self.fig.canvas.draw_idle()

    def _to_dataframes(self):
        df_corners = pd.DataFrame({
            "corner_id": [c.id for c in self.corners],
            "x": [c.x for c in self.corners],
            "y": [c.y for c in self.corners],
        }).sort_values("corner_id")

        rows = []
        for z in self.zones:
            for order_idx, c in enumerate(z.corners):
                rows.append({
                    "zone_id": int(z.id),
                    "order": int(order_idx),
                    "corner_id": int(c.id),
                })

        df_zones = pd.DataFrame(rows).sort_values(["zone_id", "order"])

        return df_corners, df_zones

    def on_save(self, event):
        try:
            df_corners, df_zones = self._to_dataframes()
            validate_zero_based_zone_tables(df_corners, df_zones)

            self.corners_path.parent.mkdir(parents=True, exist_ok=True)
            self.zones_path.parent.mkdir(parents=True, exist_ok=True)

            df_corners.to_csv(self.corners_path, index=False)
            df_zones.to_csv(self.zones_path, index=False)

            self.artifact_callback(df_corners=df_corners, df_zones=df_zones)

            status(self.ax, "Saved CSVs and rebuilt all zone artifacts.")
        except Exception as e:
            status(self.ax, f"Save failed: {e}")

        self.fig.canvas.draw_idle()

    def on_done(self, event):
        status(self.ax, "Done. Close the window when ready.")
        self.fig.canvas.draw_idle()

    def on_pick(self, event):
        artist = event.artist

        for c in self.corners + self.temp_new_corners:
            if c.artist is artist:
                if self.mode == "select_shared":
                    if c in self.selected_shared:
                        self.selected_shared.remove(c)
                        c.set_highlight(False)
                    else:
                        if len(self.selected_shared) < 4:
                            self.selected_shared.append(c)
                            c.set_highlight(True)
                    status(self.ax, f"Selected corners: {[cc.id for cc in self.selected_shared]}")
                    self.fig.canvas.draw_idle()

                elif self.mode == "placing_remaining":
                    if (
                        self.total_current_corners() < 4
                        and c not in self.selected_shared
                        and c not in self.temp_new_corners
                    ):
                        self.selected_shared.append(c)
                        c.set_highlight(True)
                        status(self.ax, f"Corner {c.id} added.")
                        self.fig.canvas.draw_idle()
                    else:
                        self.dragging_corner = c
                else:
                    self.dragging_corner = c
                return

        for z in self.zones:
            if z.patch is artist:
                if self.mode == "delete_zone":
                    self._delete_zone(z)
                    status(self.ax, f"Zone {z.id} deleted.")
                else:
                    status(self.ax, f"Zone {z.id} selected.")
                self.fig.canvas.draw_idle()
                return

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        x, y = event.xdata, event.ydata

        if self.mode == "placing_first":
            if self.total_current_corners() < 4:
                existing = self.nearest_corner(x, y)
                if existing is not None:
                    if existing not in self.temp_new_corners:
                        self.temp_new_corners.append(existing)
                else:
                    c = Corner(x, y, self.ax)
                    self.corners.append(c)
                    self.temp_new_corners.append(c)
                status(self.ax, f"Corners selected: {self.total_current_corners()}/4.")
                self.fig.canvas.draw_idle()

        elif self.mode == "placing_remaining":
            if self.total_current_corners() >= 4:
                status(self.ax, "Already have 4 corners. Finish Zone.")
                self.fig.canvas.draw_idle()
                return

            existing = self.nearest_corner(x, y)
            if existing is not None and existing not in self.selected_shared and existing not in self.temp_new_corners:
                self.selected_shared.append(existing)
                existing.set_highlight(True)
            else:
                c = Corner(x, y, self.ax)
                self.corners.append(c)
                self.temp_new_corners.append(c)

            status(self.ax, f"Corners selected: {self.total_current_corners()}/4.")
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        self.dragging_corner = None

    def on_motion(self, event):
        if self.dragging_corner is None or event.inaxes != self.ax:
            return

        self.dragging_corner.set_xy(event.xdata, event.ydata)

        for z in self.zones:
            if self.dragging_corner in z.corners:
                z.update_patch()

        self.fig.canvas.draw_idle()

    def _delete_zone(self, z):
        for c in z.corners:
            c.zones.discard(z.id)

        try:
            z.patch.remove()
        except Exception:
            pass

        if z in self.zones:
            self.zones.remove(z)

        unused = [c for c in self.corners if len(c.zones) == 0]
        for c in unused:
            try:
                c.artist.remove()
            except Exception:
                pass
            if c in self.corners:
                self.corners.remove(c)


# ======================================================================
# Main builder class
# ======================================================================

@dataclass
class MapBuilder:
    map_info: MapInfo
    ship: Ship

    depth_df: Optional[pd.DataFrame] = None
    depth_grid: Optional[np.ndarray] = None
    navigability_map: Optional[np.ndarray] = None

    corners_df: Optional[pd.DataFrame] = None
    zones_df: Optional[pd.DataFrame] = None
    zone_ineq: Optional[np.ndarray] = None
    zone_adj: Optional[np.ndarray] = None
    transition_ineq_from: Optional[np.ndarray] = None
    transition_ineq_to: Optional[np.ndarray] = None

    def fetch_or_load_depth(self, force: bool = False, gmrt_layer: str = "topo") -> pd.DataFrame:
        out_path = Path(DEPTH_GRID)

        if out_path.exists() and not force:
            self.depth_df = pd.read_csv(out_path)
            _, _, self.depth_grid = grid_from_depth_df(self.depth_df)
            return self.depth_df

        ref_lat_sw = float(self.map_info.sw_lat)
        ref_lon_sw = float(self.map_info.sw_lon)

        width_km = float(self.map_info.span_km_east)
        height_km = float(self.map_info.span_km_north)
        dx_km = float(self.map_info.resolution_km)
        dy_km = float(self.map_info.resolution_km)

        local_crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={ref_lat_sw} +lon_0={ref_lon_sw} "
            f"+datum=WGS84 +units=m +no_defs"
        )
        wgs84 = CRS.from_epsg(4326)

        to_wgs84 = Transformer.from_crs(local_crs, wgs84, always_xy=True)

        dx_m = dx_km * 1000.0
        dy_m = dy_km * 1000.0
        width_m = width_km * 1000.0
        height_m = height_km * 1000.0

        nx = int(round(width_m / dx_m))
        ny = int(round(height_m / dy_m))

        if nx <= 0 or ny <= 0:
            raise ValueError("Computed depth grid has non-positive size.")

        dst_transform = from_origin(west=0.0, north=height_m, xsize=dx_m, ysize=dy_m)
        dst_shape = (ny, nx)

        corners_local_m = np.array([
            [0.0, 0.0],
            [width_m, 0.0],
            [width_m, height_m],
            [0.0, height_m],
        ])

        lons, lats = [], []
        for x_m, y_m in corners_local_m:
            lon, lat = to_wgs84.transform(x_m, y_m)
            lons.append(lon)
            lats.append(lat)

        west = min(lons) - 0.01
        east = max(lons) + 0.01
        south = min(lats) - 0.01
        north = max(lats) + 0.01

        mres = int(np.clip(dx_m, 50, 2000))
        url = gmrt_geotiff_url(west, south, east, north, mresolution=mres, layer=gmrt_layer)

        r = requests.get(url, timeout=60)
        r.raise_for_status()

        with MemoryFile(r.content) as memfile, memfile.open() as src:
            src_data = src.read(1)
            depth_local = np.empty(dst_shape, dtype=np.float32)
            depth_local.fill(np.nan)

            reproject(
                source=src_data,
                destination=depth_local,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=local_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )

        cols = np.arange(nx, dtype=float)
        rows = np.arange(ny, dtype=float)

        x_centers_m = (cols + 0.5) * dx_m
        y_centers_m = (rows + 0.5) * dy_m

        Xm, Ym = np.meshgrid(x_centers_m, y_centers_m)
        Lon, Lat = to_wgs84.transform(Xm, Ym)

        depth_sw = np.flipud(depth_local)

        df = pd.DataFrame({
            "lat": Lat.ravel(),
            "lon": Lon.ravel(),
            "x_km": (Xm.ravel() / 1000.0),
            "y_km": (Ym.ravel() / 1000.0),
            "depth_m": depth_sw.ravel(),
        }).sort_values(["y_km", "x_km"], ignore_index=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        self.depth_df = df
        _, _, self.depth_grid = grid_from_depth_df(df)

        print(f"Saved depth grid to {out_path}")
        print(f"Grid shape: ny={ny}, nx={nx}, resolution={dx_km} km")
        print(f"GMRT bbox: west={west:.4f}, south={south:.4f}, east={east:.4f}, north={north:.4f}")

        return df

    def build_or_load_navigability(self, force: bool = False) -> np.ndarray:
        out_path = Path(NAVIGABILITY_MAP)

        if out_path.exists() and not force:
            self.navigability_map = np.load(out_path)
            return self.navigability_map

        if self.depth_df is None:
            self.fetch_or_load_depth(force=False)

        x, y, Z = grid_from_depth_df(self.depth_df)

        min_depth = float(self.ship.info.min_depth)
        mask = np.isfinite(Z)

        nav = (Z <= -min_depth).astype(float)
        nav[~mask] = np.nan

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, nav)

        self.depth_grid = Z
        self.navigability_map = nav

        print(f"Saved navigability map to {out_path}")
        return nav

    def build_zone_artifacts(
        self,
        df_corners: Optional[pd.DataFrame] = None,
        df_zones: Optional[pd.DataFrame] = None,
        normalize: bool = True,
    ):
        if df_corners is None:
            df_corners = pd.read_csv(CORNERS)
        if df_zones is None:
            df_zones = pd.read_csv(ZONES)

        df_corners = df_corners.copy()
        df_zones = df_zones.copy()

        validate_zero_based_zone_tables(df_corners, df_zones)

        lambda_array = compute_lambda_array_zero_based(df_corners, df_zones, normalize=normalize)
        adj = build_adjacency_zero_based(df_zones)
        trans_from, trans_to, adj_from_trans = build_transition_arrays_zero_based(
            lambda_array,
            df_zones,
            df_corners,
        )

        if not np.array_equal(adj, adj_from_trans):
            raise RuntimeError("Adjacency mismatch between adjacency builder and transition builder.")

        Path(CORNERS).parent.mkdir(parents=True, exist_ok=True)

        df_corners.to_csv(CORNERS, index=False)
        df_zones.to_csv(ZONES, index=False)
        np.savez(ZONE_INEQ, lambda_array=lambda_array)
        np.save(ADJ, adj)
        np.savez(
            TRANSITION_INEQ,
            transition_ineqs_from=trans_from,
            transition_ineqs_to=trans_to,
            adjacency=adj,
        )

        self.corners_df = df_corners
        self.zones_df = df_zones
        self.zone_ineq = lambda_array
        self.zone_adj = adj
        self.transition_ineq_from = trans_from
        self.transition_ineq_to = trans_to

        print(f"Saved corners: {CORNERS}")
        print(f"Saved zones: {ZONES}")
        print(f"Saved zone inequalities: {ZONE_INEQ}")
        print(f"Saved adjacency: {ADJ}")
        print(f"Saved transition inequalities: {TRANSITION_INEQ}")

        return lambda_array, adj, trans_from, trans_to

    def launch_zone_editor(self, force_nav: bool = False, import_existing: bool = True):
        nav = self.build_or_load_navigability(force=force_nav)

        x_extent = float(self.map_info.span_km_east)
        y_extent = float(self.map_info.span_km_north)

        extent = [0.0, x_extent, 0.0, y_extent]

        editor = ZoneEditor(
            nav=nav,
            pixel_extent=extent,
            corners_path=CORNERS,
            zones_path=ZONES,
            artifact_callback=self.build_zone_artifacts,
            origin="lower",
        )

        if import_existing and Path(CORNERS).exists() and Path(ZONES).exists():
            try:
                editor.import_from_csv(CORNERS, ZONES)
            except Exception as e:
                print(f"[WARN] Could not import existing zone CSVs: {e}")

        plt.show()
        return editor