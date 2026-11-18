#!/usr/bin/env python3
"""Report figure of the mesh topologies: bilder/verification_mesh_topologies.pdf.

One panel per approach around x = 1 m, h+ 548 and L_nom/delta 0.18, read from the
polyMesh of the archived SSG/LRR runs.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "turbulentFlatPlate" / "_postProcessing"))
import _pp_common as pp  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

STUDY = REPO / "results" / "turbulentFlatPlate" / "SSGLRR"
OUTPUT = REPO / "documentation" / "report_latex" / "bilder" / "verification_mesh_topologies.pdf"
U_INF, NU = 69.4, 1.769e-5 / 1.274486
N_PRISMS, L_NOM = 40, 0.0025
X_RANGE, Y_RANGE = (1.0, 1.024), (0.0, 0.009)
PANELS = ("blockMesh", "snappy", "snappy45", "gmshTri", "gmshHex")


def binary_list(data, count_width, start=0):
    match = re.compile(rb"(\d+)\s*\(").search(data, start)
    count = int(match.group(1))
    return count, match.end(), match.end() + count * count_width


def read_points(mesh):
    data = (mesh / "points").read_bytes()
    count, begin, _ = binary_list(data, 24, data.index(b"}"))
    return np.frombuffer(data, "<f8", count * 3, begin).reshape(-1, 3)


def read_faces(mesh):
    data = (mesh / "faces").read_bytes()
    count, begin, end = binary_list(data, 4, data.index(b"}"))
    offsets = np.frombuffer(data, "<i4", count, begin)
    count, begin, _ = binary_list(data, 4, end)
    return offsets, np.frombuffer(data, "<i4", count, begin)


def empty_patch(mesh):
    text = (mesh / "boundary").read_text(errors="replace")
    block = re.search(r"frontAndBack\s*\{(.*?)\}", text, re.S).group(1)
    return (int(re.search(r"startFace\s+(\d+)", block).group(1)),
            int(re.search(r"nFaces\s+(\d+)", block).group(1)))


def cell_polygons(mesh):
    points = read_points(mesh)
    offsets, labels = read_faces(mesh)
    start, size = empty_patch(mesh)
    z_mid = 0.5 * (points[:, 2].min() + points[:, 2].max())
    polygons = []
    for face in range(start, start + size):
        xyz = points[labels[offsets[face]:offsets[face + 1]]]
        if xyz[:, 2].max() > z_mid:
            continue
        if (xyz[:, 0].max() < X_RANGE[0] or xyz[:, 0].min() > X_RANGE[1]
                or xyz[:, 1].min() > Y_RANGE[1]):
            continue
        polygons.append(xyz[:, :2])
    return polygons


def select_run(combos, approach):
    for combo in combos:
        if combo["approach"] != approach:
            continue
        params = combo["params"]
        if approach == "blockMesh":
            return combo
        size = params.get("wallCellSize", 0.02 / 2 ** params.get("wallLevel", 0))
        if params["nPrisms"] == N_PRISMS and abs(size - L_NOM) < 1e-9:
            return combo
    raise LookupError(approach)


def main():
    pp.report_style()
    scales = pp.flow_scales(U_INF, NU)
    delta = scales["delta_mean"]
    height = 1e-6 * (1.17 ** N_PRISMS - 1) / 0.17
    combos = json.loads((STUDY / "study_manifest.json").read_text())["combos"]

    fig = plt.figure(figsize=(pp.REPORT_WIDTH_IN, 5.0))
    grid = fig.add_gridspec(3, 4)
    slots = (grid[0, 1:3], grid[1, :2], grid[1, 2:], grid[2, :2], grid[2, 2:])
    for slot, approach in zip(slots, PANELS):
        ax = fig.add_subplot(slot)
        run = STUDY / "_rundata" / select_run(combos, approach)["run_dir"]
        polygons = [p / delta for p in cell_polygons(run / "constant" / "polyMesh")]
        ax.add_collection(PolyCollection(polygons, facecolors="none", edgecolors=pp.INK,
                                         linewidths=0.25))
        if approach != "blockMesh":
            ax.axhline(height / delta, **pp.STACK_TOP_STYLE, zorder=4)
        ax.set_xlim(X_RANGE[0] / delta, X_RANGE[1] / delta)
        ax.set_ylim(Y_RANGE[0] / delta, Y_RANGE[1] / delta)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(approach)
        ax.set_xlabel(pp.X_PLATE_LABEL)
        ax.set_ylabel(r"$y/\bar\delta$")
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    rect = pp.legend_below(fig, [pp.line_handle(pp.STACK_TOP_LABEL, color=pp.MUTED,
                                                linestyle=":")])
    fig.tight_layout(rect=rect)
    fig.savefig(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
