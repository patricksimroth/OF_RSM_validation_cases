#!/usr/bin/env python3
"""
cv_balance.py - integral mass and momentum balance of the control volume
around the prism stack: which fluxes cross the prism/core interface, and which
of them differ from the blockMesh reference.

Control volume (CV): the cells below the stack top y = h between two stations
x_a < x_b on the plate.  Its faces are the wall, the top (the prism/core
interface) and the two sides.  For a steady flow the x-momentum fluxes through
them add up to zero, which splits the wall shear force into the fluxes that
feed it (all divided by q_inf times the wall area of the CV, so every term is
a share of the mean skin-friction coefficient of the CV):

    C_f = shear_top + conv_top + conv_side + pressure  (+ closure)

    shear_top  viscous + Reynolds shear stress acting on the top face
    conv_top   momentum deficit rho (U_inf - u) carried out through the top
    conv_side  deficit flux leaving through the downstream side minus the
               one entering upstream: growth of the deficit inside the stack
    pressure   pressure and normal stresses on the two sides
    closure    sum of all fluxes, zero for a converged solution (in the table)

The fluxes are the face fluxes of the solver's own discretisation, summed by
the controlVolumeBalance function object (src/controlVolumeBalance) per wall
cell column, with the momentum taken relative to U_inf (the mass balance
times U_inf subtracted).  The function object runs as `rhoSimpleFoam
-postProcess` in a throw-away shadow case (profile_budget.shadow_case); its
output is cached in <run>/postProcessing/controlVolumeBalance/ and reused
while its .stamp is newer than the latest time and the mesh and its key
matches.

The blockMesh reference has no stack: there the same balance is taken for the
cells below y = h of every stack height of the study, the cell row cut by h
counted with its fraction below h (linear interpolation between two rows).

Into results/<case>/<study>/summary/ (<name>_<model> when a study has several
models):

    tables/cv_balance.csv        one row per run: plate means (0.1-1.95 m) of
                                 every term, of the reference and the closure
    cv_balance/balance.png       blockMesh: share of each flux in the local
                                 C_f along the plate, one panel per h+
    cv_balance/plate_mean.png    deviation of each flux from blockMesh, plate
                                 mean, in % of C_f: stacked per run, the
                                 marker is their sum = the C_f error
    cv_balance/error_<term>.png  the same deviation along the plate, rows h+,
                                 columns approach, lines core size
                                 (term = cf, shear_top, conv_top, conv_side,
                                 pressure)
    cv_balance/mass_top.png      mass flux out of the stack top, as a
                                 wall-normal velocity v_top / U_inf
    cv_balance/numerical_top.png numerical flux of the div(phi,U) scheme at
                                 the top: scheme minus linear face value, in %
                                 of the blockMesh C_f
"""
from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
import numpy as np

import _pp_common as pp
import profile_budget as pb
import profile_validation as pv

pp.report_style()

FUNCTION_OBJECT = "controlVolumeBalance"
CACHE_DIR = FUNCTION_OBJECT
OUT_DIR = "cv_balance"
WALL_PATCH = "bottomWall"
# Streamwise bins of the function object: at most about one wall cell column
# each, so the terms can be re-binned to any x range afterwards.
X_RANGE = (0.0, 2.0)
BIN_SIZE = 0.001
# Along-plate resolution of the figures and the range of the plate means
# (leading edge singular, last faces influenced by the outlet BC).
X_STEP = 0.05
PLATE = (0.1, 1.95)

VECTORS = ("convection", "convectionLinear", "pressure", "viscous", "reynolds")
MOMENTUM = ("convection", "pressure", "viscous", "reynolds")

# (key, legend label, colour, linestyle) of the fluxes that balance C_f
TERMS = (
    ("shear_top", "shear stress, top", "#ff7f0e", "-"),
    ("conv_top", "convection, top", "#0b0b0b", (0, (7, 2))),
    ("conv_side", "convection, sides", "#1f77b4", "-."),
    ("pressure", "pressure, sides", "#9467bd", (0, (6, 1.5, 1, 1.5))),
)
SPLIT = (
    ("tau_visc", "viscous stress, top", "#2ca02c", "-."),
    ("tau_reyn", "Reynolds stress, top", "#ff7f0e", ":"),
)
ERROR_TERMS = ("cf",) + tuple(t[0] for t in TERMS)
SYMBOL = {"cf": r"C_f", "shear_top": r"\tau,\mathrm{top}", "conv_top": r"c,\mathrm{top}",
          "conv_side": r"c,\mathrm{side}", "pressure": r"p"}
CSV_TERMS = ("cf", "shear_top", "tau_visc", "tau_reyn", "conv_top", "conv_side", "pressure",
             "closure", "v_top", "num_top")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def balance_dict(lib, heights, interpolate, u_inf, p_inf):
    edges = " ".join(f"{x:.9g}" for x in np.arange(
        X_RANGE[0], X_RANGE[1] + 0.5 * BIN_SIZE, BIN_SIZE))
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object balanceDict; }}
functions
{{
    {FUNCTION_OBJECT}
    {{
        type        {FUNCTION_OBJECT};
        libs        ("{lib}");
        heights     ({" ".join(f"{h:.12g}" for h in heights)});
        edges       ({edges});
        interpolate {"true" if interpolate else "false"};
        URef        ({u_inf:.12g} 0 0);
        pRef        {p_inf:.12g};
    }}
}}
"""


def cache_key(src_dir, text):
    fo_dir = Path(src_dir) / FUNCTION_OBJECT
    source = "".join((fo_dir / f"{FUNCTION_OBJECT}.{ext}").read_text() for ext in ("C", "H"))
    return hashlib.sha1((source + text + pp.source_key(pb.shadow_case)).encode()).hexdigest()


def extract(run_dir, latest, src_dir, heights, interpolate):
    """balance.dat of one run in its cache dir, or None."""
    run_dir = Path(run_dir)
    u_inf, _nu, _rho = pp.detect_case_parameters(run_dir)
    p_inf = pp._read_internal_field(run_dir / "0.orig" / "p") or 0.0
    cache = run_dir / "postProcessing" / CACHE_DIR
    out, stamp = cache / "balance.dat", cache / ".stamp"
    key = cache_key(src_dir, balance_dict("", heights, interpolate, u_inf, p_inf))
    if pp.outputs_fresh(stamp, [out], pp.run_data_inputs(run_dir, latest), key):
        return out
    lib = Path(src_dir) / FUNCTION_OBJECT / "lib" / f"lib{FUNCTION_OBJECT}.so"
    if not lib.is_file():
        print(f"  [warn] {lib} missing - build it with `wmake libso` in its src dir")
        return None
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="cvbalance_") as shadow:
        pb.shadow_case(run_dir, latest, shadow, src_dir)
        dict_path = Path(shadow) / "system" / "balanceDict"
        dict_path.write_text(balance_dict(lib, heights, interpolate, u_inf, p_inf))
        result = subprocess.run(pb.foam_command(shadow, dict_path), capture_output=True,
                                text=True)
        (cache / "log.balance").write_text(result.stdout + result.stderr)
        written = sorted((Path(shadow) / "postProcessing" / FUNCTION_OBJECT).glob("*/balance.dat"))
        if written:
            shutil.copy(written[-1], out)
    if result.returncode != 0 or not out.is_file():
        print(f"  [warn] {run_dir.name}: balance extraction failed (see {cache / 'log.balance'})")
        return None
    pp.write_stamp(stamp, key)
    return out


# ---------------------------------------------------------------------------
# Balance of one control-volume height
# ---------------------------------------------------------------------------
def read_balance(path):
    """{height: {group: {column: {bin: value}}}} of a balance.dat."""
    columns, out = None, {}
    with open(path) as stream:
        for line in stream:
            parts = line.split()
            if line.startswith("#"):
                if "group" in parts:
                    columns = parts[1:]
                continue
            if not parts or columns is None or len(parts) != len(columns):
                continue
            row = dict(zip(columns, parts))
            group = out.setdefault(float(row["height"]), {}).setdefault(row["group"], {})
            for name in columns[5:]:
                group.setdefault(name, {})[int(row["bin"])] = float(row[name])
    return out


def assemble(groups):
    """One height as functions of x on the wall cell columns: the edges e
    (n+1), per-column sums of the wall and top faces and the fluxes through
    the sides at the edges, all as a gain of the control volume [N, kg/s]."""
    wall = groups.get(WALL_PATCH)
    if not wall or "top" not in groups:
        return None
    bins = sorted(wall["area"])

    def column(group, name):
        values = groups.get(group, {}).get(name, {})
        return np.array([values.get(b, 0.0) for b in bins])

    # Sides: the faces to the neighbouring columns, at the plate ends a patch.
    up = {n: column("upstream", n) for n in ("area", "xMean", "massIn")}
    down = {n: column("downstream", n) for n in ("area", "xMean", "massIn")}
    for side in (up, down):
        side["xMean"] = side["xMean"] * side["area"]
    for name in VECTORS:
        up[name], down[name] = column("upstream", f"{name}_s"), column("downstream", f"{name}_s")
    for group in groups:
        if group in ("top", "upstream", "downstream", WALL_PATCH):
            continue
        side = down if np.sum(column(group, "Ss")) > 0 else up
        for name in ("area", "massIn"):
            side[name] = side[name] + column(group, name)
        side["xMean"] = side["xMean"] + column(group, "xMean") * column(group, "area")
        for name in VECTORS:
            side[name] = side[name] + column(group, f"{name}_s")
    if np.any(up["area"] <= 0) or np.any(down["area"] <= 0):
        return None
    edges = np.r_[up["xMean"] / up["area"], down["xMean"][-1] / down["area"][-1]]
    if np.any(np.diff(edges) <= 0):
        return None

    data = {"edges": edges, "wall_area": column(WALL_PATCH, "area"),
            "y_top": float(np.sum(column("top", "yMean") * column("top", "area"))
                           / np.sum(column("top", "area")))}
    for name in VECTORS:
        data[f"wall_{name}"] = column(WALL_PATCH, f"{name}_s")
        data[f"top_{name}"] = column("top", f"{name}_s")
        # flux in +x through the side at each edge: gain upstream, loss downstream
        data[f"side_{name}"] = np.r_[up[name], -down[name][-1]]
    data["top_mass"] = column("top", "massIn")
    data["side_mass"] = np.r_[up["massIn"], -down["massIn"][-1]]
    return data


def interval_terms(data, x, q_inf, rho_u):
    """Terms of the control volumes between consecutive stations x (m+1), in
    units of the mean C_f of each: {term: (m,)}."""
    edges = data["edges"]
    depth = float(np.sum(data["wall_area"]) / (edges[-1] - edges[0]))
    area = depth * np.diff(x)

    def surface(name):
        return np.diff(np.interp(x, edges, np.r_[0.0, np.cumsum(data[name])]))

    def sides(name):
        return -np.diff(np.interp(x, edges, data[f"side_{name}"]))

    wall = sum(surface(f"wall_{n}") for n in MOMENTUM)
    terms = {
        "cf": -wall,
        "tau_visc": surface("top_viscous"),
        "tau_reyn": surface("top_reynolds"),
        "conv_top": surface("top_convection"),
        "conv_side": sides("convection"),
        "pressure": surface("top_pressure") + sum(sides(n) for n in ("pressure", "viscous",
                                                                     "reynolds")),
        "num_top": surface("top_convection") - surface("top_convectionLinear"),
    }
    terms["shear_top"] = terms["tau_visc"] + terms["tau_reyn"]
    terms["closure"] = sum(terms[t] for t, *_ in TERMS) - terms["cf"]
    out = {name: value / (q_inf * area) for name, value in terms.items()}
    out["v_top"] = -surface("top_mass") / (rho_u * area)
    return out


def deviation(terms, reference, name):
    """(term - reference) in % of the reference C_f."""
    return 100.0 * (terms[name] - reference[name]) / reference["cf"]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def row_height(group, row):
    return next(h for (r, _a), names in group["cells"].items() if r == row
                for h in [group["height"][names[0]]])


def plot_reference(group, reference, x_mid, out_path):
    """blockMesh: share of each flux in the local C_f, one panel per h+."""
    rows = [r for r in group["rows"] if r is not None]
    fig, axes = plt.subplots(1, len(rows), figsize=(pp.REPORT_WIDTH_IN, 2.5), sharey=True,
                             squeeze=False)
    inside = (x_mid >= PLATE[0]) & (x_mid <= PLATE[1])
    xi = x_mid[inside] / group["delta_mean"]
    for ax, row in zip(axes[0], rows):
        terms = reference[row_height(group, row)]
        ax.axhline(0.0, color=pp.MUTED, linewidth=0.7)
        for key, _label, color, ls in TERMS + SPLIT:
            ax.plot(xi, (terms[key] / terms["cf"])[inside], color=color, linestyle=ls)
        ax.set_title(pp.fmt_hplus(row))
        ax.set_xlabel(pp.X_PLATE_LABEL)
        ax.set_xlim(PLATE[0] / group["delta_mean"], PLATE[1] / group["delta_mean"])
    axes[0][0].set_ylabel(r"flux / $C_f$")
    axes[0][0].set_ylim(-0.1, 1.1)
    handles = [pp.line_handle(label, color, ls) for _k, label, color, ls in TERMS + SPLIT]
    fig.tight_layout(rect=pp.legend_below(fig, handles, ncol=3))
    fig.savefig(out_path)
    plt.close(fig)


def plot_grid(group, series, baseline, ylabel, out_path, zero_line=True):
    """Rows h+, columns approach, one line per core size over x/delta_mean.
    series(run) -> (x, values) or None; baseline(row) the same for blockMesh."""
    rows, cols, cells, order = group["rows"], group["cols"], group["cells"], group["order"]
    style = {lab: (pp.CORE_COLORS[i % len(pp.CORE_COLORS)],
                   pp.CORE_MARKERS[i % len(pp.CORE_MARKERS)]) for i, lab in enumerate(order)}
    scale = group["delta_mean"]
    fig, axes = pp.grid_figure(len(rows), len(cols), sharex=True, sharey="row")
    plotted, has_base = set(), False
    for i, row in enumerate(rows):
        for j, approach in enumerate(cols):
            ax = axes[i][j]
            names = sorted(cells.get((row, approach), []),
                           key=lambda n: order.index(group["label"][n]))
            for name in names:
                data = series(name)
                if data is None:
                    continue
                lab = group["label"][name]
                color, marker = style[lab]
                xi = data[0] / scale
                slot = order.index(lab)
                marks = sorted({int(np.argmin(np.abs(xi - t))) for t in np.linspace(
                    xi[0], xi[-1], 6, endpoint=False) + (slot + 1) / (len(order) + 1)
                    * (xi[-1] - xi[0]) / 6})
                ax.plot(xi, data[1], color=color, marker=marker, markersize=3.5,
                        markeredgecolor="white", markeredgewidth=0.4, linewidth=1.0,
                        markevery=marks)
                plotted.add(lab)
            base = baseline(row) if baseline and names else None
            if base is not None:
                ax.plot(base[0] / scale, base[1], **pp.BASELINE_STYLE)
                has_base = True
            if zero_line:
                ax.axhline(0.0, color=pp.MUTED, linewidth=0.7)
            ax.set_xlim(PLATE[0] / scale, PLATE[1] / scale)
            if not names:
                ax.text(0.5, 0.5, "no run", transform=ax.transAxes, ha="center", va="center",
                        color=pp.EDGE)
    if not plotted:
        plt.close(fig)
        return False
    pp.label_grid(axes, [pp.fmt_hplus(r) if r is not None else "" for r in rows], cols,
                  ylabel, pp.X_PLATE_LABEL)
    handles = [pp.line_handle(lab, style[lab][0], "-", style[lab][1]) for lab in order
               if lab in plotted]
    if has_base:
        handles.append(pp.line_handle(pp.ERROR_BASELINE, pp.INK, "--"))
    fig.tight_layout(rect=pp.legend_below(fig, handles), h_pad=0.6, w_pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_plate_mean(group, mean_dev, out_path):
    """Plate-mean deviation from blockMesh of every flux, stacked per run; the
    marker is their sum, the deviation of the mean C_f."""
    rows, cols, cells, order = group["rows"], group["cols"], group["cells"], group["order"]
    fig, axes = pp.grid_figure(len(rows), len(cols), sharex=True, sharey="row")
    drawn = False
    for i, row in enumerate(rows):
        for j, approach in enumerate(cols):
            ax = axes[i][j]
            names = {group["label"][n]: n for n in cells.get((row, approach), [])
                     if n in mean_dev}
            for slot, lab in enumerate(order):
                dev = mean_dev.get(names.get(lab))
                if dev is None:
                    continue
                base = {1: 0.0, -1: 0.0}
                for key, _label, color, _ls in TERMS:
                    sign = 1 if dev[key] >= 0 else -1
                    ax.bar(slot, dev[key], 0.7, bottom=base[sign], color=color, linewidth=0)
                    base[sign] += dev[key]
                ax.plot(slot, dev["cf"], marker="D", markersize=4, color="white",
                        markeredgecolor=pp.INK, markeredgewidth=0.9, linestyle="none", zorder=5)
                drawn = True
            ax.axhline(0.0, color=pp.MUTED, linewidth=0.7)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([lab.split("=")[-1].strip() for lab in order])
            ax.grid(False, axis="x")
    if not drawn:
        plt.close(fig)
        return False
    pp.label_grid(axes, [pp.fmt_hplus(r) if r is not None else "" for r in rows], cols,
                  r"$\Delta$flux / $C_f$ [%]", pp.DX_SYMBOL)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=color, label=label) for _k, label, color, _ls in TERMS]
    handles.append(Line2D([], [], marker="D", markersize=4, color="white", linestyle="none",
                          markeredgecolor=pp.INK, markeredgewidth=0.9,
                          label=pp.error_label("cf")))
    fig.tight_layout(rect=pp.legend_below(fig, handles, ncol=3), h_pad=0.6, w_pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def flux_label(key):
    if key == "cf":
        return pp.error_label("cf")
    return rf"$\Delta c_{{{SYMBOL[key]}}}\,/\,C_f$ [%]"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def write_table(path, records):
    keys = ["name", "model", "approach", "h", "h_plus", "dx_norm", "y_top"]
    keys += [f"{t}{s}" for s in ("", "_ref") for t in CSV_TERMS]
    keys += [f"d_{t}_pct" for t in ERROR_TERMS]
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        for record in records:
            writer.writerow([record.get(k) for k in keys])


def main():
    script_dir = Path(__file__).resolve().parent
    base_path, case_name, run_case_dir, _source, results_root = pp.resolve_case_paths(script_dir)
    print("=" * 64)
    print(f"{case_name} - control-volume balance of the prism stack")
    print("=" * 64)
    manifest = pp.load_manifest(run_case_dir)
    if manifest is None:
        print(f"[info] no study manifest in {run_case_dir} - nothing to do")
        return 0
    index = pp.combo_index(manifest)
    solved = {run.name: (run, latest) for run, latest in pp.solved_run_dirs(run_case_dir)
              if run.name in index}
    groups = [g for g in pv.layout(sorted(solved), index) if g["reference"]]
    for group in groups:
        members = [n for names in group["cells"].values() for n in names]
        group["height"] = {n: pp.stack_height(index[n]) for n in members}
        if not all(group["height"].values()):
            group["reference"] = None
    groups = [g for g in groups if g["reference"]]
    if not groups:
        print(f"[info] need a {pp.ERROR_BASELINE} run and prism-stack runs - nothing to do")
        return 0
    src_dir = Path(base_path) / "src"

    jobs = []
    for group in groups:
        heights = sorted(set(group["height"].values()))
        jobs.append((group["reference"], heights, True))
        jobs += [(n, [h], False) for n, h in sorted(group["height"].items())]

    def work(job):
        name, heights, interpolate = job
        run, latest = solved[name]
        path = extract(run, latest, src_dir, heights, interpolate)
        return name, (read_balance(path) if path else None)

    raw = dict(pp.map_threads(work, jobs, pp.pv_jobs()))

    x_edges = np.arange(X_RANGE[0], X_RANGE[1] + 0.5 * X_STEP, X_STEP)
    x_mid = 0.5 * (x_edges[1:] + x_edges[:-1])
    inside = (x_mid >= PLATE[0]) & (x_mid <= PLATE[1])
    plate = np.array(PLATE)
    out_dir = Path(results_root) / pp.SUMMARY_DIR / OUT_DIR
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    tables = Path(results_root) / pp.SUMMARY_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    records, written = [], 0
    for group in groups:
        u_inf, _nu, rho = pp.detect_case_parameters(solved[group["reference"]][0])
        q_inf, rho_u = 0.5 * rho * u_inf ** 2, rho * u_inf
        group["delta_mean"] = pp.case_scales(solved[group["reference"]][0])["delta_mean"]

        def evaluate(name, height):
            groups_h = next((g for h, g in (raw.get(name) or {}).items()
                             if np.isclose(h, height, rtol=1e-9)), None)
            data = assemble(groups_h) if groups_h else None
            if data is None:
                print(f"  [warn] {name}: no balance at h = {height:g} m")
                return None
            return (interval_terms(data, x_edges, q_inf, rho_u),
                    interval_terms(data, plate, q_inf, rho_u), data["y_top"])

        reference, ref_mean = {}, {}
        for height in sorted(set(group["height"].values())):
            result = evaluate(group["reference"], height)
            if result is not None:
                reference[height], ref_mean[height], _ = result
        local, mean_dev = {}, {}
        for name, height in sorted(group["height"].items()):
            result = evaluate(name, height)
            if result is None or height not in reference:
                continue
            local[name], mean, y_top = result
            mean_dev[name] = {t: float(deviation(mean, ref_mean[height], t)[0])
                              for t in ERROR_TERMS}
            combo = index[name]
            record = {"name": name, "model": combo.get("model"),
                      "approach": combo.get("approach"), "h": height,
                      "h_plus": pp.h_plus(combo), "dx_norm": pp.dx_norm(combo), "y_top": y_top}
            record.update({t: float(mean[t][0]) for t in CSV_TERMS})
            record.update({f"{t}_ref": float(ref_mean[height][t][0]) for t in CSV_TERMS})
            record.update({f"d_{t}_pct": mean_dev[name][t] for t in ERROR_TERMS})
            records.append(record)
        if not local:
            continue
        closure = max(abs(r["closure"] / r["cf"]) for r in records if r["name"] in local)
        ref_closure = max(abs(float(m["closure"][0] / m["cf"][0])) for m in ref_mean.values())
        print(f"  {group['model']}: {len(local)} run(s), closure / C_f <= {closure:.1e}"
              f" ({pp.ERROR_BASELINE} {ref_closure:.1e})")

        suffix = f"_{group['model']}" if len(groups) > 1 else ""
        group["rows"] = [r for r in group["rows"]
                         if r is not None and row_height(group, r) in reference]
        plot_reference(group, reference, x_mid, out_dir / f"balance{suffix}.png")
        written += 1 + plot_plate_mean(group, mean_dev, out_dir / f"plate_mean{suffix}.png")

        # (file stem, axis label, value(terms, reference terms), blockMesh line)
        figures = [(f"error_{key}", flux_label(key),
                    lambda terms, ref, key=key: deviation(terms, ref, key), False)
                   for key in ERROR_TERMS]
        figures += [
            ("mass_top", r"$10^3\,v_\mathrm{top}/U_\infty$",
             lambda terms, ref: 1e3 * terms["v_top"], True),
            ("numerical_top", r"$c_\mathrm{num,top}\,/\,C_f$ [%]",
             lambda terms, ref: 100.0 * terms["num_top"] / ref["cf"], True),
        ]
        for stem, label, value, with_baseline in figures:
            def series(name, value=value):
                if name not in local:
                    return None
                return x_mid[inside], value(local[name], reference[group["height"][name]])[inside]

            def baseline(row, value=value):
                ref = reference[row_height(group, row)]
                return x_mid[inside], value(ref, ref)[inside]

            written += plot_grid(group, series, baseline if with_baseline else None, label,
                                 out_dir / f"{stem}{suffix}.png",
                                 zero_line=stem != "mass_top")

    write_table(tables / "cv_balance.csv", records)
    print(f"\nControl-volume balance complete ({written} figure(s)) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
