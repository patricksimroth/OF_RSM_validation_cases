"""Close-up comparison of a few runs of one core_refinement study and its
blockMesh reference: the along-plate errors on top, the wall-normal profiles
at every station underneath.

    .venv/bin/python3 turbulentFlatPlate/studies/core_refinement/case_sets.py [name ...]

Each entry of COMPARISONS names a results folder (study) and the manifest
values its runs must match (approach, any params key); the study's blockMesh
run is added as the reference.  An entry with "members" instead takes one
matching run from each of several studies in comparison/supporting_studies/
(the gradSchemes studies: one line per gradient scheme, faceLimited left out)
and the blockMesh run of the folder "study".  Default: every comparison.  Writes
results/turbulentFlatPlate/comparison/case_sets/<name>/:

    profiles.png        rows Ux/U_inf, nu_t/nu, k, Rxx, Ryy, Rzz, -Rxy
    budget_<eq>.png     rows = terms of <eq> (Ux, Uy, nuTilda; SSG/LRR: k, Rxx,
                        Ryy, Rzz, Rxy, omega), raw sampling, each term
                        normalised by the local norm of its run's terms
                        (profile_budget panel (a)); first row the profile
                        of the transported field: nuTilda/nu and omega (cell
                        centroids, cell_profiles.npz), k and R_ij (sampled
                        lines, as profiles.png); none for Ux, Uy

and case_sets/CAPTIONS.md.  Every figure: on top one full-width panel per
along-plate error vs blockMesh (Cf, delta99, delta*, theta, H; as
analyze.py's error_along_plate), the stations A, B, ... marked in them and
connected to their profile columns (one column per station).  Not report
figures, scaled so that the differences show: every profile panel has its own
value axis, and y/delta (delta: analytical delta99 at the station) is on a
square-root axis up to 1.1x the thickest run's delta99, so the outer layer,
where the cores differ, gets about two thirds of the height (the stack
interior below y/delta ~ 0.01 is identical in all runs).

Reads the archived _rundata/ and the caches of the regular post-processing
(boundary_layer/{delta99.csv,cell_profiles.npz}, postProcessing/turbulenceBudget/);
run_postpro.sh and
analyze.py must have run on the study.
"""
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as an  # noqa: E402
import grad_schemes as gs  # noqa: E402
import profile_budget as pb  # noqa: E402
import profile_validation as pv  # noqa: E402

pp = an.pp
RESULTS = an.ROOT / "results" / "turbulentFlatPlate"
OUT = RESULTS / "comparison" / "case_sets"
STUDIES = RESULTS / "comparison" / "supporting_studies"
GRAD_SELECT = {"approach": "gmshTri", "nPrisms": 40, "wallCellSize": 0.0025}
# Black is the blockMesh reference here, unlike in grad_schemes.py.
GAUSS_LINEAR_COLOR = "#8c564b"


def grad_members(prefix):
    return {scheme: f"{prefix}_{scheme}" for scheme in gs.PLOTTED}


COMPARISONS = (
    {"name": "SALSA_gmshTri_hplus548", "study": "SALSA",
     "select": {"approach": "gmshTri", "nPrisms": 40}},
    {"name": "SALSA_upwind_gmshTri_hplus548", "study": "SALSA_upwind",
     "select": {"approach": "gmshTri", "nPrisms": 40}},
    {"name": "SSGLRR_gmshTri_hplus548", "study": "SSGLRR",
     "select": {"approach": "gmshTri", "nPrisms": 40}},
    {"name": "SSGLRR_upwind_gmshTri_hplus548", "study": "SSGLRR_upwind",
     "select": {"approach": "gmshTri", "nPrisms": 40}},
    {"name": "gradSchemes_SALSA_gmshTri_hplus548_Lnom0p18", "study": "SALSA",
     "schemes": "base", "select": GRAD_SELECT, "members": grad_members("gradSchemes_SALSA")},
    {"name": "gradSchemes_SALSA_upwind_gmshTri_hplus548_Lnom0p18", "study": "SALSA_upwind",
     "schemes": "upwind", "select": GRAD_SELECT,
     "members": grad_members("gradSchemes_SALSA_upwind")},
)

LETTERS = "ABCDEFGHIJKL"
# Budget equations whose transported field is drawn as the first row.
# From cell_profiles.npz: stem -> (key, axis label, value(v, data), log value axis);
# the others from the sampled lines as profiles.png (profile_validation.QUANTITIES).
CELL_ROWS = {"nuTilda": ("nuTilda", r"$\tilde\nu\,/\,\nu$", lambda v, d: v / d["nu"], False),
             "omega": ("omega", r"$\omega\,/\,\omega_\infty$",
                       lambda v, d: v / d["omega_inf"], True)}
# Log value axis: range fitted above y/delta = LOG_Y_MIN (omega ~ y^-2 at the wall).
LOG_Y_MIN = 0.01
SAMPLED_ROWS = ("k", "Rxx", "Ryy", "Rzz", "Rxy")
# Wall-normal axis: y/delta on a square-root scale (outer layer, where the cores
# differ, gets ~2/3 of the height) up to past the thickest run's delta99.
Y_TICKS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5)
Y_TOP_MIN, Y_TOP_MAX = 1.1, 2.5
INK, MUTED, EDGE = pp.INK, pp.MUTED, pp.EDGE
LINK_COLOR = "#b4b3ae"
D99_STYLE = pv.D99_STYLE
VALUE_CAP = pv.VALUE_CAP
DPI = 150
RC = {**pp.REPORT_RC, "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
      "legend.fontsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
      "savefig.bbox": None}

# Layout [in]: full-width error panels, then one column per station.
LEFT, RIGHT = 1.2, 0.3
COL_W, COL_GAP = 1.75, 0.32
TOP = 0.65
ERR_H, ERR_GAP = 1.15, 0.1
LINK_H = 1.3
ROW_H, ROW_GAP = 1.75, 0.42
BOTTOM = 1.05

CAPTIONS = {
    "profiles.png":
        "Top: signed deviation of $C_f$, $\\delta_{99}$, $\\delta^*$, $\\theta$ and $H$ from the"
        " blockMesh run of the same study along the plate (0.1 m $\\le x \\le$ 1.95 m, as"
        " summary/error_along_plate/), one line per run; dotted black: $x_\\mathrm{crit}$,"
        " where the analytical $\\delta_{99}$ reaches the prism-stack height; grey verticals"
        " A, B, ...: the profile stations. Below: one column per station (title: letter and"
        " $x/\\bar\\delta$), rows $U_x/U_\\infty$, $\\nu_t/\\nu$, $k$, $\\overline{u'u'}$,"
        " $\\overline{v'v'}$, $\\overline{w'w'}$, $-\\overline{u'v'}$ over $y/\\delta$"
        " ($\\delta$ the analytical $\\delta_{99}$ at the station; square-root axis up to"
        " 1.1 times the largest $\\delta_{99}$ of the runs); black"
        " dashed: blockMesh; dotted grey: prism-stack top; dash-dot grey: $\\delta_{99}$ of"
        " the blockMesh run. Every panel has its own value axis, capped at 3x the blockMesh"
        " maximum. For eddy-viscosity models $k$ and $R_{ij}$ are the Boussinesq stresses"
        " (normal stresses isotropic by construction).",
    "budget_<eq>.png":
        "Top as profiles.png. Below: the terms of the <eq> budget (rows) at every station"
        " (columns), raw sampling (the cell value on the line $x = x_s$), each run's terms"
        " divided by the local norm of all its terms (sum of squares = 1, as panel (a) of"
        " summary/profiles/x_*/budget/<eq>.png), signed as gain (right) / loss (left),"
        " over $y/\\delta$ as profiles.png. First row (all but Ux, Uy): the profile of the"
        " transported field, $k$ and $R_{ij}$ as in profiles.png, $\\tilde\\nu/\\nu$ and"
        " $\\omega/\\omega_\\infty$ at the cell centroids (linear between them); capped as"
        " profiles.png, $\\omega$ on a logarithmic axis fitted to $y/\\delta \\ge 0.01$"
        " (it grows as $y^{-2}$ towards the wall)."
        " Every panel has its own value axis. Single-cell"
        " terms on non-aligned cores carry an O(1) zero-mean operator error"
        " (documentation/analysis_core_refinement/budget_analysis/BUDGETS.md).",
}


def matches(combo, select):
    params = combo.get("params") or {}
    return all((combo.get(k) if k in ("approach", "model") else params.get(k)) == v
               for k, v in select.items())


def order_key(run):
    approach = (pp.APPROACHES.index(run["approach"])
                if run["approach"] in pp.APPROACHES else len(pp.APPROACHES))
    return approach, run["h_plus"] or 0.0, -(run["dx_norm"] or 0.0)


def labels(runs):
    """Legend label per run: the descriptors that differ within the set."""
    def varies(key):
        return len({round(r[key], 6) if isinstance(r[key], float) else r[key]
                    for r in runs}) > 1

    keys = [k for k in ("approach", "h_plus", "dx_norm") if varies(k)] or ["dx_norm"]
    out = {}
    for r in runs:
        parts = []
        if "approach" in keys:
            parts.append(r["approach"])
        if "h_plus" in keys and r["h_plus"]:
            parts.append(pp.fmt_hplus(r["h_plus"]))
        if "dx_norm" in keys and r["dx_norm"]:
            parts.append(pp.fmt_dx(r["dx_norm"]))
        out[r["name"]] = ", ".join(parts) or r["name"]
    return out


def at_station(data, station):
    return next((v for k, v in data.items() if np.isclose(k, station)), None)


def scheme_style(scheme):
    label, color, linestyle, marker = gs.SCHEMES[scheme]
    return (GAUSS_LINEAR_COLOR if scheme == "GaussLinear" else color), marker, label, linestyle


def load_members(comparison, results):
    """blockMesh run of the study folder + one matching run per member study,
    renamed to the member key (the run dirs are the same in every study)."""
    select = comparison["select"]
    found = an.load_runs(results, select=lambda c: c["approach"] == pp.ERROR_BASELINE)
    others = []
    for scheme, folder in comparison["members"].items():
        runs = an.load_runs(STUDIES / folder, select=lambda c: matches(c, select))
        if len(runs) != 1:
            print(f"  [warn] {folder}: {len(runs)} runs match {select}, left out")
            continue
        runs[0]["name"] = scheme
        others.append(runs[0])
    return (found[0] if found else None), others, {r["name"]: scheme_style(r["name"]) for r in others}


def load_single(comparison, results):
    select = comparison["select"]
    runs = an.load_runs(results, select=lambda c: c["approach"] == pp.ERROR_BASELINE
                        or matches(c, select))
    base = next((r for r in runs if r["approach"] == pp.ERROR_BASELINE), None)
    others = sorted((r for r in runs if r is not base), key=order_key)
    text = labels(others)
    styles = {r["name"]: (pp.CORE_COLORS[i % len(pp.CORE_COLORS)],
                          pp.CORE_MARKERS[i % len(pp.CORE_MARKERS)], text[r["name"]], "-")
              for i, r in enumerate(others)}
    return base, others, styles


def load(comparison):
    results = RESULTS / comparison["study"]
    select = comparison["select"]
    loader = load_members if "members" in comparison else load_single
    base, others, styles = loader(comparison, results)
    if base is None:
        print(f"  [skip] {comparison['name']}: no completed blockMesh run")
        return None
    if not others:
        print(f"  [skip] {comparison['name']}: no completed run matches {select}")
        return None
    for r in [base] + others:
        r["profiles"] = pv.read_run_profiles(r["case"])
        cache = r["case"] / "postProcessing" / pb.CACHE_DIR
        r["budget"] = pb.read_budget(cache) if cache.is_dir() else {}
        if not r["budget"]:
            print(f"  [warn] {r['name']}: no turbulenceBudget cache, run profile_budget.py")
        cells = r["case"].parent.parent / r["case"].name / "boundary_layer" / an.pcov.CACHE_NAME
        r["cells"] = dict(np.load(cells)) if cells.exists() else None
        if r["cells"] is None:
            print(f"  [warn] {r['name']}: no {an.pcov.CACHE_NAME}, run prism_coverage.py")
    stations = [s for s in pv.STATIONS
                if at_station(base["profiles"], s) is not None]
    u_inf, nu = pp.detect_case_parameters(base["case"])[:2]
    an.RE_PER_M = u_inf / nu
    edges = [0.0]
    for r in [base] + others:
        for s in stations:
            d99 = pv.sampled_delta99(at_station(r["profiles"], s) or {})
            if d99:
                edges.append(d99 / pp.flow_scales(u_inf, nu, s)["delta_L"])
    y_top = min(Y_TOP_MAX, max(Y_TOP_MIN, 1.1 * max(edges)))
    model = pb.run_model(base["case"])
    return {"name": comparison["name"], "results": results, "base": base, "runs": others,
            "members": comparison.get("members"),
            "stations": stations, "u_inf": u_inf, "nu": nu, "styles": styles, "y_top": y_top,
            "model": model, "schemes": comparison.get("schemes") or pp.study_schemes(results),
            "omega_inf": pp._read_internal_field(base["case"] / "0.orig" / "omega"),
            "delta_mean": pp.flow_scales(u_inf, nu)["delta_mean"],
            "stacks": sorted({r["h_stack"] for r in others if r["h_stack"]})}


def marks(u, lo, hi, slot, n_slots, n):
    """Marker indices spread evenly in u over [lo, hi], staggered per line."""
    targets = lo + (hi - lo) * (np.arange(n) + (slot + 1) / (n_slots + 1)) / n
    return sorted({int(np.argmin(np.abs(u - t))) for t in targets})


def sqrt_marks(y, y_top, slot, n_slots, n=5):
    return marks(np.sqrt(np.maximum(y, 0.0)), 0.0, np.sqrt(y_top), slot, n_slots, n)


def y_axis(ax, y_top):
    ax.set_yscale("function", functions=(lambda y: np.sqrt(np.maximum(y, 0.0)), np.square))
    ax.set_ylim(0.0, y_top)
    ax.yaxis.set_major_locator(FixedLocator([t for t in Y_TICKS if t <= y_top]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def line_kw(style, markevery):
    color, marker, _label, linestyle = style
    return dict(color=color, marker=marker, linestyle=linestyle, markersize=3.5,
                markeredgecolor="white", markeredgewidth=0.4, linewidth=1.0, markevery=markevery)


class Layout:
    """Axes placed in inches: n_err full-width panels over an n_rows x n_cols grid."""

    def __init__(self, n_err, n_rows, n_cols, bottom=BOTTOM):
        self.grid_w = n_cols * COL_W + (n_cols - 1) * COL_GAP
        self.w = LEFT + self.grid_w + RIGHT
        err_h = n_err * ERR_H + (n_err - 1) * ERR_GAP
        grid_h = n_rows * ROW_H + (n_rows - 1) * ROW_GAP
        self.h = TOP + err_h + LINK_H + grid_h + bottom
        self.err_bottom = self.h - TOP - err_h
        self.grid_top = self.err_bottom - LINK_H
        self.fig = plt.figure(figsize=(self.w, self.h))
        self.err = [self.add(LEFT, self.h - TOP - (i + 1) * ERR_H - i * ERR_GAP,
                             self.grid_w, ERR_H) for i in range(n_err)]
        self.grid = [[self.add(self.col_left(j), self.grid_top - (i + 1) * ROW_H - i * ROW_GAP,
                               COL_W, ROW_H) for j in range(n_cols)] for i in range(n_rows)]

    def add(self, left, bottom, width, height):
        return self.fig.add_axes((left / self.w, bottom / self.h, width / self.w, height / self.h))

    def col_left(self, j):
        return LEFT + j * (COL_W + COL_GAP)

    def link(self, x_frac, j):
        """Elbow from x_frac of the error panels' width to the top of column j."""
        x0 = LEFT + x_frac * self.grid_w
        x1 = self.col_left(j) + 0.5 * COL_W
        y0, y1 = self.err_bottom - 0.28, self.grid_top + 0.36
        xs = np.array([x0, x0, x1, x1]) / self.w
        ys = np.array([y0, y0 - 0.12, y1 + 0.14, y1]) / self.h
        self.fig.add_artist(Line2D(xs, ys, transform=self.fig.transFigure, color=LINK_COLOR,
                                   linewidth=0.9))


def draw_errors(lay, data):
    base, runs, dm = data["base"], data["runs"], data["delta_mean"]
    xlim = (an.PLATE[0] / dm, 2.0 / dm)
    for i, (ax, (key, _symbol, error)) in enumerate(zip(lay.err, an.ERROR_QUANTITIES)):
        for slot, r in enumerate(runs):
            curve = error(r, base)
            if curve is None:
                continue
            xi = curve[0] / dm
            ax.plot(xi, curve[1], **line_kw(data["styles"][r["name"]],
                                             marks(xi, *xlim, slot, len(runs), 10)))
        ax.axhline(0.0, color=MUTED, linewidth=0.7)
        for h in data["stacks"]:
            xc = an.x_crit_an(h)
            if xc and an.PLATE[0] <= xc <= an.PLATE[1]:
                ax.axvline(xc / dm, color=INK, linestyle=":", linewidth=1.0)
        for s in data["stations"]:
            ax.axvline(s / dm, color=LINK_COLOR, linewidth=0.9, zorder=0)
        ax.set_xlim(*xlim)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylabel(pp.error_label(key))
        ax.tick_params(labelbottom=False, labeltop=(i == 0))
    top = lay.err[0]
    top.xaxis.set_label_position("top")
    top.set_xlabel(pp.X_PLATE_LABEL)
    bottom = lay.err[-1]
    for j, s in enumerate(data["stations"]):
        bottom.annotate(LETTERS[j], (s / dm, 0.0), xycoords=bottom.get_xaxis_transform(),
                        xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                        fontweight="bold")
        lay.link((s / dm - xlim[0]) / (xlim[1] - xlim[0]), j)


def value_limits(values, ref=None, cap=False):
    """Value range of one panel over the plotted heights, always including 0;
    with cap, at most VALUE_CAP x the reference maximum."""
    finite = np.concatenate([v[np.isfinite(v)] for v in values] + [np.zeros(1)])
    lo, hi = float(min(0.0, finite.min())), float(max(0.0, finite.max()))
    if cap and ref is not None and ref.size and np.nanmax(np.abs(ref)) > 0:
        bound = VALUE_CAP * float(np.nanmax(np.abs(ref)))
        lo, hi = max(lo, -bound), min(hi, bound)
    pad = 0.06 * (hi - lo) if hi > lo else 1.0
    return lo - (pad if lo < 0 else 0.0), hi + pad


def profile_panel(ax, data, station, curves, ref, cap, log=False):
    """curves: {run name: (y/delta, value)}, ref: (y/delta, value) or None;
    log: logarithmic value axis over the plotted heights (no cap)."""
    runs, styles, y_top = data["runs"], data["styles"], data["y_top"]
    delta = pp.flow_scales(data["u_inf"], data["nu"], station)["delta_L"]
    pool = []
    for slot, r in enumerate(runs):
        c = curves.get(r["name"])
        if c is None:
            continue
        ax.plot(c[1], c[0], **line_kw(styles[r["name"]],
                                       sqrt_marks(c[0], y_top, slot, len(runs))))
        pool.append(c[1][(c[0] <= y_top) & (c[0] >= (LOG_Y_MIN if log else -np.inf))])
    ref_values = None
    if ref is not None:
        ax.plot(ref[1], ref[0], **pp.BASELINE_STYLE)
        ref_values = ref[1][(ref[0] <= y_top) & (ref[0] >= (LOG_Y_MIN if log else -np.inf))]
        pool.append(ref_values)
    for h in data["stacks"]:
        ax.axhline(h / delta, **pp.STACK_TOP_STYLE)
    d99 = pv.sampled_delta99(at_station(data["base"]["profiles"], station) or {})
    if d99:
        ax.axhline(d99 / delta, color=EDGE, linestyle=D99_STYLE, linewidth=0.9)
    y_axis(ax, y_top)
    if log:
        ax.set_xscale("log")
        positive = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in pool] + [np.ones(0)])
        if positive.size:
            ax.set_xlim(0.8 * positive.min(), 1.25 * positive.max())
        pp.plain_log_axis(ax, "x", subs=(1.0,))
        return
    ax.axvline(0.0, color=INK, linewidth=0.5)
    if pool:
        ax.set_xlim(*value_limits(pool, ref_values, cap))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useMathText=True)


def label_columns(lay, data):
    for j, s in enumerate(data["stations"]):
        lay.grid[0][j].set_title(f"{LETTERS[j]}:  {pp.X_PLATE_LABEL} = "
                                 f"{s / data['delta_mean']:.0f}", fontweight="bold")
    for i, row in enumerate(lay.grid):
        for ax in row[1:]:
            ax.tick_params(labelleft=False)


def legend_columns(data):
    n = len(data["runs"]) + 4
    return n if n <= 8 else 6


def bottom_margin(data):
    rows = math.ceil((len(data["runs"]) + 4) / legend_columns(data))
    return BOTTOM + 0.28 * (rows - 1)


def legend(lay, data):
    handles = [pp.line_handle(lab, color, linestyle, marker)
               for color, marker, lab, linestyle in (data["styles"][r["name"]] for r in data["runs"])]
    handles += [pp.line_handle(pp.ERROR_BASELINE, INK, "--"),
                pp.line_handle(pp.STACK_TOP_LABEL, MUTED, ":"),
                pp.line_handle(rf"$\delta_{{99}}$, {pp.ERROR_BASELINE}", EDGE, D99_STYLE),
                pp.line_handle(an.XCRIT, INK, ":")]
    lay.fig.legend(handles=handles, loc="lower center", ncol=legend_columns(data), frameon=False,
                   bbox_to_anchor=(0.5, 0.12 / lay.h), handlelength=2.6, columnspacing=1.6)


def fig_profiles(data, path):
    quantities = pv.QUANTITIES
    stations = data["stations"]
    u_inf, nu = data["u_inf"], data["nu"]
    with plt.rc_context(RC):
        lay = Layout(len(an.ERROR_QUANTITIES), len(quantities), len(stations),
                     bottom_margin(data))
        draw_errors(lay, data)
        for i, (stem, axis_label, value_fn, *_tmr) in enumerate(quantities):
            for j, s in enumerate(stations):
                delta = pp.flow_scales(u_inf, nu, s)["delta_L"]
                curves = {r["name"]: pv.curve(at_station(r["profiles"], s), value_fn,
                                              u_inf, nu, delta) for r in data["runs"]}
                ref = pv.curve(at_station(data["base"]["profiles"], s), value_fn, u_inf, nu,
                               delta)
                profile_panel(lay.grid[i][j], data, s, curves, ref, cap=True)
            lay.grid[i][0].set_ylabel(f"{axis_label}\n" + r"$y/\delta$")
        label_columns(lay, data)
        legend(lay, data)
        lay.fig.savefig(path, dpi=DPI)
        plt.close(lay.fig)


def equation_specs(data):
    sources = {eq for r in [data["base"]] + data["runs"] for eq in pb.run_equations(r["case"])}
    return [spec for spec in pb.PLOT_EQUATIONS if spec[2] in sources]


def equation_rows(data, spec):
    """Terms of one equation drawn for any run at any station, in TERMS order."""
    present = set()
    for r in [data["base"]] + data["runs"]:
        for s in data["stations"]:
            prof = at_station(r["budget"], s)
            if prof is not None:
                present.update(pb.shown_terms(pb.equation_terms(prof, spec)))
    return [(t, lab) for t, lab, *_ in pb.TERMS if t in present]


def cell_curve(run, row, station, data, delta):
    """(y/delta, value) of one cell_profiles.npz field at the station, or None."""
    key, _label, value, _log = row
    field = run.get("cells")
    if field is None or key not in field:
        return None
    i = int(np.argmin(np.abs(field["x"] - station)))
    if not np.isclose(field["x"][i], station):
        return None
    o = field["offsets"]
    y, v = field["y"][o[i]:o[i + 1]], field[key][o[i]:o[i + 1]]
    keep = (y > pv.WALL_POINT_Y) & np.isfinite(v)
    return y[keep] / delta, value(v[keep], data)


def field_row(stem, data):
    """(axis label, curve(run, station, delta), log) of the profile row, or None."""
    u_inf, nu = data["u_inf"], data["nu"]
    if stem == "omega" and not data["omega_inf"]:
        return None
    if stem in CELL_ROWS:
        row = CELL_ROWS[stem]
        return row[1], lambda r, s, delta: cell_curve(r, row, s, data, delta), row[3]
    if stem in SAMPLED_ROWS:
        _stem, label, value_fn, *_tmr = next(q for q in pv.QUANTITIES if q[0] == stem)
        return label, lambda r, s, delta: pv.curve(at_station(r["profiles"], s), value_fn,
                                                   u_inf, nu, delta), False
    return None


def fig_budget(data, spec, path):
    rows = equation_rows(data, spec)
    if not rows:
        return False
    stations = data["stations"]
    u_inf, nu = data["u_inf"], data["nu"]
    field = field_row(spec[0], data)
    field_curves = []
    if field:
        for s in stations:
            delta = pp.flow_scales(u_inf, nu, s)["delta_L"]
            field_curves.append(({r["name"]: field[1](r, s, delta) for r in data["runs"]},
                                 field[1](data["base"], s, delta)))
        if not any(ref is not None or any(c is not None for c in curves.values())
                   for curves, ref in field_curves):
            field = None
    n_field = 1 if field else 0
    with plt.rc_context(RC):
        lay = Layout(len(an.ERROR_QUANTITIES), n_field + len(rows), len(stations),
                     bottom_margin(data))
        draw_errors(lay, data)
        if field:
            for j, (s, (curves, ref)) in enumerate(zip(stations, field_curves)):
                profile_panel(lay.grid[0][j], data, s, curves, ref, cap=True, log=field[2])
            lay.grid[0][0].set_ylabel(f"{field[0]}\n" + r"$y/\delta$")
        for i, (term, term_label) in enumerate(rows, start=n_field):
            for j, s in enumerate(stations):
                scales = pp.flow_scales(u_inf, nu, s)

                def curve(run):
                    c = pb.term_curve(at_station(run["budget"], s), spec, term, "ydelta",
                                      nu, scales)
                    return None if c is None else (c[1], c[0])

                curves = {r["name"]: curve(r) for r in data["runs"]}
                profile_panel(lay.grid[i][j], data, s, curves, curve(data["base"]),
                              cap=False)
            lay.grid[i][0].set_ylabel(f"{spec[1]}: {term_label}\n" + r"$y/\delta$")
        for ax in lay.grid[-1]:
            pb.gain_loss(ax)
        label_columns(lay, data)
        legend(lay, data)
        lay.fig.savefig(path, dpi=DPI)
        plt.close(lay.fig)
    return True


def comparison_line(data):
    model = an.MODEL_LABELS.get(data["model"], data["model"])
    runs = ", ".join(data["styles"][r["name"]][2] for r in data["runs"])
    first = data["runs"][0]
    stack = pp.fmt_hplus(first["h_plus"]) if first["h_plus"] else "no stack"
    if data["members"]:
        folder = next(iter(data["members"].values())).rsplit("_", 1)[0]
        return (f"- `{data['name']}/`: {model}, {data['schemes']} schemes, one run from each study"
                f" `{folder}_<scheme>` in supporting_studies (only gradSchemes differs, faceLimited"
                f" left out); {first['approach']}, {stack}, {pp.fmt_dx(first['dx_norm'])};"
                f" lines: {runs}; reference {data['base']['name']} of `{data['results'].name}`.")
    return (f"- `{data['name']}/`: {model}, {data['schemes']} schemes, study folder"
            f" `{data['results'].name}`; {first['approach']}, {stack}; {runs};"
            f" reference {data['base']['name']}.")


def write_captions(drawn):
    path = OUT / "CAPTIONS.md"
    entries = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            found = re.match(r"- `([^/`]+)/`", line)
            if found:
                entries[found.group(1)] = line
    entries.update({data["name"]: comparison_line(data) for data in drawn})
    lines = ["# Case-set comparisons",
             "",
             "Caption drafts for case_sets.py (not report figures; the figures carry no titles).",
             "Normalisation as everywhere: $x/\\bar\\delta$ and $L_\\mathrm{nom}/\\bar\\delta$ with"
             " $\\bar\\delta$ the analytical $\\delta_{99}$ averaged over the plate, $h^+$ with the"
             " analytical $u_\\tau$ at x = 2 m.",
             ""]
    for name, text in CAPTIONS.items():
        lines += [f"## {name}", "", text, ""]
    lines += ["## Comparisons", ""]
    lines += [entries[c["name"]] for c in COMPARISONS if c["name"] in entries]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    wanted = set(sys.argv[1:])
    unknown = wanted - {c["name"] for c in COMPARISONS}
    if unknown:
        sys.exit(f"unknown comparison(s): {', '.join(sorted(unknown))}")
    OUT.mkdir(parents=True, exist_ok=True)
    drawn = []
    for comparison in COMPARISONS:
        if wanted and comparison["name"] not in wanted:
            continue
        print(f"{comparison['name']}:")
        data = load(comparison)
        if data is None:
            continue
        out = OUT / comparison["name"]
        out.mkdir(exist_ok=True)
        for stale in out.glob("*.png"):
            stale.unlink()
        fig_profiles(data, out / "profiles.png")
        for spec in equation_specs(data):
            fig_budget(data, spec, out / f"budget_{spec[0]}.png")
        print(f"  wrote {out}")
        drawn.append(data)
    if drawn:
        write_captions(drawn)


if __name__ == "__main__":
    main()
