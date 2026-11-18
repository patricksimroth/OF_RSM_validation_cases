"""Gradient-scheme test: gmshTri, h+ 548 (nPrisms 40), SALSA, 4 core sizes, base and
first-order upwind scheme sets, eight gradSchemes.

    .venv/bin/python3 turbulentFlatPlate/studies/core_refinement/grad_schemes.py

Reads the studies gradSchemes_{SALSA,SALSA_upwind}_<scheme> in
results/turbulentFlatPlate/comparison/supporting_studies/ (only the gradSchemes block of
fvSchemes differs from studies/core_refinement resp. core_refinement_SALSA_upwind) and,
since they carry no blockMesh run, the blockMesh baseline of results/turbulentFlatPlate/SALSA
resp. SALSA_upwind.  Needs boundary_layer/delta99.csv per run (bl_thickness.py); the
cell-centroid profiles (prism_coverage cache) and cell sizes (_core_size cache) are
extracted here.  Writes results/turbulentFlatPlate/comparison/grad_schemes/:

    grad_schemes.csv                 one row per run
    cf_error_vs_core.png             Cf error at x_ref (a) and L1 over the plate (b) over L_nom
    cf_error_along_plate.png         Cf error along the plate, rows core size
    bl_error_vs_core.png             L1 error of Cf / delta99 / theta / H / nu_t over the plate over L_nom
    cost.png                         iterations to the stop criterion, time per iteration
    CAPTIONS.md                      caption drafts (CAPTIONS)

faceLimited is in the table only: none of its runs met the stop criterion.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as an  # noqa: E402

pp, pcov = an.pp, an.pcov
RESULTS = an.ROOT / "results" / "turbulentFlatPlate"
STUDIES = RESULTS / "comparison" / "supporting_studies"
OUT = RESULTS / "comparison" / "grad_schemes"
SETS = {"base": "SALSA", "upwind": "SALSA_upwind"}
SET_LABELS = {"base": "base schemes", "upwind": "upwind schemes"}
SCHEMES = {
    "GaussLinear": ("Gauss linear", an.INK, "-", "o"),
    "GaussPointLinear": ("Gauss pointLinear", "#0072B2", "-", "s"),
    "leastSquares": ("leastSquares", "#009E73", "-", "^"),
    "pointCellsLeastSquares": ("pointCellsLeastSquares", "#56B4E9", "-", "v"),
    "iterativeGauss": ("iterativeGauss", "#CC79A7", "-", "D"),
    "cellMDLimited": ("cellMDLimited", "#E69F00", "--", "P"),
    "cellLimited": ("cellLimited", "#D55E00", "--", "X"),
    "faceLimited": ("faceLimited", an.MUTED, ":", "*"),
}
PLOTTED = [s for s in SCHEMES if s != "faceLimited"]
NUT_TOP = 0.055
L1_KEYS = ("delta99", "dstar", "theta", "H", "nut")
SETUP = ("gmshTri, $h^+$ = 548, SALSA. Only the gradSchemes entry differs between the lines:"
         " Gauss linear, Gauss pointLinear, leastSquares, pointCellsLeastSquares and"
         " iterativeGauss linear 5 for every gradient; cellMDLimited and cellLimited ="
         " Gauss linear with grad(U) and grad($\\tilde\\nu$) limited by"
         " cellMDLimited / cellLimited Gauss linear 1 (grad(p) unlimited). faceLimited is"
         " omitted: none of its runs met the stop criterion ($C_D$ 26-44 % low).")
REF = ("Errors relative to the blockMesh run of the same scheme set (Gauss linear) of"
       " results/SALSA resp. SALSA_upwind; $x_\\mathrm{ref}$ = 0.97 m. The core size is the"
       " nominal wall core cell size; the measured one is 0.64 $L_\\mathrm{nom}$ (gmshTri).")
NORM = ("$x$ and the core size are divided by $\\bar\\delta$, the analytical $\\delta_{99}$"
        " averaged over the plate; $h^+$ uses the analytical $u_\\tau$ at x = 2 m.")
CAPTIONS = {
    "cf_error_vs_core.png":
        "Skin-friction error at $x_\\mathrm{ref}$ (a) and L1 error over"
        " 0.1 m $\\le x \\le$ 1.95 m (b) over the core size, one line per gradient scheme;"
        " left: base schemes, right: first-order upwind convection schemes. " + SETUP + " "
        + REF + " " + NORM,
    "cf_error_along_plate.png":
        "Skin-friction error along the plate per gradient scheme. Rows: core size;"
        " columns: scheme set. Dotted: $x_\\mathrm{crit}$, where the analytical $\\delta_{99}$"
        " reaches the prism-stack height. " + SETUP + " " + REF,
    "bl_error_vs_core.png":
        "L1 deviation $\\varepsilon_{L1} = \\int |q - q_\\mathrm{blockMesh}|\\,dx /"
        " \\int q_\\mathrm{blockMesh}\\,dx$ over 0.1 m $\\le x \\le$ 1.95 m of $q$ = $C_f$, $\\delta_{99}$,"
        " $\\theta$, $H$ and the integrated eddy viscosity"
        " $\\int_0^{55\\,\\mathrm{mm}} \\nu_t\\,dy$ over the core size; columns: scheme set."
        " $\\theta$ and $H$ are integrated up to y = 55 mm. " + SETUP + " "
        + REF.replace("; $x_\\mathrm{ref}$ = 0.97 m", ""),
    "cost.png":
        "Iterations to the stop criterion (a) and wall time per iteration relative to Gauss"
        " linear at the same core size (b); columns: scheme set. 8000 = iteration cap. Shared"
        " Romeo nodes, 4 / 4 / 8 / 16 MPI ranks from the coarsest to the finest core, so (b)"
        " carries node-to-node noise of the order of 20 %. " + SETUP,
}


def study_key(path):
    name = path.name[len("gradSchemes_"):]
    for sset, prefix in (("upwind", "SALSA_upwind_"), ("base", "SALSA_")):
        if name.startswith(prefix):
            return sset, name[len(prefix):]
    return None, None


def extract_profiles(results):
    manifest = json.loads((results / "study_manifest.json").read_text())
    names = [c["run_dir"] for c in manifest["combos"] if c["approach"] != "blockMesh"
             and pp.run_completed(results / "_rundata" / c["run_dir"])]
    pvbatch, env = pp.find_pvbatch(), pp.pvbatch_env()
    script_dir = Path(pcov.__file__).resolve().parent
    pp.map_threads(lambda n: pcov.extract_cells(
        results / "_rundata" / n, results / n / "boundary_layer" / pcov.CACHE_NAME,
        pvbatch, env, script_dir), names, pp.pv_jobs())


def nut_curve(run):
    """(x, integral of nu_t over 0 <= y <= NUT_TOP) along the plate, or None."""
    cache = Path(run["case"]).parent.parent / run["name"] / "boundary_layer" / pcov.CACHE_NAME
    if not cache.is_file():
        return None
    field = dict(np.load(cache))
    yq = np.linspace(0.0, NUT_TOP, 4000)
    values = np.full(len(field["x"]), np.nan)
    for i in range(len(field["x"])):
        y, _u, nut = pcov.station_profile(field, i)
        if len(y) >= 10:
            values[i] = np.trapezoid(np.interp(yq, y, nut), yq)
    return field["x"], values


def nut_at(curve, x):
    return float(curve[1][int(np.argmin(np.abs(curve[0] - x)))]) if curve else None


def plate_curve(run, key):
    """(x, q) of one boundary-layer quantity along the plate, or None."""
    if key == "nut":
        return run["nut_curve"]
    if key == "delta99":
        return None if run["delta"] is None else (run["delta"][:, 0], run["delta"][:, 1])
    return None if run["bl"] is None else (run["bl"]["x"], run["bl"][key])


def l1_error(curve, base_curve):
    """100 int|q - q_b| dx / int q_b dx over the plate, as analyze.metrics does for Cf."""
    if curve is None or base_curve is None:
        return None
    grid = np.linspace(*an.PLATE, 400)
    q, q_b = (np.interp(grid, x[np.isfinite(v)], v[np.isfinite(v)])
              for x, v in (curve, base_curve))
    return float(100.0 * np.trapezoid(np.abs(q - q_b), grid) / np.trapezoid(q_b, grid))


def load():
    reference = an.cfv.load_reference(str(an.ROOT / "turbulentFlatPlate" / "_validation_data"))
    tmr = float(np.interp(an.STATION_X, reference[:, 0], reference[:, 1]))
    bases = {}
    for sset, folder in SETS.items():
        base = an.load_runs(RESULTS / folder, select=lambda c: c["approach"] == "blockMesh")[0]
        base["nut_curve"] = nut_curve(base)
        base["nut_int"] = nut_at(base["nut_curve"], an.STATION_X)
        bases[sset] = base
    runs = []
    for path in sorted(STUDIES.glob("gradSchemes_*")):
        sset, scheme = study_key(path)
        if scheme not in SCHEMES or not (path / "study_manifest.json").is_file():
            continue
        extract_profiles(path)
        study_runs = an.load_runs(path, select=lambda c: c["approach"] != "blockMesh")
        an.metrics(study_runs, bases[sset], tmr)
        for r in study_runs:
            r.update(set=sset, scheme=scheme, study=path.name)
            r["nut_curve"] = nut_curve(r)
            r["nut_int"] = nut_at(r["nut_curve"], an.STATION_X)
            for key in L1_KEYS:
                r[f"{key}_err_l1"] = l1_error(plate_curve(r, key), plate_curve(bases[sset], key))
            b = bases[sset]["nut_int"]
            r["nut_err_station"] = (100.0 * (r["nut_int"] - b) / b
                                    if r["nut_int"] is not None and b else None)
            r["s_per_iter"] = r["clock_s"] / r["iters"] if r["clock_s"] and r["iters"] else None
        runs += study_runs
    for r in runs:
        ref = next((q for q in runs if q["set"] == r["set"] and q["scheme"] == "GaussLinear"
                    and q["name"] == r["name"]), None)
        r["time_ratio"] = (r["s_per_iter"] / ref["s_per_iter"]
                           if ref and r["s_per_iter"] and ref["s_per_iter"] else None)
    return runs, bases


def write_csv(runs, path):
    keys = ["set", "scheme", "study", "name", "dx_norm", "l_core_norm", "nCells", "iters",
            "converged", "clock_s", "s_per_iter", "time_ratio", "cd", "cd_err", "cf_station",
            "err_station", "err_l1", "err_rms", "err_max", "nut_err_station"]
    keys += [f"{q}_err_{m}" for q, *_ in an.ERROR_QUANTITIES[1:] for m in ("station", "rms")]
    keys += [f"{key}_err_l1" for key in L1_KEYS]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in sorted(runs, key=lambda r: (r["set"], list(SCHEMES).index(r["scheme"]),
                                             -r["dx_norm"])):
            writer.writerow([r.get(k) for k in keys])


def style(scheme, marker=True):
    _label, color, ls, mk = SCHEMES[scheme]
    s = dict(color=color, linestyle=ls, linewidth=1.1, zorder=4 if scheme == "GaussLinear" else 3)
    if marker:
        s.update(marker=mk, markersize=4, markeredgecolor="white", markeredgewidth=0.4)
    return s


def handles(marker=True):
    return [pp.line_handle(SCHEMES[s][0], SCHEMES[s][1], SCHEMES[s][2],
                           SCHEMES[s][3] if marker else None) for s in PLOTTED]


def series(runs, sset, scheme, value):
    sel = sorted((r for r in runs if r["set"] == sset and r["scheme"] == scheme
                  and value(r) is not None), key=lambda r: -r["dx_norm"])
    return [r["dx_norm"] for r in sel], [value(r) for r in sel]


def vs_core(runs, rows, path, row_height=1.9):
    """rows = [(value(run), ylabel, scale)], columns = scheme set."""
    fig, axes = pp.grid_figure(len(rows), len(SETS), row_height=row_height, sharex=True)
    for i, (value, ylabel, scale) in enumerate(rows):
        for j, sset in enumerate(SETS):
            ax = axes[i][j]
            for scheme in PLOTTED:
                xs, ys = series(runs, sset, scheme, value)
                if xs:
                    ax.plot(xs, ys, **style(scheme))
            if scale == "log":
                ax.set_yscale("log")
                pp.plain_log_axis(ax)
            elif scale == "zero":
                ax.axhline(0, color=an.MUTED, linewidth=0.7)
            an.dx_axis(ax, runs)
        axes[i][0].set_ylabel(ylabel)
    for j, sset in enumerate(SETS):
        axes[0][j].set_title(SET_LABELS[sset])
    for ax in axes[-1]:
        ax.set_xlabel(pp.DX_SYMBOL)
    rect = pp.legend_below(fig, handles(), ncol=4)
    fig.tight_layout(rect=rect, h_pad=0.6, w_pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def station_label(key):
    return pp.error_label(key).replace("}$", r"}(x_\mathrm{ref})$", 1)


def fig_cf_vs_core(runs, path):
    vs_core(runs, [(lambda r: r["err_station"], "(a)  " + station_label("cf"), "zero"),
                   (lambda r: r["err_l1"], r"(b)  $\varepsilon_{L1}$ [%]", "log")], path)


def fig_bl_vs_core(runs, path):
    rows = (("err_l1", "C_f"), ("delta99_err_l1", r"\delta_{99}"), ("theta_err_l1", r"\theta"),
            ("H_err_l1", "H"), ("nut_err_l1", r"\int\nu_t dy"))
    vs_core(runs, [((lambda r, k=key: r[k]),
                    rf"$\varepsilon_{{L1,\,{symbol}}}$ [%]", "log") for key, symbol in rows],
            path, row_height=1.5)


def fig_cost(runs, path):
    vs_core(runs, [(lambda r: r["iters"], "(a)  iterations", "linear"),
                   (lambda r: r["time_ratio"], "(b)  time per iteration\nrel. to Gauss linear",
                    "linear")], path)


def fig_along_plate(runs, bases, path):
    cores = an.CORE_ORDER
    fig, axes = pp.grid_figure(len(cores), len(SETS), row_height=1.5, sharex=True)
    xc = an.x_crit_an(an.stack(40))
    for i, d in enumerate(cores):
        for j, sset in enumerate(SETS):
            ax = axes[i][j]
            for scheme in PLOTTED:
                for r in runs:
                    if (r["set"] == sset and r["scheme"] == scheme
                            and round(r["dx_norm"], 6) == d):
                        grid, err = an.plate_error(r, bases[sset])
                        ax.plot(grid / an.DELTA_MEAN, err, **style(scheme, marker=False))
            ax.axhline(0, color=an.MUTED, linewidth=0.7)
            if xc and an.PLATE[0] <= xc <= an.PLATE[1]:
                ax.axvline(xc / an.DELTA_MEAN, color=an.INK, linestyle=":", linewidth=1.0)
            ax.set_xlim(an.PLATE[0] / an.DELTA_MEAN, an.PLATE[1] / an.DELTA_MEAN)
    pp.label_grid(axes, [pp.fmt_dx(d) for d in cores], [SET_LABELS[s] for s in SETS],
                  pp.error_label("cf"), an.XI_LABEL)
    rect = pp.legend_below(fig, handles(marker=False) + [pp.line_handle(an.XCRIT, an.INK, ":")],
                           ncol=4)
    fig.tight_layout(rect=rect, h_pad=0.5, w_pad=0.6)
    fig.savefig(path)
    plt.close(fig)


def write_captions(path):
    lines = ["# Caption drafts - comparison/grad_schemes", ""]
    for name, text in CAPTIONS.items():
        lines += [f"## {name}", "", text, ""]
    path.write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, bases = load()
    if not runs:
        print(f"[error] no gradSchemes_* studies in {STUDIES}")
        return 1
    an.CORE_ORDER[:] = an.core_sizes(runs)
    write_csv(runs, OUT / "grad_schemes.csv")
    fig_cf_vs_core(runs, OUT / "cf_error_vs_core.png")
    fig_along_plate(runs, bases, OUT / "cf_error_along_plate.png")
    fig_bl_vs_core(runs, OUT / "bl_error_vs_core.png")
    fig_cost(runs, OUT / "cost.png")
    write_captions(OUT / "CAPTIONS.md")
    print(f"wrote {OUT}  ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
