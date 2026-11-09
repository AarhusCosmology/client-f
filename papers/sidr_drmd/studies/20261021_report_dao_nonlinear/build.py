"""Figures, tables and quoted numbers of the report on the non-linear corrections at points with dark acoustic
oscillations (SIDR + DRMD, 21 October 2026). Everything comes from data/ (see README.md for where each file came
from); nothing here runs CLASS.

    python build.py            -> figures/*.pdf, tables/*.tex, numbers.tex
    xelatex report.tex (twice)
"""
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG, TAB = HERE / "data", HERE / "figures", HERE / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": False})
C_BAD, C_PATCH, C_REF, C_LCDM, C_GREY = "#c0392b", "#2471a3", "#222222", "#7d3c98", "#b3b3b3"
ELL = np.arange(2, 2501)
NUMBERS = {}


def number(name, value):
    NUMBERS[name] = value


def corner_mask(t):
    return (t["f_idm_drmd"] > 0.2) & (t["log10_zstop"] > 4.3) & (t["delta_Neff_drmd"] < 0.05)


# ---------------------------------------------------------------- 1. where the points are
def fig_parameter_space():
    t = np.load(DATA / "training_points.npz")
    best = float(t["best"])
    ok = ~t["refused"] & t["inbox"]
    near = ok & (t["loglkl"] > best - 10)
    corner = corner_mask(t)
    sep = lambda n: f"{int(n):,}".replace(",", "\\,")
    number("NearBest", sep(near.sum()))
    number("NearBestCorner", int((near & corner).sum()))
    number("NearBestCornerPercent", f"{100 * (near & corner).sum() / near.sum():.0f}")
    number("CornerInBox", sep((ok & corner).sum()))
    number("TrainingPoints", sep(135306))
    bad = t["refused"]
    number("BadPoints", int(bad.sum()))

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    low = t["delta_Neff_drmd"] < 0.13
    a = ax[0]
    a.scatter(t["delta_Neff_drmd"][ok & low & ~near], t["f_idm_drmd"][ok & low & ~near], s=0.3, c="#dddddd", rasterized=True,
              label="training points in the new box")
    a.scatter(t["delta_Neff_drmd"][near & low], t["f_idm_drmd"][near & low], s=0.5, c="#888888", rasterized=True,
              label=r"within $\Delta\ln L = 10$ of the best")
    a.scatter(t["delta_Neff_drmd"][bad], t["f_idm_drmd"][bad], s=14, marker="x", c=C_BAD, lw=1, label="NaN spectra on Grendel")
    a.axvline(0.05, c="k", lw=0.6, ls=":")
    a.axhline(0.2, c="k", lw=0.6, ls=":")
    a.set_xlim(0, 0.13); a.set_ylim(0, 1.0)
    a.set_xlabel(r"$\Delta N_{\rm eff}$ (DRMD)"); a.set_ylabel(r"$f_{\rm idm}$")
    a.axhline(0.4, c="k", lw=0.8)
    a.text(0.128, 0.41, "second set: $f_{\\rm idm} < 0.4$", ha="right", va="bottom", fontsize=7)
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=3, markerscale=4, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    b = ax[1]
    sel = t["delta_Neff_drmd"] < 0.05
    b.scatter(t["log10_zstop"][ok & sel & ~near], t["f_idm_drmd"][ok & sel & ~near], s=0.3, c="#dddddd", rasterized=True)
    b.scatter(t["log10_zstop"][near & sel], t["f_idm_drmd"][near & sel], s=0.5, c="#888888", rasterized=True)
    b.scatter(t["log10_zstop"][bad], t["f_idm_drmd"][bad], s=14, marker="x", c=C_BAD, lw=1)
    b.axvline(4.3, c="k", lw=0.6, ls=":"); b.axhline(0.2, c="k", lw=0.6, ls=":"); b.axhline(0.4, c="k", lw=0.8)
    b.set_xlim(2, 5); b.set_ylim(0, 1.0)
    b.set_xlabel(r"$\log_{10} z_{\rm stop}$"); b.set_ylabel(r"$f_{\rm idm}$")
    b.set_title(r"$\Delta N_{\rm eff} < 0.05$", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "parameter_space.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- 2. the linear spectra
def fig_linear_pk():
    z = np.load(DATA / "pk_lin.npz")
    k = z["k"]
    labels = [str(s) for s in z["labels"]]
    ref = z["pk best fit"]
    fig, ax = plt.subplots(2, 1, figsize=(5.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    styles = {"best fit": (C_REF, 1.4, "-"), "f_idm 0.003": ("#7f8c8d", 1.0, "--"), "f_idm 0.39 (finite)": (C_PATCH, 1.0, "-")}
    reds = iter(["#c0392b", "#e67e22", "#a93226", "#d35400", "#922b21"])
    slopes = {}
    for lab in labels:
        p = z[f"pk {lab}"]
        par = z[f"par {lab}"]
        col, lw, ls = styles[lab] if lab in styles else (next(reds), 0.8, "-")
        text = lab if not lab.startswith("NaN") else f"{lab} ($f_{{\\rm idm}}$ {par[2]:.2f})"
        if lab.startswith("f_idm"):
            text = f"finite, $f_{{\\rm idm}}$ {par[2]:.3f}" if par[2] < 0.01 else f"finite, $f_{{\\rm idm}}$ {par[2]:.2f}"
        with np.errstate(all="ignore"):
            ax[0].loglog(k, p / ref, c=col, lw=lw, ls=ls, label=text)
            n = np.gradient(np.log(p), np.log(k))
        ax[1].semilogx(k, n, c=col, lw=lw, ls=ls)
        m = (k > 1) & (k < 30)
        slopes[lab] = (np.nanmin(n[m]), np.nanmax(n[m]))
    for a in ax:
        a.axvline(5, c="k", lw=0.6, ls=":")
        a.axvspan(5, 5 * 5 / np.sqrt(-np.log(0.05)), color="#f2f2f2", zorder=0)
    ax[0].text(5.2, 1.4, r"$k_{\max}$", fontsize=7)
    ax[0].set_ylabel(r"$P_{\rm lin}(k) / P_{\rm lin}^{\rm best}(k)$, $z=0$")
    ax[0].set_ylim(1e-2, 2)
    ax[0].legend(loc="lower left", fontsize=6.5, ncol=2)
    ax[1].set_ylabel(r"$d\ln P_{\rm lin}/d\ln k$")
    ax[1].set_xlabel(r"$k$ [1/Mpc]")
    ax[1].set_ylim(-9.5, 4)
    ax[1].set_xlim(0.05, 40)
    fig.tight_layout()
    fig.savefig(FIG / "linear_pk.pdf")
    plt.close(fig)
    bad = [v for lab, v in slopes.items() if lab.startswith("NaN")]
    number("SlopeBadMin", f"{min(v[0] for v in bad):.1f}")
    number("SlopeBadMax", f"{max(v[1] for v in bad):+.1f}")
    number("SlopeBestMin", f"{slopes['best fit'][0]:.1f}")
    number("SlopeBestMax", f"{slopes['best fit'][1]:.1f}")


# ---------------------------------------------------------------- 3. the end of the source table
def fig_table_end():
    probe = json.loads((DATA / "table_end.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for a, (name, title) in zip(ax, [("refused_7111", "row 7111: last step $-1.2\\times10^{-4}$"),
                                     ("refused_101819", "row 101819: last step $+2.3\\times10^{-4}$")]):
        p = probe[name]
        kt, St = np.array(p["k"]), np.array(p["S"])
        kmax, smax = kt[-1], St[-1]
        L = p["ln_ak_max"]
        kk = np.logspace(np.log10(kmax), 6, 300)
        exact = 1 + np.log(kk / kmax) / L                      # the original fit, in exact arithmetic
        a.plot(kt, St / smax, "o-", c=C_REF, ms=3, lw=0.8, label="source table (computed)")
        a.plot(kk, exact, c=C_BAD, lw=1.2, ls="--",
               label=f"original fit, exact: $\\ln(a k_{{\\max}})={L:.0f}$\n(evaluated: NaN, $a$ {'underflows' if L < 0 else 'overflows'})")
        if p["branch"] == 2:
            b = p["b_hmcode"]
            guard = np.log(np.e + b * kk) / np.log(np.e + b * kmax)
            a.plot(kk, guard, c=C_PATCH, lw=1.2, label="patch: extrap_hmcode form")
        else:
            a.plot(kk, exact, c=C_PATCH, lw=1.2, ls=":", label="patch: the same function,\nwithout the overflow")
        a.axvline(kmax * 5 / np.sqrt(-np.log(0.05)), c="k", lw=0.5, ls=":")
        a.text(kmax * 5 / np.sqrt(-np.log(0.05)) * 1.15, 0.86, "halofit's\nrange", fontsize=6.5, va="bottom")
        a.set_xscale("log")
        a.set_xlabel(r"$k$ [1/Mpc]")
        a.set_ylabel(r"$S(k)/S(k_{\max})$  ($\delta_m$, $z=0$)")
        a.set_title(title, fontsize=8)
        a.set_ylim(0.85, 4.2 if p["branch"] == 2 else 1.2)
        a.set_xlim(1, 1e6)
        a.legend(fontsize=6.3, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "table_end.pdf")
    plt.close(fig)
    number("LnAkSeven", f"{probe['refused_7111']['ln_ak_max']:.0f}")
    number("LnAkHundred", f"{probe['refused_101819']['ln_ak_max']:.0f}")


# ---------------------------------------------------------------- 4. k_max
def deviations(npz, ref_k=30.0):
    """{row: {kmax: (TT l<=2500, phiphi 8<=L<=400)}} relative to ref_k, finite points only."""
    z = np.load(npz)
    idx = {(int(r), float(k)): i for i, (r, k) in enumerate(zip(z["row"], z["kmax"]))}
    out = {}
    for (r, k), i in idx.items():
        j = idx.get((r, ref_k))
        if j is None or not (z["finite"][i] and z["finite"][j]):
            continue
        x, y = z["spectra"][i], z["spectra"][j]
        tt = np.max(np.abs(x[0] / y[0] - 1))
        m = (ELL >= 8) & (ELL <= 400)
        pp = np.max(np.abs(x[2, m] / y[2, m] - 1))
        out.setdefault(r, {})[k] = (tt, pp)
    return out


def fig_kmax():
    first = json.loads((DATA / "kmax_ladder_laptop_hmcode_refused.first.json").read_text())
    ks = [s["kmax"] for s in first["summary"]]
    finite = [s["finite"] for s in first["summary"]]
    gl = np.load(DATA / "guard_ladder_halofit.npz")
    pts = [str(p) for p in gl["points"]]
    unpatched = deviations(DATA / "kmax_ladder_laptop_halofit_refused.npz")
    t = np.load(DATA / "training_points.npz")
    fidm = dict(zip(t["row"], t["f_idm_drmd"]))
    sample = deviations(DATA / "kmax_ladder_laptop_halofit_sample40.npz")

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
    a = ax[0]
    a.bar(range(len(ks)), finite, color=C_BAD, width=0.7)
    a.set_xticks(range(len(ks)), [f"{k:g}" for k in ks])
    a.set_ylim(0, 44); a.axhline(41, c="k", lw=0.5, ls=":")
    a.set_xlabel(r"$k_{\max}$ [1/Mpc]"); a.set_ylabel("finite of 41 points")
    a.set_title("HMcode, unpatched (laptop)", fontsize=8)

    b = ax[1]
    kk = [5, 10, 20]
    for r, dev in unpatched.items():
        b.plot([k for k in sorted(dev) if k < 30], [dev[k][1] for k in sorted(dev) if k < 30], c=C_BAD, lw=0.4, alpha=0.5)
    for p in pts:
        y = [gl[f"{p}|{k}|phiphi"] for k in kk]
        if p.startswith("refused"):
            b.plot(kk, y, c=C_PATCH, lw=0.9)
    b.plot(kk, [gl[f"best_112257|{k}|phiphi"] for k in kk], c=C_REF, lw=1.6)
    b.plot(kk, [gl[f"lcdm_planck|{k}|phiphi"] for k in kk], c=C_LCDM, lw=1.6, ls="--")
    lcdm_like = [r for r in sample if fidm.get(r, 1) < 0.05]
    env = [max(sample[r][k][1] for r in lcdm_like) for k in (5.0, 10.0, 20.0)]
    b.plot(kk, env, c="#7f8c8d", lw=1.0, ls=":")
    b.set_yscale("log"); b.set_xscale("log"); b.set_xticks([5, 10, 20], ["5", "10", "20"])
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_ylim(1e-4, 1.0)
    b.set_xlabel(r"$k_{\max}$ [1/Mpc]")
    b.set_ylabel(r"max $|C_L^{\phi\phi}/C_L^{\phi\phi}(30)-1|$, $L\leq400$")
    b.plot([], [], c=C_BAD, lw=0.8, label="41 points, unpatched")
    b.plot([], [], c=C_PATCH, lw=0.9, label="8 of them, patched")
    b.plot([], [], c=C_REF, lw=1.6, label="best fit")
    b.plot([], [], c=C_LCDM, lw=1.6, ls="--", label=r"Planck $\Lambda$CDM")
    b.plot([], [], c="#7f8c8d", lw=1.0, ls=":", label=r"worst of 20 with $f_{\rm idm}<0.05$")
    b.legend(fontsize=5.8, loc="upper right")
    b.set_title("halofit: convergence in $k_{\\max}$", fontsize=8)

    c = ax[2]
    kk4 = [5, 10, 20, 30]
    for p in pts:
        y = [gl[f"{p}|{k}|zoff"] for k in kk4]
        if p.startswith("refused"):
            c.plot(kk4, y, c=C_PATCH, lw=0.9, marker="o", ms=2)
    c.plot(kk4, [gl[f"best_112257|{k}|zoff"] for k in kk4], c=C_REF, lw=1.6, marker="o", ms=2)
    c.plot(kk4, [gl[f"lcdm_planck|{k}|zoff"] for k in kk4], c=C_LCDM, lw=1.6, ls="--", marker="o", ms=2)
    c.plot([5], [0.0], marker="x", c=C_BAD, ms=6, ls="none", label="8 points, unpatched (laptop)")
    c.set_xscale("log"); c.set_xticks([5, 10, 20, 30], ["5", "10", "20", "30"])
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.set_xlabel(r"$k_{\max}$ [1/Mpc]"); c.set_ylabel(r"$z$ above which halofit is off")
    c.set_ylim(-0.2, 6.5)
    c.legend(fontsize=6, loc="upper left")
    c.set_title("halofit: where the correction stops", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "kmax.pdf")
    plt.close(fig)

    pp5 = [gl[f"{p}|5|phiphi"] for p in pts if p.startswith("refused")]
    tt5 = [gl[f"{p}|5|TT"] for p in pts if p.startswith("refused")]
    number("PatchedPPMin", f"{100 * min(pp5):.1f}"); number("PatchedPPMax", f"{100 * max(pp5):.1f}")
    number("PatchedTTMin", f"{100 * min(tt5):.2f}"); number("PatchedTTMax", f"{100 * max(tt5):.1f}")
    number("BestPP", f"{100 * gl['best_112257|5|phiphi']:.2f}"); number("BestTT", f"{100 * gl['best_112257|5|TT']:.2f}")
    number("LcdmPP", f"{100 * gl['lcdm_planck|5|phiphi']:.2f}"); number("LcdmTT", f"{100 * gl['lcdm_planck|5|TT']:.2f}")
    un5 = [dev[5.0] for dev in unpatched.values() if 5.0 in dev]
    number("UnpatchedTTMax", f"{100 * max(v[0] for v in un5):.1f}")
    number("UnpatchedPPMax", f"{100 * max(v[1] for v in un5):.0f}")


# ---------------------------------------------------------------- 5. the patch on Grendel
def fig_grendel():
    t = np.load(DATA / "training_points.npz")
    fidm = dict(zip(t["row"], t["f_idm_drmd"]))
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.6))
    counts = {}
    for module in ("halofit", "hmcode"):
        for points in ("refused", "sample40", "best"):
            a, b = np.load(DATA / f"grendel_env_{module}_{points}.npz"), np.load(DATA / f"grendel_guard_{module}_{points}.npz")
            same = changed = finite_a = finite_b = 0
            rows, dtt, dpp = [], [], []
            for i in range(len(a["row"])):
                fa, fb = bool(a["finite"][i]), bool(b["finite"][i])
                finite_a += fa; finite_b += fb
                if fa and fb:
                    if np.array_equal(a["spectra"][i], b["spectra"][i]):
                        same += 1
                    else:
                        changed += 1
                        x, y = a["spectra"][i], b["spectra"][i]
                        m = (ELL >= 8) & (ELL <= 400)
                        rows.append(int(a["row"][i])); dtt.append(np.max(np.abs(y[0] / x[0] - 1)))
                        dpp.append(np.max(np.abs(y[2, m] / x[2, m] - 1)))
            counts[(module, points)] = dict(n=len(a["row"]), finite_unpatched=finite_a, finite_patched=finite_b,
                                            identical=same, changed=changed)
            if module == "hmcode" and points == "refused":
                ax.scatter([fidm[r] for r in rows], dpp, s=10, c=C_PATCH, label=r"$C_L^{\phi\phi}$, $8\leq L\leq400$")
                ax.scatter([fidm[r] for r in rows], dtt, s=10, c=C_BAD, marker="s", label=r"lensed TT, $l\leq2500$")
                number("HmcodeChanged", changed); number("HmcodeIdentical", same)
                number("HmcodeChangePP", f"{100 * max(dpp):.1f}"); number("HmcodeChangeTT", f"{100 * max(dtt):.1f}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f_{\rm idm}$"); ax.set_ylabel("max |patched/unpatched $-$ 1|")
    ax.set_title("HMcode on Grendel, the 41 points", fontsize=8)
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "grendel_hmcode.pdf")
    plt.close(fig)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"module & points & finite, unpatched & finite, patched & identical & changed \\", r"\midrule"]
    names = {"refused": "41 NaN points", "sample40": "40 random in the box", "best": "best fit"}
    for (module, points), c in counts.items():
        lines.append(f"{module} & {names[points]} & {c['finite_unpatched']}/{c['n']} & {c['finite_patched']}/{c['n']} & "
                     f"{c['identical']} & {c['changed']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "grendel_patch.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 6. the switch-off redshifts (laptop executables)
def table_switch_off():
    rows = {}
    for line in (DATA / "nl_switch_off_laptop.txt").read_text().split("\n"):
        if not line.strip():
            continue
        build, point, module, z, err = line.split()
        z = z.split("=")[1]
        rows.setdefault(point, {})[(build, module)] = "error" if err.endswith("1") else ("$z=$" + z if z != "none" else "--")
    order = ["best_112257", "best_nodrmd", "lcdm_planck"] + sorted(p for p in rows if p.startswith("refused"))
    names = {"best_112257": "best fit", "best_nodrmd": "best fit, DRMD off", "lcdm_planck": r"Planck $\Lambda$CDM"}
    lines = [r"\begin{tabular}{lllll}", r"\toprule",
             r"point & halofit, unpatched & halofit, patched & HMcode, unpatched & HMcode, patched \\", r"\midrule"]
    for p in order:
        r = rows[p]
        lines.append(f"{names.get(p, 'row ' + p.split('_')[1])} & {r[('orig', 'halofit')]} & {r[('guard', 'halofit')]} & "
                     f"{r[('orig', 'hmcode')]} & {r[('guard', 'hmcode')]} \\\\")
        if p == "lcdm_planck":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "switch_off.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 7. the likelihood at corner points
def fig_corner():
    L = json.loads((DATA / "kmax_loglike_laptop_corner.json").read_text())
    res = {}
    for e in L:
        res[(e["kind"], e["row"], e["module"], e["kmax"])] = e["loglikes"]
    pts = []
    for e in L:
        if (e["kind"], e["row"]) not in pts:
            pts.append((e["kind"], e["row"]))
    tot = lambda d: sum(d.values())
    lin = [tot(res[(k, r, "none", 5.0)]) - tot(res[(k, r, "halofit", 30.0)]) for k, r in pts]
    hf5 = [tot(res[(k, r, "halofit", 5.0)]) - tot(res[(k, r, "halofit", 30.0)]) for k, r in pts]
    sp = np.load(DATA / "corner_spectra.npz")
    ppl, pp5 = [], []
    for k, r in pts:
        l = sp[f"guard|{r}|halofit30|l"]
        m = (l >= 40) & (l <= 400)
        ref = sp[f"guard|{r}|halofit30|phiphi"][m]
        ppl.append(np.mean(sp[f"guard|{r}|none5|phiphi"][m] / ref - 1))
        pp5.append(np.mean(sp[f"guard|{r}|halofit5|phiphi"][m] / ref - 1))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    x = np.arange(len(pts))
    cols = [C_REF if k == "best" else (C_BAD if k == "corner" else "#7f8c8d") for k, _ in pts]
    ax[0].bar(x - 0.2, lin, width=0.4, color=cols, label="linear $-$ halofit ($k_{\\max}$ 30)")
    ax[0].bar(x + 0.2, hf5, width=0.4, color=cols, alpha=0.45, hatch="///", label="halofit $k_{\\max}$ 5 $-$ 30")
    ax[0].set_ylabel(r"$\Delta\ln L$ (CamSpec + low-$l$)")
    ax[0].axhline(0, c="k", lw=0.5)
    ax[1].bar(x - 0.2, 100 * np.array(ppl), width=0.4, color=cols)
    ax[1].bar(x + 0.2, 100 * np.array(pp5), width=0.4, color=cols, alpha=0.45, hatch="///")
    ax[1].set_ylabel(r"mean $\Delta C_L^{\phi\phi}/C_L^{\phi\phi}$ [%], $40\leq L\leq400$")
    ax[1].axhline(0, c="k", lw=0.5)
    ax[1].axhspan(-2.4, 2.4, color="#eeeeee", zorder=0)
    ax[1].text(len(pts) - 0.5, 2.6, "PR4 amplitude $\\pm1\\sigma$ (approx.)", ha="right", fontsize=6)
    for a in ax:
        a.set_xticks(x, [f"{r}" for _, r in pts], rotation=60, fontsize=6.5)
        a.set_xlabel("training row (black: best fit, red: corner, grey: other)", fontsize=7)
    ax[0].legend(fontsize=6.3, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "corner_likelihood.pdf")
    plt.close(fig)
    corner = [i for i, (k, _) in enumerate(pts) if k == "corner"]
    control = [i for i, (k, _) in enumerate(pts) if k == "control"]
    best = [i for i, (k, _) in enumerate(pts) if k == "best"][0]
    number("CornerLinMin", f"{min(-lin[i] for i in corner):.1f}"); number("CornerLinMax", f"{max(-lin[i] for i in corner):.1f}")
    number("CornerHfMax", f"{max(abs(hf5[i]) for i in corner):.2f}")
    number("ControlLinMin", f"{min(-lin[i] for i in control):.1f}"); number("ControlLinMax", f"{max(-lin[i] for i in control):.1f}")
    number("BestLin", f"{-lin[best]:.2f}")
    number("CornerPPMin", f"{-100 * max(ppl[i] for i in corner):.0f}"); number("CornerPPMax", f"{-100 * min(ppl[i] for i in corner):.0f}")
    number("ControlPPMin", f"{-100 * max(ppl[i] for i in control):.1f}"); number("ControlPPMax", f"{-100 * min(ppl[i] for i in control):.1f}")
    number("BestPPLin", f"{-100 * ppl[best]:.1f}")
    number("CornerPPfiveMax", f"{-100 * min(pp5[i] for i in corner):.1f}")
    worst = 0.0
    for r in sp["rows"]:
        for v in ("none5", "halofit5", "halofit30"):
            for s in ("TT", "phiphi"):
                worst = max(worst, float(np.max(np.abs(sp[f"guard|{r}|{v}|{s}"] / sp[f"orig|{r}|{v}|{s}"] - 1))))
    number("CornerGuardOrig", f"{worst:.0e}".replace("e-", r"\times10^{-") + "}")


# ---------------------------------------------------------------- timing
def timing():
    for module in ("halofit", "hmcode"):
        s = json.loads((DATA / f"kmax_ladder_laptop_{module}_best.json").read_text())["summary"]
        t = {x["kmax"]: x["median_seconds"] for x in s}
        number(f"Time{module.capitalize()}Ten", f"{100 * (t[10.0] / t[5.0] - 1):.0f}")
        number(f"Time{module.capitalize()}Thirty", f"{100 * (t[30.0] / t[5.0] - 1):.0f}")


def write_numbers():
    lines = [f"\\newcommand{{\\N{name}}}{{{value}}}" for name, value in NUMBERS.items()]
    (HERE / "numbers.tex").write_text("% written by build.py\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    fig_parameter_space()
    fig_linear_pk()
    fig_table_end()
    fig_kmax()
    fig_grendel()
    table_switch_off()
    fig_corner()
    timing()
    write_numbers()
    for name, value in NUMBERS.items():
        print(f"{name:24s} {value}")
