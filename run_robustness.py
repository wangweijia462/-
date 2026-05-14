"""
Robustness regressions on actual data:
  1. Lagged suitability (L1–L3 baseline, L1+HC interaction) — PanelOLS DK SE
  2. IV/2SLS — linearmodels IV2SLS on pyhdfe-demeaned data, DK SE
     Spec A: entity+year FE, Bartik shift-share instruments
     Spec B: entity+year FE, other-country suitability instruments
  3. Winsorisation check
Outputs: robustness_results.txt  robustness_summary.json
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import pyhdfe
import pyreadstat
import scipy.stats
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

DATA_PATH = Path("/root/.claude/uploads/357fed84-0ba0-4950-93d2-6341dc3545a4/4a3341af-__.dta")
OUT_PATH  = Path("/home/user/-/robustness_results.txt")

def star(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""
def fmt(v, d=4):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "."
    return f"{v:.{d}f}"

# ── load & sort ───────────────────────────────────────────────────────────────
df_raw, _ = pyreadstat.read_dta(str(DATA_PATH), apply_value_formats=False)
df = df_raw.sort_values(["id","year"]).reset_index(drop=True)

# ── boundary variables ────────────────────────────────────────────────────────
for var in ["efi","lnedu70"]:
    cm  = df.groupby("id")[var].transform("mean")
    med = cm.dropna().median()
    df[f"high_{var}"] = np.where(cm.notna(), (cm >= med).astype(float), np.nan)
    df[f"low_{var}"]  = np.where(cm.notna(), (cm <  med).astype(float), np.nan)

df["lnedu70_c"] = df["lnedu70"] - df["lnedu70"].mean(skipna=True)

vb = df[["low_efi","low_lnedu70"]].notna().all(axis=1)
df["low_efi_low_hc"] = np.where(
    vb, ((df["low_efi"]==1)&(df["low_lnedu70"]==1)).astype(float), np.nan)

# ── lags ──────────────────────────────────────────────────────────────────────
for col in ["lnmpkl","lnbk70","bartik","lnothersmpkl"]:
    for lag in [1,2,3,4]:
        df[f"l{lag}_{col}"] = df.groupby("id")[col].shift(lag)

df["l1_lnmpkl_x_lnedu70_c"] = df["l1_lnmpkl"] * df["lnedu70_c"]

CONTROLS = ["tra","efi","old","lnpop"]

# ── helpers ───────────────────────────────────────────────────────────────────
def fit_dk(df_in, y, xvars, bw=3):
    """PanelOLS + Driscoll-Kraay SE."""
    data = df_in[["id","year",y]+xvars].dropna().set_index(["id","year"])
    mod  = PanelOLS(data[y], data[xvars],
                    entity_effects=True, time_effects=True,
                    drop_absorbed=True, check_rank=False)
    r = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)
    return r.params, r.std_errors, r.pvalues, int(r.nobs)

def absorb_fes(df_c, cols, fe_cols):
    """Demean by specified fixed effects using pyhdfe MAP algorithm."""
    ids_arr = np.column_stack([df_c[f].values.astype(int) for f in fe_cols])
    alg     = pyhdfe.create(ids_arr, residualize_method="map")
    dm      = alg.residualize(df_c[cols].values.astype(float))
    return pd.DataFrame(dm, columns=cols)

def fit_iv(df_in, y, endog, exog, instruments, fe_cols, bw=3):
    """
    IV/2SLS: absorb specified FEs, then linearmodels IV2SLS with DK SE.
    Returns dict or None if under-identified.
    """
    need = ["id","year",y]+endog+exog+instruments
    dc   = df_in[need].dropna().reset_index(drop=True)
    dm   = absorb_fes(dc, [y]+endog+exog+instruments, fe_cols)

    # Drop near-zero-variance or perfectly-collinear instruments
    inst_ok = instruments.copy()
    # remove zero-variance columns
    inst_ok = [c for c in inst_ok if dm[c].std() > 1e-10]
    # remove collinear pairs (keep the one with higher partial corr to endog)
    changed = True
    while changed and len(inst_ok) > 1:
        changed = False
        C = np.abs(np.corrcoef(dm[inst_ok].values.T))
        np.fill_diagonal(C, 0)
        if C.max() > 0.9999:
            i, j = np.unravel_index(C.argmax(), C.shape)
            e = dm[endog[0]].values
            c_i = abs(float(np.corrcoef(dm[inst_ok[i]], e)[0,1]))
            c_j = abs(float(np.corrcoef(dm[inst_ok[j]], e)[0,1]))
            inst_ok.pop(i if c_i < c_j else j)
            changed = True

    if len(inst_ok) < len(endog):
        return None

    mod = IV2SLS(dm[y], dm[exog] if exog else None, dm[endog], dm[inst_ok])
    r   = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)

    # First-stage approximate F (chi2 / df)
    diag   = r.first_stage.diagnostics
    chi2   = float(diag["f.stat"].iloc[0])
    fdist  = str(diag["f.dist"].iloc[0])
    df_num = int(fdist.replace("chi2(","").replace(")",""))
    kp_f   = chi2 / df_num

    # Sargan J overidentification test (available if over-identified)
    try:
        sarg   = r.sargan
        j_stat = float(sarg.stat)
        j_pval = float(sarg.pval)
    except Exception:
        j_stat = float("nan")
        j_pval = float("nan")

    var_names = endog + exog
    coef  = {v: float(r.params[v])      for v in var_names if v in r.params}
    se_d  = {v: float(r.std_errors[v])  for v in var_names if v in r.std_errors}
    pval  = {v: float(r.pvalues[v])     for v in var_names if v in r.pvalues}
    return {"coef": coef, "se": se_d, "pval": pval,
            "nobs": int(r.nobs), "kp_f": kp_f,
            "j_stat": j_stat, "j_pval": j_pval,
            "instruments_used": inst_ok}

# ══════════════════════════════════════════════════════════════════════════════
lines = ["="*72, "REAL REGRESSION RESULTS (actual data)", "="*72]

# ─── 1. LAGGED SUITABILITY ────────────────────────────────────────────────────
lines += ["","="*72,"LAGGED SUITABILITY REGRESSIONS (PanelOLS, DK SE bw=3)","="*72]

lag_res = {}
for lag in [1,2,3]:
    v = f"l{lag}_lnmpkl"
    p,se,pv,n = fit_dk(df, "lngini", [v]+CONTROLS)
    lag_res[lag] = {"coef":float(p[v]),"se":float(se[v]),"pval":float(pv[v]),"nobs":n}
    lines.append(f"\nL{lag} baseline  N={n}")
    lines.append(f"  {v}: {fmt(p[v])} ({fmt(se[v])}) p={fmt(pv[v],3)}{star(pv[v])}")

# L1 + HC interaction
p,se,pv,n = fit_dk(df,"lngini",["l1_lnmpkl","l1_lnmpkl_x_lnedu70_c"]+CONTROLS)
lag_res["L1HC"] = {
    "coef_main": float(p["l1_lnmpkl"]),   "se_main": float(se["l1_lnmpkl"]),
    "pval_main": float(pv["l1_lnmpkl"]),
    "coef_int":  float(p["l1_lnmpkl_x_lnedu70_c"]),
    "se_int":    float(se["l1_lnmpkl_x_lnedu70_c"]),
    "pval_int":  float(pv["l1_lnmpkl_x_lnedu70_c"]),
    "nobs": n}
lines.append(f"\nL1 + HC interaction  N={n}")
for v in ["l1_lnmpkl","l1_lnmpkl_x_lnedu70_c"]:
    lines.append(f"  {v}: {fmt(p[v])} ({fmt(se[v])}) p={fmt(pv[v],3)}{star(pv[v])}")

# ─── 2. IV / 2SLS ─────────────────────────────────────────────────────────────
lines += ["","="*72,
          "IV/2SLS (linearmodels IV2SLS, pyhdfe FE absorption, DK SE bw=3)",
          "="*72]

iv_specs = {
    "bartik": {
        "desc": "Bartik shift-share instruments (entity+year FE)",
        "instruments": ["bartik","l1_bartik"],
        "fe_cols": ["id","year"]},
    "othersmpkl": {
        "desc": "Other-country suitability instruments (entity+year FE)",
        "instruments": ["lnothersmpkl","l1_lnothersmpkl"],
        "fe_cols": ["id","year"]},
}

iv_res = {}
for key, spec in iv_specs.items():
    r = fit_iv(df,"lngini",["lnmpkl"],CONTROLS,
               spec["instruments"], spec["fe_cols"])
    iv_res[key] = r
    if r:
        lines.append(f"\nIV ({spec['desc']})  N={r['nobs']}")
        lines.append(f"  KP-F(approx)={fmt(r['kp_f'],2)}  Sargan J p={fmt(r['j_pval'],3)}")
        v = "lnmpkl"
        lines.append(
            f"  {v}: {fmt(r['coef'][v])} ({fmt(r['se'][v])}) "
            f"p={fmt(r['pval'][v],3)}{star(r['pval'][v])}")
    else:
        lines.append(f"\nIV ({spec['desc']}): under-identified, skipped")

# ─── 3. WINSORISATION ─────────────────────────────────────────────────────────
lines += ["","="*72,"WINSORISATION (1st/99th percentile)","="*72]
from scipy.stats.mstats import winsorize as wz
df["lngini_w"] = wz(df["lngini"].fillna(df["lngini"].median()),  limits=[.01,.01])
df["lnmpkl_w"] = wz(df["lnmpkl"].fillna(df["lnmpkl"].median()), limits=[.01,.01])

p,se,pv,n = fit_dk(df,"lngini_w",["lnmpkl_w"]+CONTROLS)
win_res = {"base":{"coef":float(p["lnmpkl_w"]),"se":float(se["lnmpkl_w"]),
                   "pval":float(pv["lnmpkl_w"]),"nobs":n}}
lines.append(f"\nWinsorised baseline  N={n}")
lines.append(f"  lnmpkl_w: {fmt(p['lnmpkl_w'])} ({fmt(se['lnmpkl_w'])}) "
             f"p={fmt(pv['lnmpkl_w'],3)}{star(pv['lnmpkl_w'])}")

df["lnmpkl_w_x_lnedu70_c"] = df["lnmpkl_w"] * df["lnedu70_c"]
p,se,pv,n = fit_dk(df,"lngini_w",["lnmpkl_w","lnmpkl_w_x_lnedu70_c"]+CONTROLS)
win_res["hc"] = {"coef":float(p["lnmpkl_w_x_lnedu70_c"]),
                 "se":float(se["lnmpkl_w_x_lnedu70_c"]),
                 "pval":float(pv["lnmpkl_w_x_lnedu70_c"]),"nobs":n}
lines.append(f"Winsorised HC interaction  N={n}")
lines.append(f"  lnmpkl_w_x_lnedu70_c: {fmt(p['lnmpkl_w_x_lnedu70_c'])} "
             f"({fmt(se['lnmpkl_w_x_lnedu70_c'])}) "
             f"p={fmt(pv['lnmpkl_w_x_lnedu70_c'],3)}{star(pv['lnmpkl_w_x_lnedu70_c'])}")

# ─── write ────────────────────────────────────────────────────────────────────
OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved → {OUT_PATH}")

import json
summary = {"lags": lag_res, "iv": iv_res, "winsorised": win_res}
Path("/home/user/-/robustness_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
print("JSON summary saved.")
