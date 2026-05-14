"""
Generate three figures for the paper:
  fig2_binscatter.pdf  – Conditional binned scatterplot (FWL residuals)
  fig3_marginal_hc.pdf – Marginal effect of suitability over HC distribution
  fig4_lag_coefs.pdf   – Coefficient plot for lagged suitability specifications
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import pyhdfe, pyreadstat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from linearmodels.panel import PanelOLS

DATA  = Path("/root/.claude/uploads/357fed84-0ba0-4950-93d2-6341dc3545a4/4a3341af-__.dta")
OUT   = Path("/home/user/-")

# ── aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        180,
})

# ── load & prepare data ───────────────────────────────────────────────────────
df_raw, _ = pyreadstat.read_dta(str(DATA), apply_value_formats=False)
df = df_raw.sort_values(["id","year"]).reset_index(drop=True)
df["lnedu70_c"] = df["lnedu70"] - df["lnedu70"].mean(skipna=True)
df["lnmpkl_x_lnedu70_c"] = df["lnmpkl"] * df["lnedu70_c"]

for col in ["lnmpkl"]:
    for lag in [1, 2, 3]:
        df[f"l{lag}_{col}"] = df.groupby("id")[col].shift(lag)

CONTROLS = ["tra", "efi", "old", "lnpop"]

# ── helper: absorb two-way FE and run PanelOLS ────────────────────────────────
def twfe(df_in, y, xvars, bw=3):
    data = df_in[["id","year",y]+xvars].dropna().set_index(["id","year"])
    mod  = PanelOLS(data[y], data[xvars],
                    entity_effects=True, time_effects=True,
                    drop_absorbed=True, check_rank=False)
    r = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw)
    return r

def demean_twfe(df_in, cols):
    """Partial out entity + year FE via pyhdfe MAP."""
    need = ["id","year"] + cols
    dc = df_in[need].dropna().reset_index(drop=True)
    ids_arr = np.column_stack([dc["id"].values.astype(int),
                               dc["year"].values.astype(int)])
    alg = pyhdfe.create(ids_arr, residualize_method="map")
    dm  = alg.residualize(dc[cols].values.astype(float))
    return pd.DataFrame(dm, columns=cols), dc

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Conditional Binned Scatterplot
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2 ...")
cols2 = ["lngini","lnmpkl"] + CONTROLS
dm2, _ = demean_twfe(df, cols2)

# FWL slope via OLS on residuals
x = dm2["lnmpkl"].values
y_res = dm2["lngini"].values
slope = np.polyfit(x, y_res, 1)[0]

# 20-quantile bins of residualised lnSuit
bins = pd.qcut(dm2["lnmpkl"], q=20, duplicates="drop")
bin_x = dm2.groupby(bins)["lnmpkl"].mean()
bin_y = dm2.groupby(bins)["lngini"].mean()

fig2, ax2 = plt.subplots(figsize=(5.0, 3.8))
ax2.scatter(bin_x, bin_y, color="#2166AC", s=28, zorder=5, label="_nolegend_")
xline = np.linspace(dm2["lnmpkl"].min(), dm2["lnmpkl"].max(), 200)
ax2.plot(xline, slope*xline, color="#D6604D", lw=1.5,
         label=f"FWL slope = {slope:.3f}")
ax2.axhline(0, color="grey", lw=0.5, ls="--")
ax2.axvline(0, color="grey", lw=0.5, ls="--")
ax2.set_xlabel("Residualized $\\ln \\mathrm{Suit}$")
ax2.set_ylabel("Residualized $\\ln \\mathrm{Gini}$")
ax2.set_title("Conditional Binned Scatterplot", fontsize=10, pad=6)
ax2.legend(loc="upper right", framealpha=0.0)
fig2.tight_layout()
fig2.savefig(OUT / "fig2_binscatter.pdf", bbox_inches="tight")
fig2.savefig(OUT / "fig2_binscatter.png", bbox_inches="tight", dpi=180)
plt.close(fig2)
print(f"  FWL slope = {slope:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Marginal effect of suitability over the HC distribution
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3 ...")
r3 = twfe(df, "lngini", ["lnmpkl","lnmpkl_x_lnedu70_c"] + CONTROLS)
b1 = float(r3.params["lnmpkl"])
b2 = float(r3.params["lnmpkl_x_lnedu70_c"])

# Full DK covariance for (b1, b2)
vcov = r3.cov.loc[["lnmpkl","lnmpkl_x_lnedu70_c"],
                   ["lnmpkl","lnmpkl_x_lnedu70_c"]].values

hc_grid = np.linspace(df["lnedu70"].dropna().min() - 0.05,
                      df["lnedu70"].dropna().max() + 0.05, 300)
hc_c    = hc_grid - df["lnedu70"].mean(skipna=True)  # demeaned

me  = b1 + b2 * hc_c
# Var(me) = Var(b1) + hc_c^2 * Var(b2) + 2*hc_c*Cov(b1,b2)
var_me = (vcov[0,0]
          + hc_c**2 * vcov[1,1]
          + 2 * hc_c * vcov[0,1])
se_me = np.sqrt(np.maximum(var_me, 0))
ci_lo = me - 1.96 * se_me
ci_hi = me + 1.96 * se_me

# HC percentile markers
p25, p50, p75 = np.nanpercentile(df["lnedu70"], [25, 50, 75])

# threshold where CI upper bound = 0
# find first x where ci_hi < 0
threshold_mask = ci_hi < 0
threshold_x = hc_grid[threshold_mask][0] if threshold_mask.any() else None

fig3, ax3 = plt.subplots(figsize=(5.5, 4.0))
ax3.fill_between(hc_grid, ci_lo, ci_hi, color="#9ECAE1", alpha=0.5, label="95% CI")
ax3.plot(hc_grid, me, color="#2166AC", lw=1.8, label="Marginal effect")
ax3.axhline(0, color="grey", lw=0.8, ls="--")
for pct in [p25, p50, p75]:
    ax3.axvline(pct, color="#AAAAAA", lw=0.8, ls=":")
# annotate threshold
if threshold_x is not None:
    ax3.axvline(threshold_x, color="#D6604D", lw=1.0, ls="--", alpha=0.8)
    yspan = ax3.get_ylim()
    ax3.text(threshold_x + 0.03, ci_lo.max() * 0.6,
             "95% CI below zero", fontsize=7.5, color="#D6604D",
             ha="left", va="top")
ax3.set_xlabel("Pre-sample human capital, $\\ln \\mathrm{HC}_{70}$")
ax3.set_ylabel("$\\partial \\ln \\mathrm{Gini}\\ /\\ \\partial \\ln \\mathrm{Suit}$")
ax3.set_title("Suitability Becomes More Equalizing at Higher Human Capital",
              fontsize=9.5, pad=6)
ax3.legend(loc="upper right", framealpha=0.0)
fig3.tight_layout()
fig3.savefig(OUT / "fig3_marginal_hc.pdf", bbox_inches="tight")
fig3.savefig(OUT / "fig3_marginal_hc.png", bbox_inches="tight", dpi=180)
plt.close(fig3)
print(f"  b1={b1:.4f}  b2={b2:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Coefficient plot: lagged suitability L1, L2, L3
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4 ...")

lag_results = {}
for lag in [1, 2, 3]:
    v = f"l{lag}_lnmpkl"
    r = twfe(df, "lngini", [v] + CONTROLS)
    coef = float(r.params[v])
    se   = float(r.std_errors[v])
    pval = float(r.pvalues[v])
    lag_results[lag] = dict(coef=coef, se=se, pval=pval)
    print(f"  L{lag}: {coef:.4f} ({se:.4f}) p={pval:.3f}")

lags   = [1, 2, 3]
coefs  = [lag_results[l]["coef"] for l in lags]
ses    = [lag_results[l]["se"]   for l in lags]
pvals  = [lag_results[l]["pval"] for l in lags]
ci95   = [1.96 * s for s in ses]

fig4, ax4 = plt.subplots(figsize=(4.8, 3.6))
x_pos = np.array([1, 2, 3])
ax4.errorbar(x_pos, coefs, yerr=ci95,
             fmt="o", color="#2166AC", ecolor="#2166AC",
             elinewidth=1.4, capsize=4, capthick=1.4,
             ms=6, zorder=5)
ax4.axhline(0, color="grey", lw=0.8, ls="--")

# p-value annotations – carefully positioned to avoid cut-off
for xi, coef, se, pval in zip(x_pos, coefs, ses, pvals):
    p_str = f"p={pval:.3f}"  # always "p=0.0XX"
    y_ann = coef - 1.96 * se - 0.002
    ax4.annotate(p_str, xy=(xi, y_ann),
                 xytext=(0, -14), textcoords="offset points",
                 ha="center", va="top", fontsize=8,
                 color="#555555")

ax4.set_xticks(x_pos)
ax4.set_xticklabels([r"$t-1$", r"$t-2$", r"$t-3$"])
ax4.set_xlabel("Lag of suitability")
ax4.set_ylabel("Coefficient on lagged $\\ln \\mathrm{Suit}$")
ax4.set_title("Lagged Suitability Effects Remain Negative", fontsize=10, pad=6)
# expand bottom margin so annotations are fully visible
ax4.set_ylim(min(coefs) - 3.0 * max(ses), max(coefs) + 1.5 * max(ses))
fig4.subplots_adjust(bottom=0.18)
fig4.savefig(OUT / "fig4_lag_coefs.pdf", bbox_inches="tight")
fig4.savefig(OUT / "fig4_lag_coefs.png", bbox_inches="tight", dpi=180)
plt.close(fig4)

print("All figures saved.")
