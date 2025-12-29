import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# plot Hua-Sheng's results:
# envelope: whether to include an envelope
# points: whether to plot the actual points
def plot_HS(HSresults, K3N3LL2, result_type, energy, pdfset, MH,
            envelope=True, points=True):
    """
    HSresults[(result_type, order, energy, pdfset, MH)] = XS DataFrame
    K3N3LL2[(result_type, energy, pdfset, MH)]          = df1/df2 ratios (same structure)
    """
    colnames = ["(1.,1.)", "(2.,1.)", "(0.5,1.)",
                "(1.,2.)", "(2.,2.)", "(1.,0.5)", "(0.5,0.5)"]
    central_col = "(1.,1.)"

    # --- Grab XS ---
    df3 = HSresults[(result_type, 'N3LON3LL', energy, pdfset, MH)]
    df2 = HSresults[(result_type, 'NNLO',     energy, pdfset, MH)]

    # --- Grab ratio df (df3/df2) ---
    kdf = K3N3LL2[(result_type, energy, pdfset, MH)]

    # Figure with 2 merged panels
    fig, (ax, ax_ratio) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(7, 6)
    )

    # ================= TOTALXS =================
    if result_type != "Mhh":
        scale_cols = [c for c in colnames if c in df3.columns and c in df2.columns]

        xs3 = df3[scale_cols].iloc[0]  # N3LON3LL
        xs2 = df2[scale_cols].iloc[0]  # NNLO

        ratio = (df3[scale_cols] / df2[scale_cols]).iloc[0]

        # optional sanity check vs K3N3LL2
        if set(scale_cols) <= set(kdf.columns):
            diff = (ratio - kdf[scale_cols].iloc[0]).abs().max()
            if diff > 1e-10:
                print(f"[plot_HS] WARNING: max diff between HS ratio and K3N3LL2 = {diff}")

        x = np.arange(len(scale_cols))

        # ----- MAIN PANEL -----
        if envelope:
            ax.fill_between(x, xs3.min(), xs3.max(), alpha=0.15, color="C0",
                            label="N3LON3LL scale vars")
            ax.fill_between(x, xs2.min(), xs2.max(), alpha=0.15, color="C1",
                            label="NNLO scale vars")

        if points:
            ax.plot(x, xs3.values, "o-", color="C0", label="N3LON3LL")
            ax.plot(x, xs2.values, "s-", color="C1", label="NNLO")

        ax.set_ylabel(r"$\sigma$ [pb]")
        ax.set_xticks(x)
        ax.set_xticklabels(scale_cols, rotation=45, ha="right")
        ax.set_title(
            rf"Total cross section, $\sqrt{{s}} = {energy}$ TeV, "
            rf"{pdfset}, $m_H = {MH}$ GeV"
        )
        ax.legend()
        ax.minorticks_on()

        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.15)

        # Hide x tick labels/marks on the main (top) panel
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # ----- RATIO PANEL -----
        ax_ratio.plot(x, ratio.values, "k-",
                      label=r"$\sigma_{\mathrm{N3LON3LL}}/\sigma_{\mathrm{NNLO}}$")

        yvals = ratio.values.copy()

        if envelope:
            r_min = ratio.min()
            r_max = ratio.max()
            ax_ratio.fill_between(x, r_min, r_max,
                                  alpha=0.2, color="grey",
                                  label="scale vars")
            yvals = np.concatenate([yvals, [r_min, r_max]])

        ax_ratio.set_ylabel(r"$K$", fontsize=10)
        ax_ratio.set_xticks(x)
        ax_ratio.set_xticklabels(scale_cols, rotation=45, ha="right")
        ax_ratio.grid(True, which='major', alpha=0.3)
        ax_ratio.grid(True, which='minor', alpha=0.15)
        ymin, ymax = np.min(yvals), np.max(yvals)
        if ymin == ymax:
            ymin *= 0.9
            ymax *= 1.1
        ax_ratio.set_ylim(0.95 * ymin, 1.05 * ymax)
        ax_ratio.minorticks_on()
        ax_ratio.legend()

    # ================= Mhh =================
    else:
        # ---- Bin edges from MultiIndex (correct source) ----
        if isinstance(df3.index, pd.MultiIndex) and df3.index.nlevels == 2:
            bin_low = df3.index.get_level_values(0).to_numpy()
            bin_high = df3.index.get_level_values(1).to_numpy()
        else:
            # fallback if read_HS ever changes
            bin_low = df3.iloc[:, 0].to_numpy()
            bin_high = df3.iloc[:, 1].to_numpy()

        bin_centers = 0.5 * (bin_low + bin_high)
        nbins = len(bin_centers)
        bin_width = bin_high - bin_low  # bin widths

        # XS scale columns (all are scales; bin edges are in the index)
        scale_cols3 = [c for c in colnames if c in df3.columns]
        scale_cols2 = [c for c in colnames if c in df2.columns]

        df3_scales = df3[scale_cols3]
        df2_scales = df2[scale_cols2]

        if central_col not in df3_scales.columns or central_col not in df2_scales.columns:
            raise ValueError(f"Central column {central_col!r} not found in Mhh DataFrames")

        # --- divide by bin width for the main panel ---
        df3_scales_dw = df3_scales.div(bin_width, axis=0)
        df2_scales_dw = df2_scales.div(bin_width, axis=0)

        # central y-values AFTER dividing by bin width
        y3 = df3_scales_dw[central_col].to_numpy()
        y2 = df2_scales_dw[central_col].to_numpy()

        # ratio from HS directly (bin width cancels)
        ratio_df = df3_scales / df2_scales
        ratio_central = ratio_df[central_col].to_numpy()

        # sanity check vs K3N3LL2: drop first two columns (bin edges)
        if kdf.shape[1] > 2:
            k_ratio_df = kdf.iloc[:, 2:]
            if central_col in k_ratio_df.columns:
                diff = np.max(np.abs(ratio_central - k_ratio_df[central_col].to_numpy()))
                if diff > 1e-10:
                    print(f"[plot_HS] WARNING: Mhh central ratio vs K3N3LL2 differ by up to {diff}")

        # ----- MAIN PANEL -----
        if envelope:
            # envelopes also / bin width
            y3_min = df3_scales_dw.min(axis=1).to_numpy()
            y3_max = df3_scales_dw.max(axis=1).to_numpy()
            ax.fill_between(bin_centers, y3_min, y3_max,
                            step="mid", alpha=0.2, color="C0",
                            label="N3LON3LL scale vars")

            y2_min = df2_scales_dw.min(axis=1).to_numpy()
            y2_max = df2_scales_dw.max(axis=1).to_numpy()
            ax.fill_between(bin_centers, y2_min, y2_max,
                            step="mid", alpha=0.2, color="C1",
                            label="NNLO scale vars")

        if points:
            ax.step(bin_centers, y3, where="mid", color="C0", label="N3LON3LL (1,1)")
            ax.step(bin_centers, y2, where="mid", color="C1", label="NNLO (1,1)")

        # log y-scale, labels, title
        ax.set_yscale("log")
        ax.set_ylabel(r"$\mathrm{d}\sigma / \mathrm{d}M_{hh}$ [fb / GeV]", fontsize=18)
        ax.set_title(
            rf"$M_{{hh}}$ distribution, $\sqrt{{s}} = {energy}$ TeV, "
            rf"{pdfset}, $m_H = {MH}$ GeV"
        )
        ax.legend()
        ax.minorticks_on()

        
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.15)
        
        # Hide x tick labels/marks on the main (top) panel
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # limit plots to 2000 GeV max
        max_x = min(2000.0, bin_high[-1])
        ax.set_xlim(bin_low[0], max_x)

        min_y = 0.9E-3
        ax.set_ylim(min_y)

        # ----- RATIO PANEL -----
        ax_ratio.step(bin_centers, ratio_central, where="mid",
                      color="k")#,
                      #label=r"$\frac{(\mathrm{d}\sigma/\mathrm{d}M_{hh})_{\mathrm{N3LON3LL}}}{(\mathrm{d}\sigma/\mathrm{d}M_{hh})_{\mathrm{NNLO}}}$")

        yvals = ratio_central.copy()

        if envelope:
            r_min = ratio_df.min(axis=1).to_numpy()
            r_max = ratio_df.max(axis=1).to_numpy()
            ax_ratio.fill_between(bin_centers, r_min, r_max,
                                  step="mid", alpha=0.2, color="grey",
                                  label="scale vars")
            yvals = np.concatenate([yvals, r_min, r_max])

        ax_ratio.set_xlabel(r"$M_{hh}$ [GeV]", fontsize=18)
        ax_ratio.set_ylabel(r"$K$(N3LON3LL/NNLO)", fontsize=10)
        ax_ratio.set_xlim(bin_low[0], max_x)
        ax_ratio.grid(True, which='major', alpha=0.3)
        ax_ratio.grid(True, which='minor', alpha=0.15)
        ymin, ymax = np.min(yvals), np.max(yvals)
        if ymin == ymax:
            ymin *= 0.9
            ymax *= 1.1
        ax_ratio.set_ylim(0.95 * ymin, 1.05 * ymax)

        ax_ratio.minorticks_on()
        # optional reference line at 1
        ax_ratio.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5)

        ax_ratio.legend()

    # visually merge panels
    plt.subplots_adjust(hspace=0.0)
    fig.tight_layout()
    fig.savefig('plots/' + f"HS_{result_type}_{energy}TeV_{pdfset}_mH{MH}.pdf")
    return fig
