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

# plot Hua-Sheng's and EW results
# envelope: whether to include an envelope
# points: whether to plot the actual points
def plot_HS_EW(HSresults, EWresults, K3N3LL2, result_type, energy, pdfset, MH,
            envelope=True, points=True):
    """
    HSresults[(result_type, order, energy, pdfset, MH)] = XS DataFrame
    K3N3LL2[(result_type, energy, pdfset, MH)]          = df1/df2 ratios (same structure)

    EWresults is expected to contain the EW K-factor information. Supported formats:
      - pandas.DataFrame with (bin_low, K) as 2 columns (any names), or with columns like
        ["M", "K"], ["Mhh", "K_EW"], etc.
      - numpy array / list of shape (N, 2): [[bin_low, K], ...]
      - dict-like in the same key-space as HSresults (see key guesses below)
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
                      color="k",
                      label=r"$K_{\mathrm{QCD}}=\mathrm{N3LON3LL}/\mathrm{NNLO}$")

        yvals = ratio_central.copy()

        if envelope:
            r_min = ratio_df.min(axis=1).to_numpy()
            r_max = ratio_df.max(axis=1).to_numpy()
            ax_ratio.fill_between(bin_centers, r_min, r_max,
                                  step="mid", alpha=0.2, color="grey",
                                  label="scale vars")
            yvals = np.concatenate([yvals, r_min, r_max])

        # ===================== EW K-factor overlay (from EWresults) =====================
        ew_obj = None
        if EWresults is not None:
            # try a few common key conventions without forcing one
            candidate_keys = [
                (result_type, energy, pdfset, MH),
                ('EW', result_type, energy, pdfset, MH),
                (result_type, 'EW', energy, pdfset, MH),
                (result_type, 'EW', energy, pdfset),
                (result_type, energy, pdfset),
            ]
            for kk in candidate_keys:
                try:
                    ew_obj = EWresults[kk]
                    break
                except Exception:
                    pass

        ew_centers = None
        ew_k = None

        if ew_obj is not None:
            # Normalize to two 1D arrays: m_low_list and k_list
            if isinstance(ew_obj, pd.DataFrame):
                if ew_obj.shape[1] == 1:
                    # index = bin_low, single column = K
                    m_low_list = np.asarray(ew_obj.index.to_numpy(), dtype=float)
                    k_list = np.asarray(ew_obj.iloc[:, 0].to_numpy(), dtype=float)
                else:
                    cols = list(ew_obj.columns)

                    # heuristic: pick first mass-like col and first K-like col
                    mass_like = [c for c in cols if str(c).lower() in ["m", "mhh", "mh", "bin_low", "low", "edge", "mass"]]
                    k_like = [c for c in cols if "k" in str(c).lower()]

                    mcol = mass_like[0] if len(mass_like) else cols[0]
                    kcol = k_like[0] if len(k_like) else cols[1]

                    m_low_list = np.asarray(ew_obj[mcol].to_numpy(), dtype=float)
                    k_list     = np.asarray(ew_obj[kcol].to_numpy(), dtype=float)

            else:
                arr = np.asarray(ew_obj, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    m_low_list = arr[:, 0].astype(float)
                    k_list     = arr[:, 1].astype(float)
                else:
                    m_low_list = None
                    k_list = None

            if m_low_list is not None and k_list is not None and len(m_low_list) >= 2:
                # Given convention: K at midpoint of consecutive "bin_low" values:
                # center_i = (m_low[i] + m_low[i+1]) / 2 corresponds to k[i]
                ncent = min(len(k_list), len(m_low_list) - 1)
                ew_centers = 0.5 * (m_low_list[:ncent] + m_low_list[1:ncent + 1])
                ew_k = k_list[:ncent]

                # Interpolate EW K to your HS bin centers (so it overlays cleanly)
                ew_k_on_bins = np.interp(bin_centers, ew_centers, ew_k,
                                         left=np.nan, right=np.nan)
                valid = np.isfinite(ew_k_on_bins)
                if np.any(valid):
                    i0 = np.argmax(valid)
                    i1 = len(valid) - 1 - np.argmax(valid[::-1])

                    x_ew = bin_centers[i0:i1 + 1]
                    y_ew = ew_k_on_bins[i0:i1 + 1]

                    ax_ratio.step(x_ew, y_ew, where="mid",
                                  color="C3", linestyle="--", linewidth=1.5,
                                  label=r"$K_{\mathrm{EW}}$")
                    yvals = np.concatenate([yvals, y_ew])

        # Labels, limits, legend
        ax_ratio.set_xlabel(r"$M_{hh}$ [GeV]", fontsize=18)
        ax_ratio.set_ylabel(r"$K$", fontsize=10)
        ax_ratio.set_xlim(bin_low[0], max_x)
        ax_ratio.grid(True, which='major', alpha=0.3)
        ax_ratio.grid(True, which='minor', alpha=0.15)

        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
        if ymin == ymax:
            ymin *= 0.9
            ymax *= 1.1
        ax_ratio.set_ylim(0.95 * ymin, 1.05 * ymax)

        ax_ratio.minorticks_on()
        ax_ratio.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax_ratio.legend()

    # visually merge panels
    plt.subplots_adjust(hspace=0.0)
    fig.tight_layout()
    fig.savefig('plots/' + f"HS_EW_{result_type}_{energy}TeV_{pdfset}_mH{MH}.pdf")
    return fig


# plot NNLO_FTapprox only (single panel)
# envelope: whether to include an envelope (scale-min .. scale-max)
# points: whether to plot the central line (scale-central)
def plot_NNLOFTapprox(NNLO_FTapprox_results, result_type, order, energy,
                      envelope=True, points=True):
    """
    NNLO_FTapprox_results[(result_type, order, energy)] = DataFrame with columns:
      ["left-edge","right-edge","scale-central","central-error",
       "scale-min","min-error","scale-max","max-error","rel-down","rel-up"]

    Parameters
    ----------
    order : str
        In your case this is "NNLO" (passed explicitly).
    divide_by_bin_width : bool
        If True, plot dσ/dX by dividing scale-central/min/max by (right-left).
    """

    key = (result_type, order, energy)
    if key not in NNLO_FTapprox_results:
        raise KeyError(f"Key {key} not found in NNLO_FTapprox_results")

    df = NNLO_FTapprox_results[key].copy()

    # ensure numeric
    for c in ["left-edge", "right-edge", "scale-central", "scale-min", "scale-max"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bin_low  = df["left-edge"].to_numpy(dtype=float)
    bin_high = df["right-edge"].to_numpy(dtype=float)
    bin_centers = 0.5 * (bin_low + bin_high)
    bin_width   = bin_high - bin_low

    y_c   = df["scale-central"].to_numpy(dtype=float)
    y_min = df["scale-min"].to_numpy(dtype=float)
    y_max = df["scale-max"].to_numpy(dtype=float)

    # --- Create figure (single panel) ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    if envelope:
        ax.fill_between(
            bin_centers, y_min, y_max,
            step="mid", alpha=0.2, color="grey",
            label="scale envelope"
        )

    if points:
        ax.step(bin_centers, y_c, where="mid", color="k", lw=1.8,
                label="scale-central")

    # Labels/titles (minimal logic; keep generic)
    if str(result_type).lower() == "mhh":
        ax.set_xlabel(r"$M_{hh}$ [GeV]", fontsize=18)
        ax.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}M_{hh}$ [fb / GeV]", fontsize=18)
        ax.set_yscale("log")
        max_x = min(2000.0, float(np.nanmax(bin_high)))
        ax.set_xlim(float(np.nanmin(bin_low)), max_x)
        ax.set_ylim(0.9e-3)
        obs_title = r"$M_{hh}$"
    else:
        ax.set_xlabel(r"$X$ [GeV]", fontsize=18)
        ax.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}X$ [fb / GeV]", fontsize=18)
        ax.set_yscale("log")
        ax.set_xlim(float(np.nanmin(bin_low)), float(np.nanmax(bin_high)))
        obs_title = str(result_type)

    ax.set_title(rf"{order} FTapprox, {obs_title}, $\sqrt{{s}}={energy}$ TeV", fontsize=14)

    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    ax.minorticks_on()
    ax.legend()

    fig.tight_layout()
    fig.savefig('plots/' + f"NNLO_FTapprox_{order}_{result_type}_{energy}TeV.pdf")
    return fig


# plot Hua-Sheng's, EW and NNLO_FTapprox results in a single plot. NO KFACTORS APPLIED TO FTapprox!
# envelope: whether to include an envelope
# points: whether to plot the actual points
def plot_HS_EW_NNLOFTapprox(HSresults, EWresults, NNLO_FTapprox_results, K3N3LL2,
                            result_type, energy, pdfset, MH,
                            envelope=True, points=True,
                            ft_order="NNLO"):
    """
    HSresults[(result_type, order, energy, pdfset, MH)] = XS DataFrame
    K3N3LL2[(result_type, energy, pdfset, MH)]          = df3/df2 ratios (same structure)

    NNLO_FTapprox_results[(result_type, ft_order, energy)] = DataFrame with columns:
      ["left-edge","right-edge","scale-central","central-error",
       "scale-min","min-error","scale-max","max-error","rel-down","rel-up"]

    EWresults is expected to contain the EW K-factor information (see earlier notes).
    """
    colnames = ["(1.,1.)", "(2.,1.)", "(0.5,1.)",
                "(1.,2.)", "(2.,2.)", "(1.,0.5)", "(0.5,0.5)"]
    central_col = "(1.,1.)"

    # --- Grab HS XS ---
    df3 = HSresults[(result_type, 'N3LON3LL', energy, pdfset, MH)]
    df2 = HSresults[(result_type, 'NNLO',     energy, pdfset, MH)]
    kdf = K3N3LL2[(result_type, energy, pdfset, MH)]

    # --- Grab FTapprox ---
    ft_key = (result_type, ft_order, energy)
    if ft_key not in NNLO_FTapprox_results:
        raise KeyError(f"Key {ft_key} not found in NNLO_FTapprox_results")
    df_ft = NNLO_FTapprox_results[ft_key].copy()

    for c in ["left-edge", "right-edge", "scale-central", "scale-min", "scale-max"]:
        df_ft[c] = pd.to_numeric(df_ft[c], errors="coerce")

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

        xs3 = df3[scale_cols].iloc[0]
        xs2 = df2[scale_cols].iloc[0]
        ratio = (df3[scale_cols] / df2[scale_cols]).iloc[0]

        if set(scale_cols) <= set(kdf.columns):
            diff = (ratio - kdf[scale_cols].iloc[0]).abs().max()
            if diff > 1e-10:
                print(f"[plot_HS] WARNING: max diff between HS ratio and K3N3LL2 = {diff}")

        x = np.arange(len(scale_cols))

        if envelope:
            ax.fill_between(x, xs3.min(), xs3.max(), alpha=0.15, color="C0", label="N3LON3LL scale vars")
            ax.fill_between(x, xs2.min(), xs2.max(), alpha=0.15, color="C1", label="NNLO scale vars")
        if points:
            ax.plot(x, xs3.values, "o-", color="C0", label="N3LON3LL")
            ax.plot(x, xs2.values, "s-", color="C1", label="NNLO")

        ax.set_ylabel(r"$\sigma$ [pb]")
        ax.set_xticks(x)
        ax.set_xticklabels(scale_cols, rotation=45, ha="right")
        ax.set_title(rf"Total cross section, $\sqrt{{s}} = {energy}$ TeV, {pdfset}, $m_H = {MH}$ GeV")
        ax.legend()
        ax.minorticks_on()
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.15)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        ax_ratio.plot(x, ratio.values, "k-", label=r"$\sigma_{\mathrm{N3LON3LL}}/\sigma_{\mathrm{NNLO}}$")
        yvals = ratio.values.copy()
        if envelope:
            r_min = ratio.min()
            r_max = ratio.max()
            ax_ratio.fill_between(x, r_min, r_max, alpha=0.2, color="grey", label="scale vars")
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
        # ---- HS bin edges from MultiIndex ----
        if isinstance(df3.index, pd.MultiIndex) and df3.index.nlevels == 2:
            bin_low = df3.index.get_level_values(0).to_numpy()
            bin_high = df3.index.get_level_values(1).to_numpy()
        else:
            bin_low = df3.iloc[:, 0].to_numpy()
            bin_high = df3.iloc[:, 1].to_numpy()

        bin_centers = 0.5 * (bin_low + bin_high)
        bin_width = bin_high - bin_low

        scale_cols3 = [c for c in colnames if c in df3.columns]
        scale_cols2 = [c for c in colnames if c in df2.columns]
        df3_scales = df3[scale_cols3]
        df2_scales = df2[scale_cols2]

        if central_col not in df3_scales.columns or central_col not in df2_scales.columns:
            raise ValueError(f"Central column {central_col!r} not found in Mhh DataFrames")

        # HS: divide by bin width (so main panel is per GeV)
        df3_scales_dw = df3_scales.div(bin_width, axis=0)
        df2_scales_dw = df2_scales.div(bin_width, axis=0)
        y3 = df3_scales_dw[central_col].to_numpy()
        y2 = df2_scales_dw[central_col].to_numpy()

        ratio_df = df3_scales / df2_scales
        ratio_central = ratio_df[central_col].to_numpy()

        # ----- FTapprox (NO unit conversion here; assumed already in the same units as the HS y-axis) -----
        ft_bin_low  = df_ft["left-edge"].to_numpy(dtype=float)
        ft_bin_high = df_ft["right-edge"].to_numpy(dtype=float)
        ft_centers  = 0.5 * (ft_bin_low + ft_bin_high)

        ft_c   = df_ft["scale-central"].to_numpy(dtype=float)
        ft_min = df_ft["scale-min"].to_numpy(dtype=float)
        ft_max = df_ft["scale-max"].to_numpy(dtype=float)

        # ----- MAIN PANEL -----
        if envelope:
            y3_min = df3_scales_dw.min(axis=1).to_numpy()
            y3_max = df3_scales_dw.max(axis=1).to_numpy()
            ax.fill_between(bin_centers, y3_min, y3_max, step="mid",
                            alpha=0.2, color="C0", label="N3LON3LL scale vars")

            y2_min = df2_scales_dw.min(axis=1).to_numpy()
            y2_max = df2_scales_dw.max(axis=1).to_numpy()
            ax.fill_between(bin_centers, y2_min, y2_max, step="mid",
                            alpha=0.2, color="C1", label="NNLO scale vars")

            # FTapprox envelope in RED
            ax.fill_between(ft_centers, ft_min, ft_max, step="mid",
                            alpha=0.2, color="red", label=f"{ft_order} FTapprox scale vars")

        if points:
            ax.step(bin_centers, y3, where="mid", color="C0", label="N3LON3LL (1,1)")
            ax.step(bin_centers, y2, where="mid", color="C1", label="NNLO (1,1)")

            # FTapprox central in RED
            ax.step(ft_centers, ft_c, where="mid", color="red", lw=1.8,
                    label=f"{ft_order} FTapprox (central)")

        ax.set_yscale("log")
        ax.set_ylabel(r"$\mathrm{d}\sigma / \mathrm{d}M_{hh}$ [fb / GeV]", fontsize=18)
        ax.set_title(rf"$M_{{hh}}$ distribution, $\sqrt{{s}} = {energy}$ TeV, {pdfset}, $m_H = {MH}$ GeV")
        ax.legend()
        ax.minorticks_on()
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.15)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        max_x = min(2000.0, bin_high[-1])
        ax.set_xlim(bin_low[0], max_x)
        ax.set_ylim(0.9E-3)

        # ----- RATIO PANEL (unchanged) -----
        ax_ratio.step(bin_centers, ratio_central, where="mid",
                      color="k", label=r"$K_{\mathrm{QCD}}=\mathrm{N3LON3LL}/\mathrm{NNLO}$")

        yvals = ratio_central.copy()
        if envelope:
            r_min = ratio_df.min(axis=1).to_numpy()
            r_max = ratio_df.max(axis=1).to_numpy()
            ax_ratio.fill_between(bin_centers, r_min, r_max, step="mid",
                                  alpha=0.2, color="grey", label="scale vars")
            yvals = np.concatenate([yvals, r_min, r_max])

        # ===================== EW K-factor overlay (from EWresults) =====================
        ew_obj = None
        if EWresults is not None:
            candidate_keys = [
                (result_type, energy, pdfset, MH),
                ('EW', result_type, energy, pdfset, MH),
                (result_type, 'EW', energy, pdfset, MH),
                (result_type, 'EW', energy, pdfset),
                (result_type, energy, pdfset),
            ]
            for kk in candidate_keys:
                try:
                    ew_obj = EWresults[kk]
                    break
                except Exception:
                    pass

        if ew_obj is not None:
            if isinstance(ew_obj, pd.DataFrame):
                if ew_obj.shape[1] == 1:
                    m_low_list = np.asarray(ew_obj.index.to_numpy(), dtype=float)
                    k_list = np.asarray(ew_obj.iloc[:, 0].to_numpy(), dtype=float)
                else:
                    cols = list(ew_obj.columns)
                    mass_like = [c for c in cols if str(c).lower() in ["m", "mhh", "mh", "bin_low", "low", "edge", "mass"]]
                    k_like = [c for c in cols if "k" in str(c).lower()]
                    mcol = mass_like[0] if len(mass_like) else cols[0]
                    kcol = k_like[0] if len(k_like) else cols[1]
                    m_low_list = np.asarray(ew_obj[mcol].to_numpy(), dtype=float)
                    k_list     = np.asarray(ew_obj[kcol].to_numpy(), dtype=float)
            else:
                arr = np.asarray(ew_obj, dtype=float)
                m_low_list = arr[:, 0] if (arr.ndim == 2 and arr.shape[1] >= 2) else None
                k_list     = arr[:, 1] if (arr.ndim == 2 and arr.shape[1] >= 2) else None

            if m_low_list is not None and k_list is not None and len(m_low_list) >= 2:
                ncent = min(len(k_list), len(m_low_list) - 1)
                ew_centers = 0.5 * (m_low_list[:ncent] + m_low_list[1:ncent + 1])
                ew_k = k_list[:ncent]

                ew_k_on_bins = np.interp(bin_centers, ew_centers, ew_k, left=np.nan, right=np.nan)
                valid = np.isfinite(ew_k_on_bins)
                if np.any(valid):
                    i0 = np.argmax(valid)
                    i1 = len(valid) - 1 - np.argmax(valid[::-1])
                    ax_ratio.step(bin_centers[i0:i1 + 1], ew_k_on_bins[i0:i1 + 1],
                                  where="mid", color="C3", linestyle="--", linewidth=1.5,
                                  label=r"$K_{\mathrm{EW}}$")
                    yvals = np.concatenate([yvals, ew_k_on_bins[i0:i1 + 1]])

        ax_ratio.set_xlabel(r"$M_{hh}$ [GeV]", fontsize=18)
        ax_ratio.set_ylabel(r"$K$", fontsize=10)
        ax_ratio.set_xlim(bin_low[0], max_x)
        ax_ratio.grid(True, which='major', alpha=0.3)
        ax_ratio.grid(True, which='minor', alpha=0.15)

        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
        if ymin == ymax:
            ymin *= 0.9
            ymax *= 1.1
        ax_ratio.set_ylim(0.95 * ymin, 1.05 * ymax)

        ax_ratio.minorticks_on()
        ax_ratio.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax_ratio.legend()

    plt.subplots_adjust(hspace=0.0)
    fig.tight_layout()
    fig.savefig('plots/' + f"HS_EW_FTapprox_{result_type}_{energy}TeV_{pdfset}_mH{MH}.pdf")
    return fig
