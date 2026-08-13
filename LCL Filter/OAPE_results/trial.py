


def plot_voltage_all_components(df_without, df_with, figures_dir, filename, bands_L1, bands_C, bands_L2):

    # ----------------------------------------#
    # Common x-axis
    # ----------------------------------------#

    Ds_wo, De_wo, ts_wo, te_wo = normalise(df_without["D_cycle_C"].values)
    Ds_w, De_w, ts_w, te_w = normalise(df_with["D_cycle_C"].values)

    Dm_wo = 0.5 * (Ds_wo + De_wo)
    Dm_w = 0.5 * (Ds_w + De_w)

    D_smooth = np.linspace(0.0, 1.0, 500)

    def smooth(D_mid, y):
        return PchipInterpolator(D_mid, y, extrapolate=True)(D_smooth)

    def align(wo_s, w_s):
        offset = wo_s[0] - w_s[0]
        return w_s + offset

    # ----------------------------------------#
    # L1
    # ----------------------------------------#

    total_wo_L1 = df_without["V_L1_RMS"].values[:len(Ds_wo)]
    total_w_L1 = df_with["V_L1_RMS"].values[:len(Ds_w)]
    
    total_wo_L1_s = smooth(Dm_wo, total_wo_L1)
    total_w_L1_s = smooth(Dm_w, total_w_L1)
    total_w_L1_s = align(total_wo_L1_s, smooth(Dm_w, total_w_L1))



    # ----------------------------------------#
    # C
    # ----------------------------------------#

    total_wo_C = df_without["V_C_RMS"].values[:len(Ds_wo)]
    total_w_C = df_with["V_C_RMS"].values[:len(Ds_w)]

    total_wo_C_s = smooth(Dm_wo, total_wo_C)
    total_w_C_s = smooth(Dm_w, total_w_C)
    total_w_C_s = align(total_wo_C_s, smooth(Dm_w, total_w_C))

    # ----------------------------------------#
    # L2
    # ----------------------------------------#

    total_wo_L2 = df_without["V_L2_RMS"].values[:len(Ds_wo)]
    total_w_L2 = df_with["V_L2_RMS"].values[:len(Ds_w)]

    total_wo_L2_s = smooth(Dm_wo, total_wo_L2)
    total_w_L2_s = smooth(Dm_w, total_w_L2)
    total_w_L2_s = align(total_wo_L2_s, smooth(Dm_w, total_w_L2))


    # ----------------------------------------#
    # Figure
    # ----------------------------------------#

    n_rows = len(bands_L1) + len(bands_C) + len(bands_L2)

    fig = plt.figure(figsize=(6.4, 4.8 * 3*0.75))
    gs = fig.add_gridspec(3, 1, hspace=0.125)

    def make_axes(gs_cell, bands, share_ax):
        heights = [b[1] - b[0] for b in bands][::-1]
        sub = gs_cell.subgridspec(len(bands), 1, height_ratios=heights, hspace=0.1)
        axes = []
        for row in range(len(bands)):
            ax = fig.add_subplot(sub[row], sharex=share_ax) if share_ax is not None else fig.add_subplot(sub[row])
            axes.append(ax)
            if share_ax is None:
                share_ax = ax
        return axes, share_ax

    axes_L1, share_ax = make_axes(gs[0], bands_L1, None)
    axes_C, share_ax = make_axes(gs[1], bands_C, share_ax)
    axes_L2, share_ax = make_axes(gs[2], bands_L2, share_ax)

    def style(axes, bands):
        for row in range(len(bands)):
            ax = axes[row]
            lo, hi = bands[len(bands) - 1 - row]
            ax.set_ylim(lo, hi)
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
            if row > 0:
                ax.spines["top"].set_visible(False)
            if row < len(bands) - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False)

    # ----- L1 panel -----
    for ax_L1 in axes_L1:
        ax_L1.plot(D_smooth, total_wo_L1_s, color="red", linestyle="-", linewidth=4.0)
        ax_L1.plot(D_smooth, total_w_L1_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L1, bands_L1)
    axes_L1[0].set_title("Inverter-side inductor L1")

    # ----- C panel -----
    for ax_C in axes_C:
        ax_C.plot(D_smooth, total_wo_C_s, color="red", linestyle="-", linewidth=4.0)
        ax_C.plot(D_smooth, total_w_C_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_C, bands_C)
    axes_C[0].set_title("Filter capacitor C")

    # ----- L2 panel -----
    for ax_L2 in axes_L2:
        ax_L2.plot(D_smooth, total_wo_L2_s, color="red", linestyle="-", linewidth=4.0)
        ax_L2.plot(D_smooth, total_w_L2_s, color="blue", linestyle="-", linewidth=2.0)
    style(axes_L2, bands_L2)
    axes_L2[0].set_title("Grid-side inductor L2")

    # hide x tick labels on every row except the very last
    all_axes = axes_L1 + axes_C + axes_L2
    for ax in all_axes[:-1]:
        ax.tick_params(labelbottom=False)

    all_axes[-1].set_xlim(0, 1)
    all_axes[-1].set_xlabel("Normalised service life [-]")
    fig.supylabel("Voltage [A]")

    handles = [Patch(facecolor="red", label="Capacitance fixed"),
               Line2D([], [], color="black", linestyle="-", label="Total"),
               Patch(facecolor="blue", label="Capacitance degrading"),
               Line2D([], [], color="black", linestyle="--", label="Fundamental")]

    axes_L1[0].legend(handles=handles, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.15), frameon=True,
                      handlelength=2, markerscale=1.0, borderpad=0.5,
                      columnspacing=1, labelspacing=0.25)

    fig.savefig(f"{figures_dir}/voltage_all_components", dpi=600, bbox_inches="tight")
    plt.close(fig)