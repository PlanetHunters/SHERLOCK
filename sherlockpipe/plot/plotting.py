import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def save_transit_plot(object_id, title, plot_dir, file, time, lc, transit_result, cadence, run_no):
    # start the plotting
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), constrained_layout=True)
    fig.suptitle(title)
    # 1-Plot all the transits
    in_transit = transit_result.in_transit
    tls_results = transit_result.results
    ax1.scatter(time[in_transit], lc[in_transit], color='red', s=2, zorder=0)
    ax1.scatter(time[~in_transit], lc[~in_transit], color='black', alpha=0.05, s=2, zorder=0)
    ax1.plot(tls_results.model_lightcurve_time, tls_results.model_lightcurve_model, alpha=1, color='red', zorder=1)
    # plt.scatter(time_n, flux_new_n, color='orange', alpha=0.3, s=20, zorder=3)
    plt.xlim(time.min(), time.max())
    # plt.xlim(1362.0,1364.0)
    ax1.set(xlabel='Time (days)', ylabel='Relative flux')
    # phase folded plus binning
    bins_per_transit = 8
    half_duration_phase = transit_result.duration / 2 / transit_result.period
    if np.isnan(transit_result.period) or np.isnan(transit_result.duration):
        bins = 200
        folded_plot_range = 0.05
    else:
        bins = transit_result.period / transit_result.duration * bins_per_transit
        folded_plot_range = half_duration_phase * 10
    binning_enabled = True
    ax2.plot(tls_results.model_folded_phase, tls_results.model_folded_model, color='red')
    scatter_measurements_alpha = 0.05 if binning_enabled else 0.8
    ax2.scatter(tls_results.folded_phase, tls_results.folded_y, color='black', s=10,
                alpha=scatter_measurements_alpha, zorder=2)
    lower_x_limit = 0.5 - folded_plot_range
    upper_x_limit = 0.5 + folded_plot_range
    ax2.set_xlim(lower_x_limit, upper_x_limit)
    ax2.set(xlabel='Phase', ylabel='Relative flux')
    folded_phase_zoom_mask = np.argwhere((tls_results.folded_phase > lower_x_limit) &
                                         (tls_results.folded_phase < upper_x_limit)).flatten()
    if isinstance(tls_results.folded_phase, (list, np.ndarray)):
        folded_phase = tls_results.folded_phase[folded_phase_zoom_mask]
        folded_y = tls_results.folded_y[folded_phase_zoom_mask]
        ax2.set_ylim(np.min([np.min(folded_y), np.min(tls_results.model_folded_model)]),
                     np.max([np.max(folded_y), np.max(tls_results.model_folded_model)]))
        plt.ticklabel_format(useOffset=False)
        bins = 80
        if binning_enabled and tls_results.SDE != 0 and bins < len(folded_phase):
            bin_means, bin_edges, binnumber = stats.binned_statistic(folded_phase, folded_y, statistic='mean',
                                                                     bins=bins)
            bin_stds, _, _ = stats.binned_statistic(folded_phase, folded_y, statistic='std', bins=bins)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width / 2
            bin_size = int(folded_plot_range * 2 / bins * transit_result.period * 24 * 60)
            bin_means_data_mask = np.isnan(bin_means)
            ax2.errorbar(bin_centers[~bin_means_data_mask], bin_means[~bin_means_data_mask],
                         yerr=bin_stds[~bin_means_data_mask] / 2, xerr=bin_width / 2, marker='o', markersize=4,
                         color='darkorange', alpha=1, linestyle='none', label='Bin size: ' + str(bin_size) + "m")
            ax2.legend(loc="upper right")
    ax3 = plt.gca()
    ax3.axvline(transit_result.period, alpha=0.4, lw=3)
    plt.xlim(np.min(tls_results.periods), np.max(tls_results.periods))
    for n in range(2, 10):
        ax3.axvline(n * tls_results.period, alpha=0.4, lw=1, linestyle="dashed")
        ax3.axvline(tls_results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    ax3.set(xlabel='Period (days)', ylabel='SDE')
    ax3.plot(tls_results.periods, tls_results.power, color='black', lw=0.5)
    ax3.set_xlim(0., max(tls_results.periods))
    plt.savefig(plot_dir + file, bbox_inches='tight', dpi=200)
    fig.clf()
    plt.close(fig)
