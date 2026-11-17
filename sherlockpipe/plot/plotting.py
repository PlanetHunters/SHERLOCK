import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy.ndimage import median_filter


def plot_system_architecture(sistemas, temperaturas_estrellas, tamaños_estrellas, periodos_orbitales, tamaños_planetas, candidatos, save_dir) -> None:
    """Plot a multi-planet system architecture diagram.

    Produces a scatter plot showing stars and their confirmed/candidate planets
    ordered by stellar temperature, with planet size shown by marker size and
    stellar temperature shown by color.

    Parameters
    ----------
    sistemas : list of str
        List of system names.
    temperaturas_estrellas : list of float
        List of stellar temperatures in Kelvin.
    tamaños_estrellas : list of float
        List of stellar radii relative to the Sun.
    periodos_orbitales : list of list of float
        Nested list of orbital periods in days per system.
    tamaños_planetas : list of list of float
        Nested list of planet radii in Earth units per system.
    candidatos : dict
        Dictionary mapping system names to candidate planets with keys
        ``'radios'`` and ``'periodos'``.
    save_dir : str
        Directory where the output plot will be saved.
    """
    sistemas_info = list(zip(sistemas, periodos_orbitales, tamaños_planetas, temperaturas_estrellas, tamaños_estrellas))
    sistemas_info.sort(key=lambda x: x[3])
    sistemas_ordenados, periodos_orbitales_ordenados, tamaños_planetas_ordenados, temperaturas_estrellas_ordenadas, tamaños_estrellas_ordenados = zip(*sistemas_info)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('xkcd:white')
    separacion = 1.1
    ylim_superior = len(sistemas_ordenados) * separacion + 0.5
    def color_planeta(tamaño):
        """Map a planet radius to a color for the system architecture plot.

        Parameters
        ----------
        tamaño : float
            The planet radius in Earth units.

        Returns
        -------
        str
            ``'brown'`` for super-Earths (R < 2), ``'green'`` for sub-Neptunes
            (2 <= R <= 5), ``'blue'`` for larger planets (R > 5).
        """
        if tamaño < 2:
            return 'brown'
        elif 2 <= tamaño <= 5:
            return 'green'
        else:
            return 'blue'
    colors = [(1, 0, 0), (1, 1, 0)]  # R -> Y
    n_bins = 100  # Número de divisiones en el colormap
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    ax.set_ylim(-0.5, ylim_superior)
    ax.set_yticks(np.arange(len(sistemas_ordenados)) * separacion)
    ax.set_yticklabels(sistemas_ordenados)
    ax.set_xlabel('Orbital Period (day)', fontsize=14)
    # ax.set_ylabel('Planetary System')
    ax.tick_params(axis='x', labelsize=12)  # Ajusta el tamaño de los ticks del eje X
    ax.tick_params(axis='y', labelsize=12)  # Ajusta el tamaño de los ticks del eje Y
    max_periodo_confirmados = max([max(periodos) for periodos in periodos_orbitales])
    periodos_candidatos = [item for sublist in [c['periodos'] for c in candidatos.values()] for item in sublist]
    max_periodo_candidatos = max(periodos_candidatos) if periodos_candidatos else 0
    max_periodo_general = max(max_periodo_confirmados, max_periodo_candidatos)
    ticks_x = np.arange(1, max_periodo_general + 1, 2)
    ax.set_xticks(ticks_x)
    # Ajustar los límites del eje X para asegurarse de que todos los ticks sean visibles
    ax.set_xlim(-1.5, max_periodo_general + 1)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5,zorder=1)
    ax.grid(False, axis='x')
    norm = plt.Normalize(3900, 6000)
    for i, sistema in enumerate(sistemas_ordenados):
        y = i * separacion  # Ajustar la separación entre sistemas
        for j, periodo in enumerate(periodos_orbitales_ordenados[i]):
            tamaño_planeta = tamaños_planetas_ordenados[i][j]
            color = color_planeta(tamaño_planeta)
            ax.scatter(periodo, y, s=tamaño_planeta*35, c=color, marker='o', alpha=1.0, zorder=2)  # Sin transparencia para confirmados
        tamaño_estrella = tamaños_estrellas_ordenados[i] * 20 * 100  # Escala de tamaño de estrella\n",
        #norm = plt.Normalize(3900, 6000)
        color_estrella = cm(norm(temperaturas_estrellas_ordenadas[i]))
        ax.scatter(-0.25, y, s=tamaño_estrella, c=color_estrella, marker='o', alpha=1.0,zorder=2)  # Sin transparencia para estrellas
        # Planetas candidatos (rellenos con borde negro y transparencia)
        if sistema in candidatos:
            for k, periodo_candidato in enumerate(candidatos[sistema]['periodos']):
                tamaño_candidato = candidatos[sistema]['radios'][k]
                color_candidato = color_planeta(tamaño_candidato)
                ax.scatter(periodo_candidato, y, s=tamaño_candidato*35, c=color_candidato, edgecolors='black', marker='o', alpha=0.5, linewidth=2,zorder=2)  # Añadir transparencia
    # Crear colorbar para las temperaturas de las estrellas con el colormap personalizado
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax,pad=0.02)
    cbar.set_label('Stellar Temperature (K)', fontsize=14)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', markersize=10, label=r'≤ 2.0 R$_{\\oplus}$'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label=r'> 2.0 and ≤ 5.0 R$_{\\oplus}$'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=r'> 5.0 R$_{\\oplus}$')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.87, 0.98), ncol=1)
    plt.tight_layout()
    plt.savefig(save_dir, 'sistemas_planetarios.png', dpi=200)


def plot_stability(masses, eccentricities, megnos, save_dir):
    """Plot a MEGNO stability grid over mass-eccentricity parameter space.

    Parameters
    ----------
    masses : ndarray
        Array of planet mass values in Earth masses.
    eccentricities : ndarray
        Array of orbital eccentricity values.
    megnos : ndarray
        Array of MEGNO values corresponding to each mass-eccentricity pair.
    save_dir : str
        Directory where the output plot will be saved.
    """
    unique_masses = np.unique(masses)
    unique_eccs = np.unique(eccentricities)
    results = np.full((len(unique_masses), len(unique_eccs)), 10.0)
    for index, mass in enumerate(masses):
        mass_index = np.argwhere(unique_masses == mass).flatten()[0]
        ecc = eccentricities[index]
        ecc_index = np.argwhere(unique_eccs == ecc).flatten()[0]
        results[mass_index, ecc_index] = megnos[index]
    ecc_index = np.argwhere(unique_eccs == ecc).flatten()[0]
    results = np.nan_to_num(results)
    n = len(results)
    pixels = len(results[1])
    Ngrid = len(np.unique(masses))
    ecc1max = max(eccentricities)
    ecc1min = min(eccentricities)
    par_m1 = np.linspace(ecc1min, ecc1max, Ngrid)
    m2max = max(masses)
    m2min = min(masses)
    par_m2 = np.linspace(m2min, m2max, Ngrid)
    ax = plt.subplot(111)
    extent = [min(par_m2), max(par_m2), min(par_m1), max(par_m1)] #x,y
    ax.set_xlim(extent[0], extent[1])
    ax.set_xlabel("Mass $M_\\oplus$")
    ax.set_ylim(extent[2], extent[3])
    ax.set_ylabel("Eccentricity")
    im = ax.imshow(np.transpose(results), interpolation="none", vmin=1., vmax=5, cmap="plasma", origin="lower", aspect='auto', extent=extent)
    cb = plt.colorbar(im, ax=ax, ticks=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    cb.ax.set_yticklabels(["1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"])
    cb.set_label("MEGNO $\\langle Y \\rangle$")
    ecc = np.linspace(0, 0.356, 50)
    ebb = 0.27 - (1.50*ecc)
    plt.plot(ecc, ebb, '-', color="black")
    plt.savefig(save_dir + '/stability_megno.png', bbox_inches='tight', dpi=200)

def plot_stability_by_ecc(results_dir: str, smooth_size: int = 3, limits=[[4.95, 'red'], [4.0, 'white']]):
    """Plot a MEGNO stability grid over eccentricity-pair parameter space.

    Reads a CSV file of stability results and plots the MEGNO values as a
    function of two eccentricity parameters with configurable contour levels.

    Parameters
    ----------
    results_dir : str
        Path to the CSV file containing stability results with columns
        ``'eccentricities'`` and ``'megno'``.
    smooth_size : int, optional
        Size of the median filter window for smoothing, by default 3.
    limits : list of list, optional
        Contour levels and colors to draw on the plot. Each entry is
        ``[value, color]``, by default ``[[4.95, 'red'], [4.0, 'white']]``.
    """
    df = pd.read_csv(results_dir)
    eccentricity_pairs = []
    megno_values = []
    for _, row in df.iterrows():
        e1, e2 = map(float, row['eccentricities'].split(','))
        eccentricity_pairs.append((e1, e2))
        megno_values.append(float(row['megno']))
    eccentricity_pairs = np.array(eccentricity_pairs)
    megno_values = np.array(megno_values)
    e1_unique = np.sort(np.unique(eccentricity_pairs[:, 0]))
    e2_unique = np.sort(np.unique(eccentricity_pairs[:, 1]))
    megno_grid = np.zeros((len(e1_unique), len(e2_unique)))
    megno_grid[:] = np.nan
    for (e1, e2), megno in zip(eccentricity_pairs, megno_values):
        i = np.where(e1_unique == e1)[0][0]
        j = np.where(e2_unique == e2)[0][0]
        megno_grid[i, j] = megno
    megno_grid = median_filter(megno_grid, size=smooth_size)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(megno_grid,
                   origin='lower',
                   extent=[min(e2_unique), max(e2_unique), min(e1_unique), max(e1_unique)],
                   aspect='auto',
                   cmap='viridis_r')
    for limit in limits:
        contour = ax.contour(
            e2_unique,  # X grid
            e1_unique,  # Y grid
            megno_grid,  # Z grid
            levels=[limit[0]],  # Contour at MEGNO = 5
            colors=limit[1],
            linewidths=2,
            linestyles='--'
        )
    ax.clabel(contour, inline=True, fontsize=15, fmt='%.2f')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MEGNO Value', rotation=270, labelpad=15)

# megno_df = pd.read_csv("/home/martin/Downloads/sherlock/sherlock_confusion_matrix/cps/done/TIC461239485_all/stability_stability/stability_megno.csv")
# megno_df[['mass1', 'mass2']] = megno_df['masses'].str.split(',', expand=True)
# megno_df['mass2'] = megno_df['mass2'].astype(float)
# megno_df[['ecc1', 'ecc2']] = megno_df['eccentricities'].str.split(',', expand=True)
# megno_df['ecc2'] = megno_df['ecc2'].astype(float)
# megno_df['ecc1'] = megno_df['ecc1'].astype(float)
# megno_df = megno_df.groupby(['mass2', 'ecc2'])['megno'].mean().reset_index()
# megno_df.sort_values(by=['mass2', 'ecc2'], ascending=True, inplace=True)
# plot_stability(megno_df['mass2'].to_numpy(), megno_df['ecc2'].to_numpy(), megno_df['megno'].to_numpy(),'./')

# sistemas = ['TOI-2441', 'WASP-16', 'HAT-P-26', 'HAT-P-27']
# periodos_orbitales = [[0.8], [3.2], [4.2], [3.03]]
# tamaños_planetas = [[1.7], [11.8], [7.1], [11]]
# temperaturas_estrellas = [4099, 5717, 5062, 5316]
# tamaños_estrellas = [0.72, 1.07, 0.86, 0.86]
# candidatos = {
#     'TOI-2441': {'radios': [3.0], 'periodos': [18.7]},
#     'WASP-16': {'radios': [2.0], 'periodos': [10.4]},
#     'HAT-P-26': {'radios': [2.1], 'periodos': [6.6]},
#     'HAT-P-27': {'radios': [3.03], 'periodos': [1.2]}
# }
# plot_system_architecture(sistemas, temperaturas_estrellas, tamaños_estrellas, periodos_orbitales, tamaños_planetas, candidatos, './')


