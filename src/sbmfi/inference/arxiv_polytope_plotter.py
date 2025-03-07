import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

def compute_convex_hull(df, x_col, y_col):
    """
    Computes the convex hull for a set of points in a DataFrame using the specified columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing arbitrary columns.
        x_col (str): Column name for the x-coordinate.
        y_col (str): Column name for the y-coordinate.

    Returns:
        pd.DataFrame: A DataFrame containing the convex hull points in order
                      (with the first point repeated at the end to close the polygon).
                      If fewer than 3 points are available, returns the available points.
    """
    points = df[[x_col, y_col]].dropna().values
    if len(points) < 3:
        return pd.DataFrame(points, columns=[x_col, y_col])

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_points = np.vstack([hull_points, hull_points[0]])

    return pd.DataFrame(hull_points, columns=[x_col, y_col])


def plot_dataframes(
        vertices, samples,
        n_sample_cdf=5000,
        points=None,
        x_col='x', y_col='y',
        x_label='X-axis', y_label='Y-axis',
        label1='Vertices Convex Hull',
        label2='Samples Convex Hull',
        label3='Samples KDE Density',
        label4='Additional Points',
        bw_method=None,
        bw_adjust=1,
        levels=15,
        font_dict=None,
        vertices_fill_color='#ffec58',
        samples_fill_color='#e02450',
        figsize=(10, 8),
        show_legend=True,
        legend_loc='best',
        s_points=25,
        points_color='#C41E3A',
        yticker=1.0,
        xticker=1.0,
):
    """
    Plots:
      - The convex hull of 'vertices' filled with a semi-transparent color (default red, alpha=0.2)
        and outlined with the same color (opaque and thicker).
      - The convex hull of 'samples' filled with a semi-transparent color (default blue, alpha=0.2)
        and outlined with the same color (opaque and thicker).
      - A KDE density estimate of 'samples' using Seaborn.kdeplot with the specified parameters (alpha=0.8).
      - Optionally, extra points from 'points' plotted as green markers.

    All font sizes for axes labels, tick labels, and legend are read from the dictionary `font_dict`
    using the .get() method, so that if a key is missing, a default value is used.

    New parameters:
      - show_legend (bool): If False, no legend is drawn. Default is True.
      - legend_loc (str): If a legend is drawn, this parameter specifies its location. Default is 'best'.

    Parameters:
        vertices (pd.DataFrame): DataFrame for the first convex hull.
        samples (pd.DataFrame): DataFrame for both the second convex hull and the KDE density.
        points (pd.DataFrame, optional): DataFrame containing extra points to plot.
        x_col (str): Column name for x-coordinates.
        y_col (str): Column name for y-coordinates.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        label1 (str): Legend label for the vertices convex hull.
        label2 (str): Legend label for the samples convex hull.
        label3 (str): Legend label for the KDE density.
        label4 (str): Legend label for the additional points.
        kde_threshold (float): (Informational) Density threshold below which the background appears white.
        bw_method: Bandwidth method to pass to kdeplot (can be a scalar, 'scott', 'silverman', or a callable).
        bw_adjust (float): Multiplicative factor that adjusts the bandwidth.
        levels (int): Number of contour levels for the KDE density plot.
        font_dict (dict, optional): A dictionary for font sizes. Keys used:
                                    'xlabel' (default 14), 'ylabel' (default 14),
                                    'ticks' (default 12), 'legend' (default 12).
        vertices_fill_color (str): Fill color for the vertices convex hull.
        samples_fill_color (str): Fill color for the samples convex hull.
        figsize (tuple): Figure size (width, height). Default is (10, 8).
        show_legend (bool): If True, the legend is displayed. Default is True.
        legend_loc (str): Legend location (if legend is displayed). Default is 'best'.

    Returns:
        tuple: (fig, ax) where fig is the Matplotlib Figure object and ax is the Axes object.
    """
    # Define default font sizes if none provided.
    if font_dict is None:
        font_dict = {'xlabel': 14, 'ylabel': 14, 'ticks': 12, 'legend': 12}

    # Create the figure and axis with the specified size.
    fig, ax = plt.subplots(figsize=figsize)

    # --- Compute and fill convex hull for vertices ---
    hull_vertices = compute_convex_hull(vertices, x_col, y_col)
    ax.fill(hull_vertices[x_col], hull_vertices[y_col],
            color=vertices_fill_color, alpha=0.2, label=label1)
    ax.plot(hull_vertices[x_col], hull_vertices[y_col],
            color=vertices_fill_color, lw=2, alpha=1)

    # --- Compute and fill convex hull for samples ---
    hull_samples = compute_convex_hull(samples, x_col, y_col)
    ax.fill(hull_samples[x_col], hull_samples[y_col],
            color=samples_fill_color, alpha=0.2, label=label2)
    ax.plot(hull_samples[x_col], hull_samples[y_col],
            color=samples_fill_color, lw=2, alpha=1)

    # Set the axis background to white.
    ax.set_facecolor('white')

    # Get a copy of the 'viridis' colormap using the recommended API.
    cmap = mpl.colormaps['viridis'].copy()
    cmap.set_under('white')

    # --- Plot KDE density from samples using Seaborn ---
    sns.kdeplot(
        data=samples[:n_sample_cdf],
        x=x_col,
        y=y_col,
        fill=True,
        cut=0,
        levels=levels,
        bw_adjust=bw_adjust,
        bw_method=bw_method,
        cmap=cmap,
        alpha=0.8,
        ax=ax
    )

    # Create a proxy patch for the KDE for the legend.
    proxy_kde = mpatches.Patch(color=mpl.colormaps['viridis'](0.6), label=label3)

    # --- Optionally plot extra points ---
    if points is not None:
        extra_points = points[[x_col, y_col]].dropna().values
        ax.scatter(extra_points[:, 0], extra_points[:, 1],
                   color=points_color, s=s_points, marker='o', label=label4)

    # Finalize the plot: set axes labels, tick parameters, and grid.
    ax.set_xlabel(x_label, fontsize=font_dict.get('xlabel', 14))
    ax.set_ylabel(y_label, fontsize=font_dict.get('ylabel', 14))
    ax.tick_params(axis='both', which='major', labelsize=font_dict.get('ticks', 12))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xticker))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(yticker))
    ax.grid(True, linestyle='--', alpha=0.7)

    # Process and create the legend if requested.
    if show_legend:
        handles, labels_ = ax.get_legend_handles_labels()
        if label3 not in labels_:
            handles.append(proxy_kde)
            labels_.append(label3)

        # Reorder the legend items in the desired order: [label1, label2, label3, label4].
        desired_order = [label1, label2, label3, label4]
        legend_dict = {lab: hnd for hnd, lab in zip(handles, labels_)}
        ordered_handles = []
        ordered_labels = []
        for lab in desired_order:
            if lab in legend_dict:
                ordered_handles.append(legend_dict[lab])
                ordered_labels.append(lab)

        ax.legend(ordered_handles, ordered_labels,
                  fontsize=font_dict.get('legend', 12),
                  loc=legend_loc)

    return fig, ax


def barycentric_to_cartesian(samples):
    """Convert barycentric coordinates to Cartesian coordinates."""
    A = torch.tensor([0.0, 0.0])
    B = torch.tensor([1.0, 0.0])
    C = torch.tensor([0.5, 0.8660254])  # approx sqrt(3)/2
    coords = samples[:, 0].unsqueeze(1) * A + samples[:, 1].unsqueeze(1) * B + samples[:, 2].unsqueeze(1) * C
    return coords
