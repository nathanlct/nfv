import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# good cmaps: jet, plasma, RdYlGn_r


def plot_heatmap_on_ax(
    data_TX,
    ax,
    title=None,
    xlabel=True,
    ylabel=True,
    cbar=True,
    fontsize=None,
    xmax=None,
    tmax=None,
    vmin=0.0,
    vmax=1.0,
    cmap="jet",
    xticks=None,
    yticks=None,
):
    if (x := data_TX.min()) < vmin - 1e-5:
        print(f"WARNING: data being plotted has values < vmin (min = {x}, vmin = {vmin})")
    if (x := data_TX.max()) > vmax + 1e-5:
        print(f"WARNING: data being plotted has values > vmax (max = {x}, vmax = {vmax})")
    tmin, xmin = 0, 0
    tmax = tmax or data_TX.shape[0]
    xmax = xmax or data_TX.shape[1]
    im = ax.imshow(
        data_TX,
        origin="lower",
        interpolation="none",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=[xmin, xmax, tmin, tmax],
    )
    if cbar:
        cbar_obj = plt.colorbar(im, ax=ax)
        cbar_obj.set_label(r"Density ($\rho$)", fontsize=fontsize)
    ax.set_xticks(xticks or [])
    ax.set_yticks(yticks or [])
    if xlabel:
        ax.set_xlabel("Space (x)", fontsize=fontsize)
    if ylabel:
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel, fontsize=fontsize)
        else:
            ax.set_ylabel("Time (t)", fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    return im


def plot_heatmap(
    data, path=None, return_fig=False, width=4, height=3, dpi=300, title_col=None, title_row=None, transpose=False, tight=False, **kwargs
):
    """
    Plot heatmap(s) from data input.

    Parameters:
      - data: array-like; supports 1D (converted to 2D), 2D (single heatmap),
              3D (list/row of heatmaps), or 4D (grid of heatmaps).
      - path: if provided, the figure is saved to this path, otherwise it is shown (unless return_fig is True).
      - return_fig: if True, returns the matplotlib figure object.
      - width, height, dpi: control figure size (per heatmap) and resolution.
      - kwargs: additional arguments to pass to plot_heatmap_on_ax.
    """
    # Ensure data is a NumPy array
    data = np.asarray(data).squeeze()

    if transpose:
        if data.ndim == 4:
            data = np.swapaxes(data, 0, 1)
        elif data.ndim == 3:
            # go from single row to single column
            data = np.expand_dims(data, axis=1)

    # If data is 1D, treat it as a single-row heatmap
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim == 2:
        # Single heatmap
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        plot_heatmap_on_ax(data, ax, **kwargs)
    elif data.ndim == 3:
        # List of heatmaps on one row
        ncols = data.shape[0]
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * width, height), dpi=dpi)
        for j in range(ncols):
            title = title_col[j] if title_col is not None else None
            plot_heatmap_on_ax(data[j], axes[j], title=title, **kwargs)
    elif data.ndim == 4:
        # Grid of heatmaps
        nrows, ncols = data.shape[0], data.shape[1]
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * height), dpi=dpi)
        for i in range(nrows):
            for j in range(ncols):
                kwargs_local = kwargs.copy()
                title = title_col[j] if title_col is not None and i == 0 else None
                if nrows == 1:
                    ax = axes[j]
                elif ncols == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
                if title_row is not None and j == 0:
                    kwargs_local["ylabel"] = title_row[i]
                plot_heatmap_on_ax(data[i][j], ax, title=title, **kwargs_local)
    else:
        raise ValueError(f"Unsupported data dimension: {data.shape}")

    # finish up
    plt.tight_layout()
    if path is not None:
        if tight:
            plt.savefig(path, bbox_inches="tight", pad_inches=0)
        else:
            plt.savefig(path)
    if return_fig:
        return fig
    if path is None:
        plt.show()
    plt.close(fig)


def plot_on_ax(data, ax, plot_fn, legend_label=None, **kwargs):
    ax.plot(plot_fn(data), label=legend_label, linewidth=0.5)


def plot_agg(data, plot_fn, title=None, path=None, return_fig=False, width=4, height=3, dpi=300, legend=None, transpose=False, **kwargs):
    # Ensure data is a NumPy array
    data = np.asarray(data)

    if data.ndim == 2:
        # Single plot
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        plot_on_ax(data, ax, plot_fn, **kwargs)
        ax.legend()
        ax.grid()
    elif data.ndim == 3:
        # Aggregate all row on one plot
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        for i in range(data.shape[0]):
            legend_label = legend[i] if legend is not None else None
            plot_on_ax(data[i], ax, plot_fn, legend_label=legend_label, **kwargs)
        ax.legend()
        ax.grid()
    elif data.ndim == 4:
        # Aggregate by columns (resulting in a row plot)
        if transpose:
            data = np.swapaxes(data, 0, 1)
        nrows, ncols = data.shape[0], data.shape[1]  # flow, scheme
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * width, height), dpi=dpi)
        for col in range(ncols):
            ax = axes[col] if ncols > 1 else axes
            for row in range(nrows):
                legend_label = legend[row] if legend is not None else None
                plot_on_ax(data[row][col], ax, plot_fn, legend_label=legend_label if col == 0 else None, **kwargs)
            # axes[col].legend()
            ax.grid()
            if title:
                ax.set_title(title[col])
        fig.legend(loc="upper center", ncol=5, fontsize=10)
    else:
        raise ValueError(f"Unsupported data dimension: {data.shape}")

    # finish up
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    if path is not None:
        plt.savefig(path)
    if return_fig:
        return fig
    if path is None:
        plt.show()
    plt.close(fig)


def plot_data_distribution(data_BTX):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), dpi=300)
    coords = []
    coords_ic = []
    B, T, X = data_BTX.shape
    for b in range(B):
        for x in range(X - 1):
            coords_ic.append((data_BTX[b, 0, x], data_BTX[b, 0, x + 1]))
        for t in range(T):
            for x in range(X - 1):
                coords.append((data_BTX[b, t, x], data_BTX[b, t, x + 1]))
    axes[0].hist2d(*zip(*coords), bins=100, cmap="plasma", norm=matplotlib.colors.LogNorm())
    axes[1].hist2d(*zip(*coords_ic), bins=100, cmap="plasma", norm=matplotlib.colors.LogNorm())
    plt.savefig("data_distribution.png")
