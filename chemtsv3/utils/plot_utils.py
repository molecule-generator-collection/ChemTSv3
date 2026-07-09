
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import MultipleLocator
import pandas as pd
import warnings
from chemtsv3.utils import moving_average
from chemtsv3.utils.logging_utils import warn_with_logger

PLOT_STYLE = {
    "figure.figsize": (9, 6),
    "axes.titlesize": 28,
    "axes.labelsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 21,
    "grid.alpha": 0.8,
    "figure.dpi": 120,
    "savefig.dpi": 200,
}

DEFAULT_LINEWIDTH = 2.0
PLOT_AXES_RECT = (0.18, 0.18, 0.74, 0.76)

AXIS_LABELS = {
    "generation_order": "order",
}

def axis_label(name: str) -> str:
    return AXIS_LABELS.get(name, name)

def make_plot_figure():
    fig = plt.figure(figsize=PLOT_STYLE["figure.figsize"])
    ax = fig.add_axes(PLOT_AXES_RECT)
    return fig, ax

def _corrcoef(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()

    if mask.sum() >= 2 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
        return np.corrcoef(x[mask], y[mask])[0, 1]
    return np.nan

def corr_heatmap(x, y, cmap: str="coolwarm", **kwargs):
    r = _corrcoef(x, y)
    ax = plt.gca()
    cmap_obj = plt.get_cmap(cmap)
    norm = plt.Normalize(-1, 1)
    if np.isfinite(r):
        ax.set_facecolor(cmap_obj(norm(r)))
        text = f"{r:.2f}"
    else:
        ax.set_facecolor("#f2f2f2")
        text = "nan"
    ax.annotate(
        text,
        (0.5, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=24,
    )

def plot_cross_plot(data: str | pd.DataFrame, target: list[str]=None, columns: list[str]=None, label_dict: dict[str, str]=None, output_path: str=None, output_dir: str=None, filename: str="cross_plot.png", bins: int=25, scatter_size: float=20, cmap: str="coolwarm", save_only: bool=True, logger=None):
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data)

    target = target if target is not None else columns
    if target is None:
        ignored_columns = {"order", "time", "key", "generation_order"}
        target = []
        for col in df.columns:
            if col in ignored_columns:
                continue
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            if numeric_col.notna().any():
                target.append(col)
    else:
        missing_columns = [col for col in target if col not in df.columns]
        if missing_columns:
            warn_with_logger(f"Ignored missing cross plot columns: {missing_columns}", logger=logger)
        target = [col for col in target if col in df.columns]

    if len(target) < 2:
        warn_with_logger("Skipped cross plot because fewer than two numeric columns were available.", logger=logger)
        return None

    df_plot = df[target].apply(pd.to_numeric, errors="coerce")
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    non_numeric_columns = [col for col in target if df_plot[col].notna().sum() == 0]
    if non_numeric_columns:
        warn_with_logger(f"Ignored non-numeric cross plot columns: {non_numeric_columns}", logger=logger)
        df_plot = df_plot.drop(columns=non_numeric_columns)
        target = [col for col in target if col not in non_numeric_columns]

    if len(target) < 2:
        warn_with_logger("Skipped cross plot because fewer than two numeric columns were available.", logger=logger)
        return None

    label_dict = label_dict or {}
    df_plot = df_plot.rename(columns={col: label_dict.get(col, axis_label(col)) for col in target})

    if output_path is None:
        if output_dir is None:
            raise ValueError("Specify either output_path or output_dir.")
        output_path = os.path.join(output_dir, filename)

    with plt.rc_context({"figure.dpi": 120, "savefig.dpi": 200}):
        n_cols = len(df_plot.columns)
        fig_size = max(2.4 * n_cols, 6)
        fig, axes = plt.subplots(n_cols, n_cols, figsize=(fig_size + 1.2, fig_size), squeeze=False)
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(-1, 1)

        for i, y_col in enumerate(df_plot.columns):
            for j, x_col in enumerate(df_plot.columns):
                ax = axes[i, j]
                x = df_plot[x_col]
                y = df_plot[y_col]

                if i == j:
                    values = x.dropna()
                    ax.hist(values, bins=bins, color="#4c72b0", edgecolor="white")
                    ax.tick_params(axis="y", left=False, labelleft=False)
                elif i > j:
                    mask = x.notna() & y.notna()
                    ax.scatter(x[mask], y[mask], s=scatter_size, alpha=0.7, linewidths=0)
                else:
                    r = _corrcoef(x, y)
                    if np.isfinite(r):
                        ax.set_facecolor(cmap_obj(norm(r)))
                        text = f"{r:.2f}"
                    else:
                        ax.set_facecolor("#f2f2f2")
                        text = "nan"
                    ax.annotate(
                        text,
                        (0.5, 0.5),
                        xycoords="axes fraction",
                        ha="center",
                        va="center",
                        fontsize=24,
                    )
                    ax.tick_params(
                        axis="both",
                        bottom=False,
                        left=False,
                        labelbottom=False,
                        labelleft=False,
                    )

                if i == n_cols - 1:
                    ax.set_xlabel(x_col, fontsize=16)
                else:
                    ax.tick_params(axis="x", labelbottom=False)
                if j == 0:
                    ax.set_ylabel(y_col, fontsize=16)
                else:
                    ax.tick_params(axis="y", labelleft=False)
                ax.set_box_aspect(1)

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label="Correlation")

        output_dirname = os.path.dirname(output_path)
        if output_dirname:
            os.makedirs(output_dirname, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig) if save_only else plt.show()

    return output_path

def plot_xy(x: list[float], y: list[float], x_axis: str=None, y_axis: str=None, moving_average_window: int | float=0.01, max_curve=True, max_line=False, scatter=True, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, x_grid_interval: float=None, y_grid_interval: float=None, loc: str="lower right", linewidth: float=DEFAULT_LINEWIDTH, save_only: bool=False, top_ps: list[float]=None, output_dir: str=None, title: str=None, logger=None):
    top_ps = top_ps or []
    
    with plt.rc_context(PLOT_STYLE):
        if x_axis is None:
            x_axis = "x"
        if y_axis is None:
            y_axis = "y"
        if title is None:
            title = ""

        fig, ax = make_plot_figure()
        if scatter:
            ax.scatter(x, y, s=500/len(x), alpha=0.2)
        
        ax.set_xlabel(axis_label(x_axis))
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(0,x[-1])

        ax.set_ylabel(axis_label(y_axis))
        if ylim is not None:
            ax.set_ylim(ylim)
        
        if x_grid_interval is not None and x_grid_interval > 0:
            ax.xaxis.set_major_locator(MultipleLocator(base=x_grid_interval))
            ax.grid(axis="x", which="major")
            
        if y_grid_interval is not None and y_grid_interval > 0:
            ax.yaxis.set_major_locator(MultipleLocator(base=y_grid_interval))
            ax.grid(axis="y", which="major")
        else:
            ax.grid(axis="y")
            
        if moving_average_window is not None:
            label = f"moving average ({moving_average_window})"
            y_ma = moving_average(y, moving_average_window)
            ax.plot(x, y_ma, label=label, linewidth=linewidth)
            if top_ps is not None:
                for p in top_ps:
                    if 0 < p < 1:
                        y_ma_top = moving_average(y, moving_average_window, top_p=p)
                        label_top = f"top-{int(p*100)}% moving average"
                        ax.plot(x, y_ma_top, label=label_top, linewidth=linewidth)
                    else:
                        if logger is not None:
                            logger.warning(f"Ignored top_p={p} in top_ps (must be in (0,1))")
                        else:
                            print(f"Ignored top_p={p} in top_ps (must be in (0,1))")

        if max_curve:
            y_max_curve = np.maximum.accumulate(y)
            ax.plot(x, y_max_curve, label='max', linestyle='--', linewidth=linewidth)

        if max_line:
            max(y)
            y_max = np.max(y)
            ax.axhline(y=y_max, color='red', linestyle='--', label=f'y={y_max:.5f}', linewidth=linewidth)
        
        ax.legend(loc=loc)
        if output_dir is not None:
            fig.savefig(output_dir + title + "_" + y_axis + "_by_" + x_axis + ".png")
        plt.close(fig) if save_only else plt.show()

def plot_csv(csv_path: str, target: str="reward", moving_average_window: int | float=0.01, max_curve=True, max_line=False, scatter=True, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, x_grid_interval: float=None, y_grid_interval: float=None, loc: str="lower right", linewidth: float=DEFAULT_LINEWIDTH, save_only: bool=False, top_ps: list[float]=None, output_dir: str=None, title: str=None, logger=None, x_axis_type: str="order"):
    df = pd.read_csv(csv_path)

    if x_axis_type not in ("order", "time"):
        message = f"Unsupported x_axis_type='{x_axis_type}'. Falling back to 'order'."
        if logger is not None:
            logger.warning(message)
        else:
            warnings.warn(message)
        x_axis_type = "order"

    if x_axis_type not in df.columns:
        raise ValueError(f"No '{x_axis_type}' column in csv")
    if target not in df.columns:
        raise ValueError(f"No '{target}' column in csv.")

    x = df[x_axis_type].tolist()
    y = df[target].tolist()

    plot_xy(x, y, x_axis=x_axis_type, y_axis=target, moving_average_window=moving_average_window, max_curve=max_curve, max_line=max_line, scatter=scatter, xlim=xlim, ylim=ylim, x_grid_interval=x_grid_interval, y_grid_interval=y_grid_interval, loc=loc, linewidth=linewidth, save_only=save_only, top_ps=top_ps, output_dir=output_dir, title=title, logger=logger)
