
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import pandas as pd
from utils import moving_average

def plot_xy(x: list[float], y: list[float], x_axis: str=None, y_axis: str=None, moving_average_window: int | float=0.01, max_curve=True, max_line=False, scatter=True, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, x_grid_interval: float=None, y_grid_interval: float=None, loc: str="lower right", linewidth: float=1.0, save_only: bool=False, top_ps: list[float]=None, output_dir: str=None, title: str=None, logger=None):
    top_ps = top_ps or []
    
    if x_axis is None:
        x_axis = "x"
    if y_axis is None:
        y_axis = "y"
    if title is None:
        title = ""

    plt.clf()
    if scatter:
        plt.scatter(x, y, s=500/len(x), alpha=0.2)
    plt.title(title)
    
    plt.xlabel(x_axis)
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim(0,x[-1])

    plt.ylabel(y_axis)
    if ylim is not None:
        plt.ylim(ylim)
    
    if x_grid_interval is not None and x_grid_interval > 0:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(base=x_grid_interval))
        ax.grid(axis="x", which="major")
        
    if y_grid_interval is not None and y_grid_interval > 0:
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(base=y_grid_interval))
        ax.grid(axis="y", which="major")
    else:
        plt.grid(axis="y")
         
    
    if moving_average_window is not None:
        label = f"moving average ({moving_average_window})"
        y_ma = moving_average(y, moving_average_window)
        plt.plot(x, y_ma, label=label, linewidth=linewidth)
        if top_ps is not None:
                for p in top_ps:
                    if 0 < p < 1:
                        y_ma_top = moving_average(y, moving_average_window, top_p=p)
                        label_top = f"top-{int(p*100)}% moving average"
                        plt.plot(x, y_ma_top, label=label_top, linewidth=linewidth)
                    else:
                        if logger is not None:
                            logger.warning(f"Ignored top_p={p} in top_ps (must be in (0,1))")
                        else:
                            print(f"Ignored top_p={p} in top_ps (must be in (0,1))")

    if max_curve:
        y_max_curve = np.maximum.accumulate(y)
        plt.plot(x, y_max_curve, label='max', linestyle='--', linewidth=linewidth)

    if max_line:
        max(y)
        y_max = np.max(y)
        plt.axhline(y=y_max, color='red', linestyle='--', label=f'y={y_max:.5f}', linewidth=linewidth)
    
    plt.legend(loc=loc)
    if output_dir is not None:
        plt.savefig(output_dir + title + "_" + y_axis + "_by_" + x_axis + ".png")
    plt.close() if save_only else plt.show()
    
def plot_csv(csv_path: str, target: str="reward", moving_average_window: int | float=0.01, max_curve=True, max_line=False, scatter=True, xlim: tuple[float, float]=None, ylim: tuple[float, float]=None, x_grid_interval: float=None, y_grid_interval: float=None, loc: str="lower right", linewidth: float=1.0, save_only: bool=False, top_ps: list[float]=None, output_dir: str=None, title: str=None, logger=None):
    df = pd.read_csv(csv_path)

    if "order" not in df.columns:
        raise ValueError("No 'order' column in csv")
    if target not in df.columns:
        raise ValueError(f"No '{target}' column in csv.")

    x = df["order"].tolist()
    y = df[target].tolist()

    plot_xy(x, y, x_axis="order", y_axis=target, moving_average_window=moving_average_window, max_curve=max_curve, max_line=max_line, scatter=scatter, xlim=xlim, ylim=ylim, x_grid_interval=x_grid_interval, y_grid_interval=y_grid_interval, loc=loc, linewidth=linewidth, save_only=save_only, top_ps=top_ps, output_dir=output_dir, title=title, logger=logger)