import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_time_taken_histogram(df, bins=None, figsize=(8, 4), xlim=None):
    col = "Time taken"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataframe.")
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    series /= 60  # converting seconds to minutes
    if bins is None: # default to 2-minute bins from 0 to max observed (rounded up)
        max_min = float(series.max()) if len(series) else 0.0
        upper = max(2, int(np.ceil(max_min / 2.0) * 2) + 2)
        bins = np.arange(0, upper, 2)
    plt.figure(figsize=figsize)
    sns.histplot(series, kde=False, bins=bins)
    plt.title(col)
    plt.xlabel(col + " (minutes)")
    plt.ylabel("Count")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.show()
    print(series.describe())


def plot_total_approvals_histogram(df, bins=None, figsize=(8, 4), xlim=None):
    col = "Total approvals"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataframe.")
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if bins is None:    # default integer-width bins centered on whole numbers

        max_val = int(np.ceil(series.max())) if len(series) else 1
        bins = np.arange(-0.5, max_val + 1.5, 1)
    plt.figure(figsize=figsize)
    sns.histplot(series, kde=False, bins=bins)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Count")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.show()
    print(series.describe())


def plot_age_histogram(df, bins=np.arange(16, 100, 1), figsize=(6, 4.5), save_path=None):
    col = "Age"
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataframe.")
    series = pd.to_numeric(df[col], errors="coerce").dropna()

    plt.figure(figsize=figsize)
    sns.histplot(series, kde=False, bins=bins, color="steelblue", alpha=1.0)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(series.describe())


def plot_numeric_boxplot(
    df,
    col_name,
    transform=None,
    title=None,
    x_label=None,
    figsize=(6, 2.5),
    xlim=None,
    show_stats=True,
):
    """
    Generic horizontal boxplot for a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    col_name : str
        Column to plot.
    transform : callable, optional
        Function applied to the numeric Series before plotting (e.g., lambda s: s / 60).
    title : str, optional
        Plot title. Defaults to col_name.
    x_label : str, optional
        X-axis label. Defaults to col_name.
    figsize : tuple, default (8, 2.5)
        Figure size.
    xlim : tuple, optional
        X-axis limits, e.g., (0, 120).
    show_stats : bool, default True
        Whether to print Series.describe().
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in dataframe.")

    series = pd.to_numeric(df[col_name], errors="coerce").dropna()
    if transform is not None:
        series = transform(series)
        # Keep only finite values after transform.
        series = series.replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=figsize)
    sns.boxplot(x=series, orient="h")
    plt.title(title or col_name)
    plt.xlabel(x_label or col_name)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.show()

    if show_stats:
        print(series.describe())


def plot_categorical_counts(
    df,
    col_name,
    max_categories=20,
    include_missing=True,
    group_other=True,
    auto_orient=True,
    orient="v",
    label_counts=True,
    sort_desc=True,
    ax=None,
    palette=None,
    save_path=None,
):
    """
    Plot count distribution for a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    col_name : str
        Column to plot.
    max_categories : int, default 20
        Maximum number of categories to show before grouping tail into "Other".
    include_missing : bool, default True
        If True, missing values are shown as a "Missing" category.
        If False, missing values are dropped.
    group_other : bool, default True
        If True and categories exceed max_categories, groups remainder as "Other".
    auto_orient : bool, default True
        If True, automatically switches to horizontal bars for many categories.
    orient : {"v", "h"}, default "v"
        Orientation used when auto_orient=False.
    label_counts : bool, default True
        Whether to annotate each bar with count.
    sort_desc : bool, default True
        Whether to sort categories by descending count.
    ax : matplotlib axis, optional
        Existing axis to plot on. If None, a new figure/axis is created.

    Returns
    -------
    (ax, counts) : tuple
        matplotlib axis and final counts Series used for plotting.
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in dataframe.")

    series = df[col_name].copy()
    if include_missing:
        series = series.fillna("Missing")
    else:
        series = series.dropna()

    counts = series.value_counts(dropna=False)
    if not sort_desc:
        counts = counts.sort_index()

    if group_other and len(counts) > max_categories:
        top = counts.iloc[:max_categories]
        other = counts.iloc[max_categories:].sum()
        counts = pd.concat([top, pd.Series({"Other": other})])

    n_cats = len(counts)
    max_label_len = counts.index.astype(str).map(len).max() if n_cats > 0 else 0
    if auto_orient:
        # Use horizontal bars either when there are many categories or labels are very long.
        use_orient = "h" if (n_cats > 12 or max_label_len > 20) else "v"
    else:
        use_orient = orient

    created_fig = False
    if ax is None:
        if use_orient == "h":
            fig_h = max(5, min(0.35 * n_cats, 14))
            _, ax = plt.subplots(figsize=(7, fig_h))
        else:
            _, ax = plt.subplots(figsize=(6, 4.5))
        created_fig = True

    if use_orient == "h":
        if palette is not None:
            sns.barplot(x=counts.values, y=counts.index.astype(str), orient="h", ax=ax,
                        hue=counts.index.astype(str), palette=palette, legend=False)
        else:
            sns.barplot(x=counts.values, y=counts.index.astype(str), orient="h", ax=ax,
                        color="steelblue")
        x_max = counts.max()
        ax.set_xlim(0, x_max * 1.12 if x_max > 0 else 1)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        ax.set_title(col_name)

        if label_counts:
            for bar, value in zip(ax.patches, counts.values):
                ax.annotate(
                    f"{int(value)}",
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    ha="left",
                    va="center",
                    xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=8,
                    clip_on=False,
                )
    else:
        if palette is not None:
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                        hue=counts.index.astype(str), palette=palette, legend=False)
        else:
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax,
                        color="steelblue")
        y_max = counts.max()
        ax.set_ylim(0, y_max * 1.12 if y_max > 0 else 1)
        ax.set_ylabel("Count")
        ax.set_xlabel("")
        ax.set_title(col_name)
        # Right-align rotated tick labels so long category names do not drift/overlap.
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
            tick.set_rotation_mode("anchor")

        if label_counts:
            for bar, value in zip(ax.patches, counts.values):
                ax.annotate(
                    f"{int(value)}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    xytext=(0, 2),
                    textcoords="offset points",
                    fontsize=8,
                    clip_on=False,
                )

    if created_fig:
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    return ax, counts

def make_crosstab_tables(df, row_col, col_col, include_missing=True, margins=False, auto_print=True):
    """
    Create count / row% / col% crosstab tables for two categorical columns.

    Returns a dict with keys: "count", "row_pct", "col_pct".
    If auto_print=True, also prints and displays each table.
    """
    tmp = df[[row_col, col_col]].copy()

    if include_missing:
        tmp[row_col] = tmp[row_col].fillna("Missing")
        tmp[col_col] = tmp[col_col].fillna("Missing")
    else:
        tmp = tmp.dropna(subset=[row_col, col_col])

    tables = {
        "count": pd.crosstab(tmp[row_col], tmp[col_col], margins=margins),
        "row_pct": pd.crosstab(tmp[row_col], tmp[col_col], normalize="index") * 100,
        "col_pct": pd.crosstab(tmp[row_col], tmp[col_col], normalize="columns") * 100,
    }

    if auto_print:
        try:
            from IPython.display import display
        except ImportError:
            display = None

        print(f"Counts ({row_col} x {col_col}):")
        if display is not None:
            display(tables["count"])
        else:
            print(tables["count"])

        print(f"\nRow % (within each {row_col}):")
        if display is not None:
            display(tables["row_pct"].round(1))
        else:
            print(tables["row_pct"].round(1))

        print(f"\nColumn % (within each {col_col} category):")
        if display is not None:
            display(tables["col_pct"].round(1))
        else:
            print(tables["col_pct"].round(1))

    return tables


def plot_crosstab_heatmaps(
    tables,
    row_col,
    col_col,
    layout="combined",  # "combined" or "separate"
    cmap="Blues",
    linewidths=0.5,
):
    """
    Plot heatmaps for crosstab tables produced by make_crosstab_tables().

    layout:
      - "combined": one 1x3 figure
      - "separate": three standalone figures
    """
    expected = {"count", "row_pct", "col_pct"}
    assert expected.issubset(set(tables.keys())), f"tables must include keys {expected}"

    plot_specs = [
        ("count", "Counts", "d", "Count", None, None),
        ("row_pct", "Row % (within row)", ".1f", "Row %", 0, 100),
        ("col_pct", "Column % (within column)", ".1f", "Column %", 0, 100),
    ]

    if layout == "combined":
        fig, axes = plt.subplots(1, 3, figsize=(21, 5), constrained_layout=True)
        for ax, (key, title, fmt, cbar_label, vmin, vmax) in zip(axes, plot_specs):
            sns.heatmap(
                tables[key],
                annot=True,
                fmt=fmt,
                cmap=cmap,
                linewidths=linewidths,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": cbar_label},
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel(col_col)
            ax.set_ylabel(row_col if key == "count" else "")
            ax.tick_params(axis="x", rotation=35)
            ax.tick_params(axis="y", rotation=0)

        fig.suptitle(f"{row_col} × {col_col} Crosstabs", fontsize=13)
        plt.show()

    elif layout == "separate":
        for key, title, fmt, cbar_label, vmin, vmax in plot_specs:
            plt.figure(figsize=(8, 5))
            ax = sns.heatmap(
                tables[key],
                annot=True,
                fmt=fmt,
                cmap=cmap,
                linewidths=linewidths,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": cbar_label},
            )
            ax.set_title(f"{title}: {row_col} × {col_col}")
            ax.set_xlabel(col_col)
            ax.set_ylabel(row_col)
            plt.xticks(rotation=35, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()
    else:
        raise ValueError("layout must be either 'combined' or 'separate'")

