import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import wntr
import os
import time
import tempfile

def _contrasting_text_color(color):
    """Return 'black' or 'white' based on background luminance for legibility."""
    r, g, b = mcolors.to_rgb(color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.55 else "white"


def _draw_graph_layers(
    G,
    pos,
    nodelist,
    node_colors,
    ax,
    title=None,
    node_size=900,
    node_edgecolor="#1a2634",
    node_linewidth=1.5,
    edge_color="#4a4a4a",
    edge_width=2.5,
    edge_alpha=1.0,
    show_node_labels=True,
    node_labels=None,
    font_size=13,
    font_color=None,
    show_edge_labels=True,
    show_edge_weights=True,
    edge_font_size=10,
    margins=0.18,
):
    """Internal rendering engine shared by all graph plotting functions."""
    nx.draw_networkx_edges(
        G, pos,
        width=edge_width,
        edge_color=edge_color,
        alpha=edge_alpha,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodelist,
        node_color=node_colors,
        node_size=node_size,
        edgecolors=node_edgecolor,
        linewidths=node_linewidth,
        ax=ax,
    )

    if show_node_labels:
        labels = node_labels or {n: f"$v_{{{n}}}$" for n in nodelist}
        if font_color is not None:
            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_color=font_color,
                font_weight="bold",
                font_size=font_size,
                ax=ax,
            )
        else:
            for node, color in zip(nodelist, node_colors):
                if node in labels:
                    nx.draw_networkx_labels(
                        G, pos,
                        labels={node: labels[node]},
                        font_color=_contrasting_text_color(color),
                        font_weight="bold",
                        font_size=font_size,
                        ax=ax,
                    )

    # Uses stored 'edge_id' so e1..e5 always match the slide numbering
    if show_edge_labels:
        edge_labels = {}
        for idx, (u, v, d) in enumerate(G.edges(data=True), start=1):
            eid = d.get("edge_id", f"e{idx}")
            num = eid[1:] if isinstance(eid, str) and eid.startswith("e") else idx
            if show_edge_weights and "weight" in d:
                edge_labels[(u, v)] = f"$e_{{{num}}}$ ($w_{{{num}}}={d['weight']}$)"
            else:
                edge_labels[(u, v)] = f"$e_{{{num}}}$"
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=edge_font_size,
            ax=ax,
        )

    if title:
        ax.set_title(title, fontsize=13, pad=10)
    ax.margins(margins)
    ax.axis("off")


def _resolve_oriented_edges(G, nodelist, B=None, edgelist=None):
    """Return list of (u, v, edge_attr_dict) aligned with the column order of B / y."""
    if edgelist is not None:
        resolved = []
        for idx, e in enumerate(edgelist, start=1):
            u, v = e[0], e[1]
            data = dict(G.edges[u, v]) if G.has_edge(u, v) else {}
            data.setdefault("edge_id", f"e{idx}")
            resolved.append((u, v, data))
        return resolved

    if B is not None:
        B_arr = np.asarray(B, dtype=float)
        if B_arr.shape[0] != len(nodelist):
            raise ValueError(f"B has {B_arr.shape[0]} rows, but nodelist has {len(nodelist)} nodes.")
        resolved = []
        for k in range(B_arr.shape[1]):
            col = B_arr[:, k]
            tails = np.where(col < 0)[0]
            heads = np.where(col > 0)[0]
            if len(tails) != 1 or len(heads) != 1:
                raise ValueError(
                    f"Column {k} of B must have exactly one negative (tail) and one positive (head) entry."
                )
            u = nodelist[tails[0]]
            v = nodelist[heads[0]]
            data = dict(G.edges[u, v]) if G.has_edge(u, v) else {}
            data.setdefault("edge_id", f"e{k + 1}")
            resolved.append((u, v, data))
        return resolved

    resolved = []
    for idx, (u, v, d) in enumerate(G.edges(data=True), start=1):
        data = dict(d)
        data.setdefault("edge_id", f"e{idx}")
        resolved.append((u, v, data))
    return resolved


# ==============================================================================
# 1. Plot Graph Topology As-Is
# ==============================================================================
def plot_graph(
    G,
    pos=None,
    title=r"$\mathcal{G}=(\mathcal{V},\mathcal{E})$",
    node_color="#2b5c8f",
    node_size=900,
    edge_color="#4a4a4a",
    edge_width=2.5,
    show_node_labels=True,
    node_labels=None,
    font_size=13,
    font_color="white",
    show_edge_labels=True,
    show_edge_weights=True,
    edge_font_size=10,
    figsize=(7.5, 4.2),
    margins=0.18,
    ax=None,
):
    """Plot a NetworkX graph with uniform node styling."""
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    pos = pos if pos is not None else nx.spring_layout(G, seed=42)
    nodelist = list(G.nodes())
    node_colors = [node_color] * len(nodelist)

    _draw_graph_layers(
        G, pos, nodelist, node_colors, ax,
        title=title,
        node_size=node_size,
        edge_color=edge_color,
        edge_width=edge_width,
        show_node_labels=show_node_labels,
        node_labels=node_labels,
        font_size=font_size,
        font_color=font_color,
        show_edge_labels=show_edge_labels,
        show_edge_weights=show_edge_weights,
        edge_font_size=edge_font_size,
        margins=margins,
    )

    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax


# ==============================================================================
# 2. Plot Graph with Discrete Node Clusters
# ==============================================================================
def plot_graph_clusters(
    G,
    clusters,
    pos=None,
    title="Graph Node Clusters",
    cluster_colors=None,
    cluster_labels=None,
    unassigned_color="#cccccc",
    show_legend=True,
    node_size=900,
    edge_color="#4a4a4a",
    edge_width=2.5,
    show_node_labels=True,
    node_labels=None,
    font_size=13,
    show_edge_labels=True,
    show_edge_weights=True,
    edge_font_size=10,
    figsize=(7.5, 4.2),
    margins=0.18,
    ax=None,
):
    """
    Plot a graph coloring each node cluster distinctly.

    Parameters
    ----------
    clusters : list/tuple of iterables OR dict {cluster_name: iterable_of_nodes}
        Node indices belonging to each cluster, e.g. [[1, 2, 3], [4, 5]].
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    pos = pos if pos is not None else nx.spring_layout(G, seed=42)
    nodelist = list(G.nodes())

    if isinstance(clusters, dict):
        auto_labels = list(clusters.keys())
        cluster_groups = list(clusters.values())
    else:
        cluster_groups = list(clusters)
        auto_labels = [f"Cluster {k + 1}" for k in range(len(cluster_groups))]

    labels_to_use = cluster_labels if cluster_labels is not None else auto_labels

    default_palette = [
        "#2b5c8f", "#d95f02", "#1b9e77", "#7570b3",
        "#e7298a", "#66a61e", "#e6ab02", "#a6761d",
    ]
    palette = cluster_colors if cluster_colors is not None else default_palette

    node_to_color = {}
    for k, group in enumerate(cluster_groups):
        c = palette[k % len(palette)]
        for n in group:
            node_to_color[n] = c

    node_colors = [node_to_color.get(n, unassigned_color) for n in nodelist]

    _draw_graph_layers(
        G, pos, nodelist, node_colors, ax,
        title=title,
        node_size=node_size,
        edge_color=edge_color,
        edge_width=edge_width,
        show_node_labels=show_node_labels,
        node_labels=node_labels,
        font_size=font_size,
        font_color=None,
        show_edge_labels=show_edge_labels,
        show_edge_weights=show_edge_weights,
        edge_font_size=edge_font_size,
        margins=margins,
    )

    if show_legend:
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                label=labels_to_use[k],
                markerfacecolor=palette[k % len(palette)],
                markeredgecolor="#1a2634",
                markersize=10,
            )
            for k in range(len(cluster_groups))
        ]
        ax.legend(handles=handles, frameon=True, loc="best", fontsize=10)

    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax


# ==============================================================================
# 3. Plot Continuous Node Signal via Colormap
# ==============================================================================
def plot_graph_signal(
    G,
    values,
    pos=None,
    nodelist=None,
    title="Graph Signal",
    cmap="viridis",
    vmin=None,
    vmax=None,
    symmetric_cbar=False,
    cbar_label="Signal value",
    show_cbar=True,
    show_values=False,
    value_fmt="{:.2f}",
    node_size=900,
    edge_color="#4a4a4a",
    edge_width=2.5,
    show_node_labels=True,
    node_labels=None,
    font_size=12,
    show_edge_labels=True,
    show_edge_weights=True,
    edge_font_size=10,
    figsize=(8.0, 4.2),
    margins=0.18,
    ax=None,
):
    """
    Plot a scalar graph signal over nodes using a continuous Matplotlib colormap.

    Parameters
    ----------
    values : array-like or dict {node: value}
        One scalar value per node (ordered by `nodelist` if array-like).
    symmetric_cbar : bool
        If True, centers the color scale at 0 ([-max|v|, +max|v|]); ideal for
        Laplacian eigenvectors u_p or local deviations Lx with diverging cmaps.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    pos = pos if pos is not None else nx.spring_layout(G, seed=42)
    nodelist = list(nodelist) if nodelist is not None else list(G.nodes())

    if isinstance(values, dict):
        vals = np.array([float(values[n]) for n in nodelist], dtype=float)
    else:
        vals = np.asarray(values, dtype=float).ravel()
        if len(vals) != len(nodelist):
            raise ValueError(f"Expected {len(nodelist)} values, got {len(vals)}.")

    if symmetric_cbar:
        abs_max = float(np.max(np.abs(vals))) if len(vals) > 0 else 1.0
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax
    else:
        vmin = float(np.min(vals)) if vmin is None else vmin
        vmax = float(np.max(vals)) if vmax is None else vmax

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    node_colors = [colormap(norm(v)) for v in vals]

    if show_node_labels and show_values and node_labels is None:
        labels_to_use = {
            n: f"$v_{{{n}}}$\n({value_fmt.format(v)})"
            for n, v in zip(nodelist, vals)
        }
    else:
        labels_to_use = node_labels

    _draw_graph_layers(
        G, pos, nodelist, node_colors, ax,
        title=title,
        node_size=node_size,
        edge_color=edge_color,
        edge_width=edge_width,
        show_node_labels=show_node_labels,
        node_labels=labels_to_use,
        font_size=font_size,
        font_color=None,
        show_edge_labels=show_edge_labels,
        show_edge_weights=show_edge_weights,
        edge_font_size=edge_font_size,
        margins=margins,
    )

    if show_cbar:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=11)

    if created_fig:
        plt.tight_layout()
        plt.show()
    return ax

# ==============================================================================
# 4. Plot Continuous Edge Signal via Colormap
# ==============================================================================

def plot_graph_edge_signal(
    G,
    edge_values,
    pos=None,
    B=None,
    edgelist=None,
    nodelist=None,
    title="Graph Edge Signal",
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    symmetric_cbar=True,
    color_by_magnitude=False,
    scale_width=True,
    width_range=(1.8, 6.0),
    show_arrows=True,
    arrow_mode="flow",
    negative_as_forward=True,
    arrowstyle="-|>",
    arrowsize=20,
    zero_tol=1e-12,
    cbar_label="Edge signal value",
    show_cbar=True,
    show_edge_labels=True,
    show_edge_values=True,
    edge_value_fmt="{:.2f}",
    edge_font_size=9.5,
    node_color="#3b4252",
    node_size=900,
    node_edgecolor="#1a2634",
    node_linewidth=1.5,
    show_node_labels=True,
    node_labels=None,
    font_size=13,
    font_color=None,
    figsize=(8.0, 4.2),
    margins=0.18,
    ax=None,
):
    """
    Plot a scalar edge signal over a graph, encoding values via edge color,
    width (proportional to |y_k|), and optional flow direction arrows.

    Parameters
    ----------
    B : array-like of shape (n, m), optional
        Oriented incidence matrix (-1 at source/tail, +1 at target/head). Used to
        infer edge order and reference orientation even if `G` is undirected.
    arrow_mode : {'flow', 'reference'}
        'flow' orients arrows in the physical flow direction; 'reference' keeps
        the fixed orientation encoded in `B` / `edgelist`.
    negative_as_forward : bool
        If True (default, for gradient signals y = B^T x = x_target - x_source),
        y_k < 0 draws flow u -> v and y_k > 0 draws v -> u. Set to False when
        plotting direct flow signals q where q_k > 0 already means u -> v.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    pos = pos if pos is not None else nx.spring_layout(G, seed=42)
    nodelist = list(nodelist) if nodelist is not None else list(G.nodes())

    resolved_edges = _resolve_oriented_edges(G, nodelist, B=B, edgelist=edgelist)
    m = len(resolved_edges)

    if isinstance(edge_values, dict):
        vals = np.array(
            [
                float(edge_values[(u, v)] if (u, v) in edge_values else edge_values[(v, u)])
                for u, v, _ in resolved_edges
            ],
            dtype=float,
        )
    else:
        vals = np.asarray(edge_values, dtype=float).ravel()
        if len(vals) != m:
            raise ValueError(f"Expected {m} edge values, got {len(vals)}.")

    c_vals = np.abs(vals) if color_by_magnitude else vals

    if symmetric_cbar and not color_by_magnitude:
        abs_max = float(np.max(np.abs(c_vals))) if m > 0 else 1.0
        abs_max = abs_max if abs_max > 0 else 1.0
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax
    else:
        vmin = float(np.min(c_vals)) if vmin is None else vmin
        vmax = float(np.max(c_vals)) if vmax is None else vmax
        if np.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    edge_colors = [colormap(norm(v)) for v in c_vals]

    # Scale edge widths proportionally to magnitude |y_k|
    if scale_width and m > 0:
        abs_vals = np.abs(vals)
        max_abs = float(np.max(abs_vals))
        w_min, w_max = width_range
        if max_abs > zero_tol:
            edge_widths = [w_min + (w_max - w_min) * (a / max_abs) for a in abs_vals]
        else:
            edge_widths = [w_min] * m
    else:
        edge_widths = [float(np.mean(width_range))] * m

    draw_edgelist = []
    for (u, v, _), val in zip(resolved_edges, vals):
        if show_arrows and arrow_mode == "flow":
            reverse = (val > zero_tol) if negative_as_forward else (val < -zero_tol)
            draw_edgelist.append((v, u) if reverse else (u, v))
        else:
            draw_edgelist.append((u, v))

    if show_arrows and arrow_mode == "flow":
        active_idx = [k for k, val in enumerate(vals) if abs(val) > zero_tol]
        zero_idx = [k for k, val in enumerate(vals) if abs(val) <= zero_tol]

        if zero_idx:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[draw_edgelist[k] for k in zero_idx],
                width=[edge_widths[k] for k in zero_idx],
                edge_color=[edge_colors[k] for k in zero_idx],
                arrows=False,
                node_size=node_size,
                ax=ax,
            )
        if active_idx:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[draw_edgelist[k] for k in active_idx],
                width=[edge_widths[k] for k in active_idx],
                edge_color=[edge_colors[k] for k in active_idx],
                arrows=True,
                arrowstyle=arrowstyle,
                arrowsize=arrowsize,
                node_size=node_size,
                ax=ax,
            )
    else:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=draw_edgelist,
            width=edge_widths,
            edge_color=edge_colors,
            arrows=show_arrows,
            arrowstyle=arrowstyle,
            arrowsize=arrowsize,
            node_size=node_size,
            ax=ax,
        )

    node_colors = [node_color] * len(nodelist) if isinstance(node_color, str) else list(node_color)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodelist,
        node_color=node_colors,
        node_size=node_size,
        edgecolors=node_edgecolor,
        linewidths=node_linewidth,
        ax=ax,
    )

    if show_node_labels:
        labels = node_labels or {n: f"$v_{{{n}}}$" for n in nodelist}
        if font_color is not None:
            nx.draw_networkx_labels(
                G, pos,
                labels=labels,
                font_color=font_color,
                font_weight="bold",
                font_size=font_size,
                ax=ax,
            )
        else:
            for node, color in zip(nodelist, node_colors):
                if node in labels:
                    nx.draw_networkx_labels(
                        G, pos,
                        labels={node: labels[node]},
                        font_color=_contrasting_text_color(color),
                        font_weight="bold",
                        font_size=font_size,
                        ax=ax,
                    )

    if show_edge_labels:
        edge_labels = {}
        for k, ((u, v, d), val) in enumerate(zip(resolved_edges, vals), start=1):
            eid = d.get("edge_id", f"e{k}")
            num = eid[1:] if isinstance(eid, str) and eid.startswith("e") else k
            if show_edge_values:
                edge_labels[(u, v)] = f"$e_{{{num}}}$: {edge_value_fmt.format(val)}"
            else:
                edge_labels[(u, v)] = f"$e_{{{num}}}$"
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=edge_font_size,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            ax=ax,
        )

    if show_cbar:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=11)

    if title:
        ax.set_title(title, fontsize=13, pad=10)
    ax.margins(margins)
    ax.axis("off")

    if created_fig:
        plt.tight_layout()
        plt.show()
    return None if created_fig else ax

# ==============================================================================
# 5. Plot Matrices as Colour-Pixel Images
# ==============================================================================

def plot_matrix(
    M,
    title="Matrix Heatmap",
    cmap="viridis",
    vmin=None,
    vmax=None,
    symmetric_cbar=False,
    mask_zeros=False,
    zero_color="#f4f6f9",
    zero_tol=1e-12,
    annot=False,
    annot_fmt="{:.2g}",
    annot_fontsize=9,
    xlabel="Column index (nodes / edges)",
    ylabel="Row index (nodes)",
    cbar_label="Entry value",
    show_cbar=True,
    show_grid=False,
    figsize=(5.5, 4.8),
    ax=None,
):
    """
    Represent a dense NumPy array or SciPy sparse matrix as a pixel image
    using a Matplotlib colormap.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)

    arr = M.toarray() if sp.issparse(M) else np.asarray(M, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}.")

    if symmetric_cbar:
        abs_max = float(np.max(np.abs(arr))) if arr.size > 0 else 1.0
        abs_max = abs_max if abs_max > 0 else 1.0
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax
    else:
        vmin = float(np.min(arr)) if vmin is None else vmin
        vmax = float(np.max(arr)) if vmax is None else vmax
        if np.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5

    colormap = plt.get_cmap(cmap).copy()
    if mask_zeros:
        data_to_plot = np.ma.masked_where(np.abs(arr) <= zero_tol, arr)
        colormap.set_bad(color=zero_color)
    else:
        data_to_plot = arr

    im = ax.imshow(
        data_to_plot,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    if show_grid:
        n_rows, n_cols = arr.shape
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="#d0d7de", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

    if annot:
        n_rows, n_cols = arr.shape
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(n_rows):
            for j in range(n_cols):
                val = arr[i, j]
                if mask_zeros and abs(val) <= zero_tol:
                    txt_color = "#6e7781"
                else:
                    r, g, b, _ = colormap(norm(val))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    txt_color = "black" if lum > 0.55 else "white"
                ax.text(
                    j, i, annot_fmt.format(val),
                    ha="center", va="center",
                    color=txt_color, fontsize=annot_fontsize, fontweight="bold",
                )

    if show_cbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=10)

    if title:
        ax.set_title(title, fontsize=12, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)

    if created_fig:
        plt.tight_layout()
        plt.show()
    return None if created_fig else ax

# ==============================================================================
# Helpers for hydraulic simulations
# ==============================================================================

def select_candidate_leak_nodes(wn, node_names, pos_dict, P_baseline, n_select, min_pressure=15.0):
    """Pick spatially distributed junctions whose minimum 24h baseline pressure > min_pressure."""
    eligible = [
        name for name in wn.junction_name_list
        if P_baseline[:, node_names.index(name)].min() >= min_pressure
    ]
    if len(eligible) <= n_select:
        return eligible

    # Greedy farthest-point sampling on (x, y) coordinates for uniform city-wide coverage
    coords = np.array([pos_dict[name] for name in eligible], dtype=float)
    selected_idx = [0]
    min_dists = np.linalg.norm(coords - coords[0], axis=1)
    for _ in range(1, n_select):
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)
        min_dists = np.minimum(min_dists, np.linalg.norm(coords - coords[next_idx], axis=1))
    return [eligible[i] for i in selected_idx]

def run_single_epanet_scenario(
    wn,
    node_names,
    link_names,
    junc_names,
    base_demands_0,
    demand_mult,
    n_timesteps,
    tmpdir,
    leak_node_name=None,
    leak_ce=0.0,
):
    """Apply perturbed demands and an optional all-day (24h) emitter leak, then run EPANET."""
    for jname, d0, m in zip(junc_names, base_demands_0, demand_mult):
        ts_list = wn.get_node(jname).demand_timeseries_list
        if len(ts_list) > 0:
            ts_list[0].base_value = float(d0 * m)

    has_leak = (leak_node_name is not None) and (leak_ce > 0.0)
    if has_leak:
        # Attach an open zero-head-loss emitter branch active across all 24 hours
        target_junc = wn.get_node(leak_node_name)
        wn.add_junction("__LEAK_JUNC__", base_demand=0.0, elevation=float(target_junc.elevation))
        wn.get_node("__LEAK_JUNC__").emitter_coefficient = float(leak_ce)
        wn.add_pipe(
            "__LEAK_PIPE__",
            start_node_name=leak_node_name,
            end_node_name="__LEAK_JUNC__",
            length=0.1,
            diameter=1.0,
            roughness=150.0,
            initial_status="OPEN",
        )

    try:
        sim = wntr.sim.EpanetSimulator(wn)
        res = sim.run_sim(file_prefix=os.path.join(tmpdir, "modena_leak_run"))

        P = res.node["pressure"][node_names].iloc[:n_timesteps].to_numpy(dtype=np.float32)
        D = res.node["demand"][node_names].iloc[:n_timesteps].to_numpy(dtype=np.float32)
        H = res.node["head"][node_names].iloc[:n_timesteps].to_numpy(dtype=np.float32)
        Q = res.link["flowrate"][link_names].iloc[:n_timesteps].to_numpy(dtype=np.float32)
        q_leak = (
            res.link["flowrate"]["__LEAK_PIPE__"].iloc[:n_timesteps].to_numpy(dtype=np.float32)
            if has_leak else np.zeros(n_timesteps, dtype=np.float32)
        )
    finally:
        if has_leak:
            wn.remove_link("__LEAK_PIPE__")
            wn.remove_node("__LEAK_JUNC__")
        for jname, d0 in zip(junc_names, base_demands_0):
            ts_list = wn.get_node(jname).demand_timeseries_list
            if len(ts_list) > 0:
                ts_list[0].base_value = float(d0)

    return P, D, H, Q, q_leak
