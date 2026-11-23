import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
