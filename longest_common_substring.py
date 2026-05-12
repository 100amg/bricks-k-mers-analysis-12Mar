"""
Primer LCS Visualisations
==========================
1. Clustered heatmap with dendrogram — primers reordered by similarity
2. Network graph — primers as nodes, edges for significant LCS overlaps

Replace the `primers` list at the bottom with your 38 sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import networkx as nx


# ─────────────────────────────────────────────
#  Core LCS computation (same as before)
# ─────────────────────────────────────────────

def longest_common_substring(s1: str, s2: str) -> str:
    s1, s2 = s1.upper(), s2.upper()
    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    best_len = 0
    best_end = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_end = i
        prev = curr
    return s1[best_end - best_len:best_end]


def compute_lcs_matrix(primers):
    n = len(primers)
    substrings = [[""] * n for _ in range(n)]
    lengths = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            lcs = longest_common_substring(primers[i], primers[j])
            substrings[i][j] = lcs
            substrings[j][i] = lcs
            lengths[i][j] = len(lcs)
            lengths[j][i] = len(lcs)
    return substrings, lengths


# ─────────────────────────────────────────────
#  1.  Clustered heatmap with dendrogram
# ─────────────────────────────────────────────

def plot_clustered_heatmap(primers, lengths, substrings,
                           output_path="lcs_clustered_heatmap.png"):

    n = len(primers)
    length_arr = np.array(lengths, dtype=float)
    max_len = int(length_arr.max()) or 1

    # Convert similarity → distance for clustering
    # distance = max_len - LCS_length (shorter LCS = farther apart)
    dist_matrix = max_len - length_arr
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix)

    # Hierarchical clustering
    Z = linkage(condensed, method="average")

    # --- Layout: dendrogram on top + left, heatmap in centre ---
    fig = plt.figure(figsize=(18, 16))

    # Gridspec: col 0 = left dendro, col 1 = heatmap, col 2 = colorbar
    #           row 0 = top dendro,  row 1 = heatmap
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[0.12, 1, 0.03],
        height_ratios=[0.15, 1],
        wspace=0.01, hspace=0.01,
    )

    # Top dendrogram
    ax_top = fig.add_subplot(gs[0, 1])
    dn_top = dendrogram(Z, ax=ax_top, no_labels=True,
                        color_threshold=0, above_threshold_color="#555555")
    ax_top.set_axis_off()
    order = dn_top["leaves"]

    # Left dendrogram (rotated)
    ax_left = fig.add_subplot(gs[1, 0])
    dendrogram(Z, ax=ax_left, orientation="left", no_labels=True,
               color_threshold=0, above_threshold_color="#555555")
    ax_left.set_axis_off()

    # Reorder matrix
    reordered = length_arr[np.ix_(order, order)]

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    cmap = plt.cm.YlOrRd.copy()
    im = ax_heat.imshow(reordered, cmap=cmap, vmin=0, vmax=max_len,
                        aspect="equal", interpolation="nearest")

    # Labels
    labels = [f"P{i}" for i in order]
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    tick_fs = max(5, 9 - n // 12)
    ax_heat.set_xticklabels(labels, fontsize=tick_fs, rotation=90)
    ax_heat.set_yticklabels(labels, fontsize=tick_fs)

    # Annotate cells (upper triangle only)
    font_size = max(3, 5.5 - n // 15)
    for ri in range(n):
        for ci in range(ri + 1, n):
            oi, oj = order[ri], order[ci]
            lcs = substrings[oi][oj]
            l = lengths[oi][oj]
            if l > 0:
                display = lcs if len(lcs) <= 8 else lcs[:3] + ".." + lcs[-3:]
                txt = f"{display}\n({l})"
            else:
                txt = ""
            text_color = "white" if l > max_len * 0.6 else "black"
            ax_heat.text(ci, ri, txt, ha="center", va="center",
                         fontsize=font_size, color=text_color)

    # Blank lower triangle
    for ri in range(n):
        for ci in range(0, ri + 1):
            ax_heat.add_patch(plt.Rectangle(
                (ci - 0.5, ri - 0.5), 1, 1,
                fill=True, color="white", ec="white", zorder=2))

    # Colorbar
    ax_cb = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=ax_cb, label="LCS length")

    fig.suptitle("Clustered Heatmap — Longest Common Substring",
                 fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Clustered heatmap saved to: {output_path}")


# ─────────────────────────────────────────────
#  2.  Network graph
# ─────────────────────────────────────────────

def plot_network_graph(primers, lengths, substrings,
                       edge_threshold=5,
                       output_path="lcs_network.png"):
    """
    Network graph where:
      - Each primer is a node
      - An edge is drawn if LCS length >= edge_threshold
      - Edge width & colour scale with LCS length
      - Edge labels show the substring and length
    """
 
    n = len(primers)
    G = nx.Graph()
 
    # Add all primers as nodes
    for i in range(n):
        G.add_node(i, label=f"P{i}")
 
    # Add edges above threshold
    max_len = 0
    for i in range(n):
        for j in range(i + 1, n):
            l = lengths[i][j]
            if l >= edge_threshold:
                lcs = substrings[i][j]
                G.add_edge(i, j, weight=l, lcs=lcs)
                if l > max_len:
                    max_len = l
 
    if max_len == 0:
        max_len = 1
 
    # Layout — spring layout with distance inversely related to LCS
    pos = nx.spring_layout(G, k=2.5 / np.sqrt(n), iterations=80, seed=42,
                           weight="weight")
 
    fig, ax = plt.subplots(figsize=(16, 14))
 
    # ── Draw edges ──
    edges = G.edges(data=True)
    if edges:
        weights = [d["weight"] for _, _, d in edges]
        # Width: scale 1–6
        widths = [1.5 + 5 * (w - edge_threshold) / max(1, max_len - edge_threshold)
                  for w in weights]
        # Colour: use a truncated Reds colormap (skip the near-white end)
        reds_full = plt.cm.Reds
        truncated_reds = mcolors.LinearSegmentedColormap.from_list(
            "reds_dark", reds_full(np.linspace(0.45, 1.0, 256))
        )
        norm = mcolors.Normalize(vmin=edge_threshold, vmax=max_len)
        edge_colors = [truncated_reds(norm(w)) for w in weights]
 
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths,
                               edge_color=edge_colors, alpha=0.9)
 
        # Edge labels
        edge_labels = {(u, v): f"{d['lcs']}\n({d['weight']})"
                       for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     ax=ax, font_size=6, font_color="#333333",
                                     bbox=dict(boxstyle="round,pad=0.15",
                                               fc="white", ec="none", alpha=0.8))
 
    # ── Draw nodes ──
    # Node size proportional to how many edges it has
    degrees = dict(G.degree())
    node_sizes = [300 + 80 * degrees.get(i, 0) for i in G.nodes()]
 
    # Colour nodes by degree
    degree_vals = [degrees.get(i, 0) for i in G.nodes()]
    max_deg = max(degree_vals) if degree_vals else 1
    node_norm = mcolors.Normalize(vmin=0, vmax=max_deg)
    node_colors = [plt.cm.YlOrRd(node_norm(d)) for d in degree_vals]
 
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors="#333333", linewidths=1.2)
 
    # Node labels
    node_labels = {i: f"P{i}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=8, font_weight="bold")
 
    # Legend / info
    info_text = (f"Primers: {n}   |   Edge threshold: LCS ≥ {edge_threshold}\n"
                 f"Edges shown: {G.number_of_edges()}   |   "
                 f"Isolated nodes (no edges): "
                 f"{sum(1 for i in G.nodes() if degrees.get(i, 0) == 0)}")
    ax.text(0.01, 0.01, info_text, transform=ax.transAxes,
            fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
 
    # Colourbar for edges
    reds_full = plt.cm.Reds
    cbar_cmap = mcolors.LinearSegmentedColormap.from_list(
        "reds_dark", reds_full(np.linspace(0.45, 1.0, 256))
    )
    sm = plt.cm.ScalarMappable(cmap=cbar_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, label="LCS length")
 
    ax.set_title(f"Primer Overlap Network  (edges: LCS ≥ {edge_threshold})",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Network graph saved to: {output_path}")
 
    # Print hub primers
    ranked = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Most connected primers (edges with LCS ≥ {edge_threshold}):")
    for i, (node, deg) in enumerate(ranked[:10], 1):
        if deg == 0:
            break
        print(f"    {i}. Primer {node}: {deg} connections")
 
# ──────────────────────────────────────────────
#  Replace with your 38 primers
# ──────────────────────────────────────────────
if __name__ == "__main__":

    primers = [
"CCACAATCTTGCGTTAGGTGGAACTCTTCCAGCTTCGGATAGGAAGCA",
"TGACTAGAAGTCGTGTCGGCCATT",
"AGTCATACTGTCGCCGGTGCAGTC",
"AACTAGAACTTCGTCGTTGAGCGC",
"AACAATTCTTGCGATTAGGCGGCC",
"CAACAACTTGACGGAGTTGGCTTC",
"TATGTATTACTCGTCACGGAATGC",
"CTCTCAATTCTCGCATAAGACGTG",
"ACCTCTATAGACGATGGTGAGCTC",
"CACTCATATCACGTCAATGGCAGG",
"GTAATACTCTTCGGTGGAGCCGAC",
"CCACCAACAATCGTATTGTGGTCA",
"CGCTTGCTTAACGGCAAGGCTAAT",
"ACCTCTACTGTCGTAGAGGCAGAT",
"CACACTCCATTCGAAGTGGAGGTT",
"ATTATTCAAGGCGTCGAAGCCGCT",
"CTACATACTTGCGTAGTAGCACGG",
"CAATAACTCTCCGGTTAGGTGAGC",
"CCATATTCCATCGCCGTGGAGTAG",
"CCTCACAATTACGTTAGTGCATGA",
"CTCTCGTGTAACGAAGGTGTCACA",
"ACTTCCATTAGCGGTCGTGTCTAC",
"ACATCCGATGTCGATGAGGTATCC",
"ATCAATTAAGGCGACTGTTCCGCG",
"CGCATTCGGACCGTAATGGTAATC",
"CAAGAGTATTGCGCAGTTGCCTCA",
"CACTAACTACACGTAGCTGTGGTG",
"CTACAGACGTTCGAAGAGGTTCAC",
"CCAATACCTGACGGACATCGCTAG",
"AACTAAGTGGCCGTATCGGCATCT",
"ACACCTATATCCGTTACGTCAACC",
"CCTAATCTAAGCGGTATTGCAGCG",
"TCCGTGATCAACGTTGTGGTTGGA",
"CTTCTCTACTCCGGATGTGAGAAG",
"CTCCATTGAGACGTGAATGCATCG",
"CCGGAGCAACACGTATTGGTTACT",
"CAGCAGAGTTCCGTATAGGTTCCA",
"ACTAACTTATTCGATGCGGCCGAGCTCAAGACCTACGTGGAGGTCTAT"
]

print("Computing pairwise LCS matrix...")
substrings, lengths = compute_lcs_matrix(primers)

plot_clustered_heatmap(primers, lengths, substrings,
                           output_path="lcs_clustered_heatmap.png")

    # edge_threshold: only show edges where LCS length >= this value
    # adjust this for your data — try 5 or 6 for 38 primers
plot_network_graph(primers, lengths, substrings,
                       edge_threshold=5,
                       output_path="lcs_network_updated.png")