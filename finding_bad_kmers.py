"""
Longest Common Substring: Cross-Library Analysis
==================================================
Computes LCS between every brick in Library A (~350 bricks)
and every brick in Library B (36 bricks).

Produces:
  1. A rectangular heatmap (rows=LibA, cols=LibB) with LCS length as gradient
  2. A text report of the top pairings
  3. A per-LibA summary: which LibA brick has the highest similarity to any LibB brick

Usage:
  Fill in `library_a` and `library_b` lists at the bottom and run.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ─────────────────────────────────────────────
#  Core LCS computation
# ─────────────────────────────────────────────

def longest_common_substring(s1: str, s2: str) -> tuple[str, int]:
    """Return (substring, length) of the longest common substring."""
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
    return s1[best_end - best_len:best_end], best_len


def compute_cross_lcs(library_a: list[str], library_b: list[str]):
    """
    Compute LCS between every pair (one from each library).
    Returns:
        substrings: list[list[str]]  — [len_a][len_b]
        lengths:    np.ndarray       — shape (len_a, len_b)
    """
    na, nb = len(library_a), len(library_b)
    substrings = [[""] * nb for _ in range(na)]
    lengths = np.zeros((na, nb), dtype=int)

    total = na * nb
    for i, seq_a in enumerate(library_a):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing brick A{i} / {na}  "
                  f"({(i * nb) / total * 100:.0f}% done)...")
        for j, seq_b in enumerate(library_b):
            sub, length = longest_common_substring(seq_a, seq_b)
            substrings[i][j] = sub
            lengths[i][j] = length

    return substrings, lengths


# ─────────────────────────────────────────────
#  Rectangular heatmap
# ─────────────────────────────────────────────

def plot_cross_heatmap(library_a, library_b, lengths, substrings,
                       output_path="cross_lcs_heatmap.png",
                       label_a="Library A brick",
                       label_b="Library B brick"):
    """
    Rectangular heatmap: rows = library_a, cols = library_b.
    Cell colour = LCS length (gradient).
    """
    na, nb = lengths.shape
    max_len = int(lengths.max()) or 1

    # Figure size — adapt to matrix shape
    cell_w = max(0.25, 12 / nb)
    cell_h = max(0.08, 20 / na)
    fig_w = nb * cell_w + 3
    fig_h = na * cell_h + 2
    fig_w = min(fig_w, 40)
    fig_h = min(fig_h, 80)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.YlOrRd.copy()
    im = ax.imshow(lengths, cmap=cmap, vmin=0, vmax=max_len,
                   aspect="auto", interpolation="nearest")

    # Axis labels
    ax.set_xticks(range(nb))
    ax.set_yticks(range(na))
    x_labels = [f"B{j}" for j in range(nb)]
    y_labels = [f"A{i}" for i in range(na)]
    tick_fs = max(3, 7 - max(na, nb) // 50)
    ax.set_xticklabels(x_labels, fontsize=tick_fs, rotation=90)
    ax.set_yticklabels(y_labels, fontsize=tick_fs)

    ax.set_xlabel(label_b, fontsize=10)
    ax.set_ylabel(label_a, fontsize=10)
    ax.set_title("Longest Common Substring: Cross-Library Comparison",
                 fontsize=12, fontweight="bold", pad=10)

    # Annotate cells only if matrix is small enough
    if na * nb <= 2000:
        font_size = max(3, 5 - max(na, nb) // 30)
        for i in range(na):
            for j in range(nb):
                l = lengths[i][j]
                if l > 0:
                    sub = substrings[i][j]
                    display = sub if len(sub) <= 6 else sub[:3] + ".." + sub[-2:]
                    txt = f"{display}\n({l})"
                    text_color = "white" if l > max_len * 0.6 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=font_size, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02, label="LCS length")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nHeatmap saved to: {output_path}")


# ─────────────────────────────────────────────
#  Per-brick-A summary: max similarity to any B
# ─────────────────────────────────────────────

def plot_max_similarity_bar(library_a, lengths, substrings,
                            output_path="max_similarity_per_brick.png",
                            label_a="Library A brick"):
    """
    Horizontal bar chart: for each brick in library A, show its
    maximum LCS length to any brick in library B.
    Colour = max LCS length (gradient).
    """
    na = len(library_a)
    max_per_a = lengths.max(axis=1)                  # best match length
    best_b_idx = lengths.argmax(axis=1)              # which B brick
    best_sub = [substrings[i][best_b_idx[i]] for i in range(na)]

    # Sort by max LCS descending
    order = np.argsort(max_per_a)[::-1]

    max_len = int(max_per_a.max()) or 1
    norm = mcolors.Normalize(vmin=0, vmax=max_len)
    cmap = plt.cm.YlOrRd

    fig_h = max(6, na * 0.18)
    fig, ax = plt.subplots(figsize=(12, min(fig_h, 60)))

    y_pos = range(len(order))
    colors = [cmap(norm(max_per_a[i])) for i in order]
    bars = ax.barh(y_pos, [max_per_a[i] for i in order],
                   color=colors, edgecolor="#333333", linewidth=0.3)

    labels = []
    for idx in order:
        b_idx = best_b_idx[idx]
        sub = best_sub[idx]
        labels.append(f"A{idx} → B{b_idx}: {sub} ({max_per_a[idx]})")

    ax.set_yticks(y_pos)
    fs = max(3, 6 - na // 80)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.invert_yaxis()
    ax.set_xlabel("Max LCS length to any Library B brick")
    ax.set_title(f"Per-{label_a}: Best Match to Library B",
                 fontsize=12, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02, label="LCS length")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Max-similarity bar chart saved to: {output_path}")


# ─────────────────────────────────────────────
#  Text report
# ─────────────────────────────────────────────

def print_report(library_a, library_b, lengths, substrings, top_n=30):
    na, nb = lengths.shape

    # Flatten and sort
    pairs = []
    for i in range(na):
        for j in range(nb):
            if lengths[i][j] > 0:
                pairs.append((lengths[i][j], substrings[i][j], i, j))
    pairs.sort(reverse=True)

    print(f"\n{'=' * 70}")
    print(f"TOP {top_n} CROSS-LIBRARY LCS PAIRS")
    print(f"{'=' * 70}")
    print(f"  {'Rank':<6}{'Pair':<25}{'Length':<8}{'Substring'}")
    print(f"  {'─' * 60}")
    for rank, (l, sub, i, j) in enumerate(pairs[:top_n], 1):
        print(f"  {rank:<6}A{i} & B{j:<18}{l:<8}{sub}")

    # Distribution summary
    print(f"\n{'=' * 70}")
    print("LCS LENGTH DISTRIBUTION")
    print(f"{'=' * 70}")
    flat = lengths.flatten()
    for threshold in range(int(flat.max()), 3, -1):
        count = int((flat >= threshold).sum())
        if count > 0:
            print(f"  LCS >= {threshold}: {count} pairs")

    # Per-A summary
    max_per_a = lengths.max(axis=1)
    best_b = lengths.argmax(axis=1)
    print(f"\n{'=' * 70}")
    print("LIBRARY A BRICKS WITH HIGHEST SIMILARITY TO LIBRARY B")
    print(f"{'=' * 70}")
    flagged = [(i, max_per_a[i], best_b[i]) for i in range(na) if max_per_a[i] >= 7]
    flagged.sort(key=lambda x: x[1], reverse=True)
    if flagged:
        print(f"  {'Brick A':<12}{'Max LCS':<10}{'Best B':<10}{'Substring'}")
        print(f"  {'─' * 55}")
        for i, ml, bj in flagged:
            print(f"  A{i:<10}{ml:<10}B{bj:<9}{substrings[i][bj]}")
    else:
        print("  No Library A brick shares >= 7 bases with any Library B brick.")


# ──────────────────────────────────────────────
#  FILL IN YOUR SEQUENCES HERE
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # Library A: your ~350 bricks (24-nt each)
    library_a = [
"TATCCGAATCCTGGCGAAGG",
"GACTCGTCATCTGGCGGACA",
"TTAACGAAGGTTGACGCTTG",
"GTCACGGTATTGTGCGGTAT",
"TGCACGAATTGATCCGAGTG",
"GTTACGACACATGCCGTGTA",
"GTATCGAAGACAGACGAAGT",
"GAGACGTAATTAGGCGACAT",
"TGTACGATCATAAGCGCAGA",
"CAGCCGTAGGAATTCGCTTC",
"ACAACGGAGACCTGCGACAG",
"TATGCGAGTATAGTCGCTGA",
"ATATCGCTAACCAGCGCATG",
"TGTTCGTATGTGATCGCAGA",
"ACTGCGGAGTGAAGCGCTCA",
"AATGCGAAGGATCACGTTCT",
"GTGGCGGAGAGTTCCGTCTA",
"CCAGCGTTCATTGCCGCTAA",
"GAATCGTAAGGTACCGTCTG",
"ATAACGGACATTACCGTCCA",
"TCCACGCTAACTTACGTGCT",
"TAGGCGTATAGTTGCGGAAG",
"ACCACGATCATGGTCGGTAC",
"TAATCGAGACTGGCCGATAG",
"ATCACGATACTGCACGCACT",
"TCTTCGTTGACTAGCGGCTA",
"TGAACGTCTAGGTACGAGGT",
"TGTCCGCCTATACTCGGACT",
"AGTTCGTCATACTACGCTGA",
"TGCTCGACATATAACGTCCA",
"AGGCCGCTATGCAGCGGTAT",
"ATAACGTGTCACATCGACTT",
"CTAGCGACTCTGTACGAGCC",
"ACTGCGTGATACCACGATCT",
"TAGCCGATAACCTACGGAGC",
"GTCTCGTCCTTGTCCGAGGA",
"TTGCCGGTTGACTACGCCAA",
"CTAACGGCTTGGATCGTTAA",
"ACACCGCCTTAATGCGATTG",
"CTGACGATTGCTCTCGTACC",
"CACTCGACTATGGTCGATAT",
"CAGTCGGCTCTTGCCGCATT",
"GAAGCGCATACTTGCGCTCT",
"GGTCCGTAACATCTCGGATG",
"TCAACGTACTCATACGTCTG",
"TTACCGTGGCCTCTCGCCAT",
"CTGGCGTGATACTACGCCTG",
"AGTACGGCCTGATGCGTCTC",
"TAGCCGAATGGCCTCGTGAT",
"TCTCCGTCCATTAACGTATG",
"CAGCCGTTATTATGCGGATG",
"AGTTCGTGGTGAACCGTTGC",
"CACACGGTCATTCACGGTGT",
"ACAGCGCACCATGGCGGTTA",
"GCAGCGTTAGTCCTCGCTCT",
"TAATCGCAGCACCTCGATAG",
"GAAGCGGAACTAACCGCTGA",
"GGATCGACTGTACACGATGG",
"TCTACGAAGGAATTCGAGCT",
"CCAGCGATAGAACACGCTCC",
"AATGCGTCTCAAGGCGCAGA",
"TATCCGCCACTAGCCGGAAG",
"AGTACGGAATAATCCGGTAG",
"AACACGAGTTAGGTCGTCAT",
"AGGTCGAAGCTACTCGAATA",
"TTGACGCTGACCAACGTACC",
"ACAACGTAAGACTACGTAAC",
"ATATCGCATTGATGCGCTAA",
"CATGCGGAATGAACCGTAGC",
"AATACGTAGTGCACCGAGCC",
"TGTCCGGAGCTTCACGCTGA",
"ACTACGTACTATTCCGTGCA",
"GCTCCGAACTAGACCGATCT",
"CACACGCATGTTACCGCTGT",
"GATGCGGAATCTTCCGACCT",
"GACTCGAAGAATGTCGACAA",
"TAACCGACAGTAATCGGTAG",
"CTCTCGTCTAGTGACGGCAC",
"CTACCGATGTTGCTCGATAC",
"AGATCGATCTGCCTCGAAGA",
"CTATCGAGTGAACTCGCCTC",
"GGTACGCAAGGAAGCGTGAT",
"CAACCGTAATCCTACGAACT",
"GAAGCGTCTGCTAGCGCTTA",
"GCTTCGTATTCTAACGTCAG",
"CAATCGGTGGTTACCGTTCT",
"CAACCGCCTACTCTCGGTTG",
"AATTCGGAGATAGCCGGCCT",
"AGTGCGCAGGATGCCGCTAA",
"ATGTCGAAGGCATTCGTTGT",
"CCTACGCTACTCAGCGGAAC",
"GTTCCGGTAAGAGTCGTCTA",
"ACAGCGCAGTAATGCGAAGT",
"AGTTCGCCATTAGTCGAACC",
"GGTGCGATATTAATCGGCTT",
"GGTGCGAATGATTGCGCACA",
"CTCTCGAATGATATCGTCTA",
"GATCCGGTTGTAGCCGGTCT",
"CACACGGAGTTCACCGTGTC",
"GTAACGATAGATACCGTTGC",
"TGAGCGGCAGCTCACGGTTA",
"AGATCGGTATGGTCCGCAGA",
"TGAGCGGAAGGTTCCGGTAT",
"ATTGCGTACTCATCCGCATA",
"GCATCGAATAAGGTCGCTGA",
"CTAACGTTCACCAACGAGTC",
"TTGGCGCTTGATAGCGTAGG",
"GATCCGTGGTAGAACGGTTC",
"ATTACGTAGCAGAGCGTATA",
"TATACGAGGCTCCACGCCAG",
"GTTCCGGAGTTATCCGGTAG",
"AATTCGTCAATTAGCGTTGA",
"TTGACGACAACAATCGCCAC",
"TTACCGTATTACTTCGTGAC",
"TTGACGTGCCTTCACGCTCC",
"ATATCGAACACTGTCGTGAA",
"CATTCGTCATTGAACGAATC",
"GAGGCGGATGACTCCGTTAC",
"CTGCCGTGCACACACGTTCA",
"AGACCGTCTTATTGCGGTGA",
"GTTGCGCTGAGACACGCAAC",
"TACTCGTGTAGTACCGAGTG",
"GTCACGAGCTATCTCGGTGT",
"CAATCGAAGCATTACGCAAT",
"TGTCCGTTAACTCACGACAA",
"CAACCGATTACTATCGTAGT",
"ATAGCGGTCAAGGTCGCTCT",
"CTCACGCAATGAGCCGTGGT",
"AGAGCGAGAGTGCTCGAACC",
"GAGACGTAGGTGTACGACTA",
"GCATCGTCTCCTAACGCTAT",
"CCTACGAGTCAGTACGTGTG",
"CATTCGCCAGTTGGCGACTC",
"GTTCCGTGCTGCCACGTTAC",
"AGGTCGCTCTACAGCGCTCT",
"GACACGGTATGCACCGTCCA",
"CAGACGTATAGACACGTCAA",
"TGGACGAGTTGTGCCGTATG",
"GATGCGTAGCAATACGATAC",
"CATACGTAAGTCTACGGTGA",
"GGTTCGCTTAATAGCGGAAG",
"TTACCGTGTGAATCCGGTCT",
"ACTGCGCCATTGGCCGAATT",
"GCTCCGATCTGAATCGACCT",
"AGTACGGTTGTGACCGGACA",
"GATCCGTTACCATTCGTAAC",
"TGTGCGAGAATCTGCGCATA",
"GTTGCGCTATGGAGCGAGTA",
"ACTCCGCTCACTAGCGATAT",
"GGCACGAGTAGTGTCGAATC",
"GGATCGCACAGATGCGTGGT",
"TTCTCGAGTCTTCTCGTATA",
"CAGTCGAGTCTCTACGGAAC",
"CCATCGCCAACAGGCGATTG",
"AGGTCGGCTGTAGCCGATTA",
"AGCACGTGGAACTTCGGTGC",
"CAGCCGTCTTATACCGGCAC",
"AAGGCGCTCCTCACCGATAT",
"ATTGCGCCAAGTAGCGCATA",
"CTCACGGTTGCCACCGTTGA",
"ACAACGATTATGGACGTTCT",
"ACAGCGACAAGATCCGCTCA",
"TGGACGAGATTCCTCGGACC",
"ATGCCGTGTTCATGCGGCTT",
"TGTTCGTCACTGTACGAATG",
"TCCACGCTGTTATGCGTAGT",
"AGGTCGTGAGCATACGATTG",
"GAAGCGATTGGACTCGCTCC",
"CAAGCGACAGATGTCGCATA",
"TCATCGATTCCATGCGATGA",
"TGCTCGACCTAAGTCGACTT",
"GTTCCGATCTGTTCCGCAGA",
"CCACCGCTGAATTGCGGAAG",
"TGTACGGTAGATAACGCTAG",
"CAGGCGTGTCTTAACGCAGT"
    ]

    # Library B: your 36 CG-site bricks
    library_b = [
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

    print("Computing cross-library LCS matrix...")
    substrings, lengths = compute_cross_lcs(library_a, library_b)

    print_report(library_a, library_b, lengths, substrings, top_n=30)

    plot_cross_heatmap(library_a, library_b, lengths, substrings,
                       output_path="cross_lcs_heatmap.png",
                       label_a="~350 Bricks",
                       label_b="36 CG-site Bricks")

    plot_max_similarity_bar(library_a, lengths, substrings,
                            output_path="max_similarity_per_brick.png",
                            label_a="~350 Brick")