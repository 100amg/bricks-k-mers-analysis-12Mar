"""
CpG Context K-mer Comparison
==============================
Compares the local sequence context around CpG methylation sites between:
  - Library A: 38 bricks, 24-nt, CpG at index 11
  - Library B: 165 bricks, 20-nt, CpG at index 4 and index 14

For each CpG site, extracts a flanking window and generates k-mers (k=5–8).
Matches k-mers between Library A and each Library B CpG site to score
how similar the local methylation context is.

The goal: identify which 16 CpG sites from Library B were likely discarded
based on poor context similarity to Library A.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict


# ─────────────────────────────────────────────
#  CpG context extraction
# ─────────────────────────────────────────────

def extract_cpg_context(sequence: str, cpg_index: int, window: int = 8) -> str:
    """
    Extract a flanking window around a CpG site.
    Returns the substring centred on the CpG, padded if near edges.
    """
    seq = sequence.upper()
    start = max(0, cpg_index - window)
    end = min(len(seq), cpg_index + 2 + window)  # +2 because CpG is 2 bases
    return seq[start:end]


def generate_kmers_from_context(context: str, k: int) -> set:
    """Generate all k-mers from a context string."""
    return {context[i:i + k] for i in range(len(context) - k + 1)}


# ─────────────────────────────────────────────
#  Scoring: k-mer overlap between two contexts
# ─────────────────────────────────────────────

def context_similarity_score(context_a: str, context_b: str,
                              k_range: range = range(5, 9)) -> dict:
    """
    Score the similarity between two CpG contexts by k-mer overlap.
    Returns per-k scores and a combined score.
    """
    scores = {}
    total_shared = 0
    total_possible = 0
    shared_kmers_all = []

    for k in k_range:
        kmers_a = generate_kmers_from_context(context_a, k)
        kmers_b = generate_kmers_from_context(context_b, k)
        shared = kmers_a & kmers_b
        # Jaccard-like: shared / union
        union = kmers_a | kmers_b
        jaccard = len(shared) / len(union) if union else 0
        scores[k] = {
            "shared": len(shared),
            "total_a": len(kmers_a),
            "total_b": len(kmers_b),
            "jaccard": jaccard,
            "shared_kmers": shared,
        }
        total_shared += len(shared)
        total_possible += len(union) if union else 1
        shared_kmers_all.extend(shared)

    scores["combined"] = total_shared / max(total_possible, 1)
    scores["total_shared"] = total_shared
    scores["shared_kmers_all"] = shared_kmers_all
    return scores


# ─────────────────────────────────────────────
#  Longest common substring (for additional context)
# ─────────────────────────────────────────────

def longest_common_substring(s1: str, s2: str) -> tuple:
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


# ─────────────────────────────────────────────
#  Main analysis
# ─────────────────────────────────────────────

def run_cpg_context_analysis(
    bricks_24: list[str],  # 38 bricks, 24-nt, CpG at index 11
    bricks_20: list[str],  # 165 bricks, 20-nt, CpG at index 4 and 14
    cpg_index_24: int = 11,
    cpg_indices_20: list[int] = [4, 14],
    window: int = 8,
    k_range: range = range(5, 9),
):
    """
    For each CpG site in the 20-nt bricks (2 per brick = ~330 sites),
    score its context similarity to ALL 38 CpG contexts in the 24-nt bricks.
    """

    # Extract contexts for Library A (38 bricks)
    contexts_a = []
    for i, seq in enumerate(bricks_24):
        ctx = extract_cpg_context(seq, cpg_index_24, window)
        contexts_a.append({"brick": i, "cpg_idx": cpg_index_24, "context": ctx})

    # Extract contexts for Library B (165 bricks × 2 CpG sites)
    contexts_b = []
    for i, seq in enumerate(bricks_20):
        for cpg_idx in cpg_indices_20:
            ctx = extract_cpg_context(seq, cpg_idx, window)
            contexts_b.append({
                "brick": i,
                "cpg_idx": cpg_idx,
                "label": f"B{i}_CpG{cpg_idx}",
                "context": ctx,
            })

    num_a = len(contexts_a)
    num_b = len(contexts_b)
    print(f"Library A: {num_a} CpG contexts (from {len(bricks_24)} bricks, CpG at {cpg_index_24})")
    print(f"Library B: {num_b} CpG contexts (from {len(bricks_20)} bricks, CpG at {cpg_indices_20})")

    # ── Score every B site against all A sites ──
    # For each B site, compute:
    #   - max similarity to any A site
    #   - average similarity across all A sites
    #   - best LCS to any A context

    site_scores = []
    # Also build a matrix for heatmap: rows = B sites, cols = A bricks
    score_matrix = np.zeros((num_b, num_a))
    lcs_matrix = np.zeros((num_b, num_a), dtype=int)

    for bi, b_ctx in enumerate(contexts_b):
        if (bi + 1) % 50 == 0 or bi == 0:
            print(f"  Scoring B site {bi + 1}/{num_b}...")

        best_score = 0
        best_a = -1
        best_lcs_len = 0
        best_lcs_sub = ""
        total_score = 0

        for ai, a_ctx in enumerate(contexts_a):
            sim = context_similarity_score(a_ctx["context"], b_ctx["context"], k_range)
            combined = sim["combined"]
            score_matrix[bi, ai] = combined

            lcs_sub, lcs_len = longest_common_substring(a_ctx["context"], b_ctx["context"])
            lcs_matrix[bi, ai] = lcs_len

            total_score += combined
            if combined > best_score:
                best_score = combined
                best_a = ai
            if lcs_len > best_lcs_len:
                best_lcs_len = lcs_len
                best_lcs_sub = lcs_sub

        avg_score = total_score / num_a
        max_lcs_across_a = int(lcs_matrix[bi].max())

        site_scores.append({
            "label": b_ctx["label"],
            "brick_20": b_ctx["brick"],
            "cpg_idx": b_ctx["cpg_idx"],
            "context": b_ctx["context"],
            "max_sim": best_score,
            "avg_sim": avg_score,
            "best_a_brick": best_a,
            "max_lcs": best_lcs_len,
            "lcs_sub": best_lcs_sub,
            "max_lcs_overall": max_lcs_across_a,
        })

    # ── Sort by avg similarity (ascending = worst first) ──
    site_scores.sort(key=lambda x: x["avg_sim"])

    # ── Report ──
    print(f"\n{'=' * 90}")
    print("CpG SITE RANKING — WORST MATCHING TO LIBRARY A (likely discarded)")
    print(f"{'=' * 90}")
    print(f"  {'Rank':<6}{'Site':<16}{'CpG':<6}{'Avg Sim':<10}{'Max Sim':<10}"
          f"{'Max LCS':<10}{'LCS substr':<15}{'Context'}")
    print(f"  {'─' * 85}")
    for rank, s in enumerate(site_scores[:30], 1):
        print(f"  {rank:<6}{s['label']:<16}{s['cpg_idx']:<6}"
              f"{s['avg_sim']:<10.4f}{s['max_sim']:<10.4f}"
              f"{s['max_lcs']:<10}{s['lcs_sub']:<15}{s['context']}")

    print(f"\n{'=' * 90}")
    print("BOTTOM 16 SITES (most likely discarded)")
    print(f"{'=' * 90}")
    for rank, s in enumerate(site_scores[:16], 1):
        print(f"  {rank}. {s['label']} (Brick {s['brick_20']}, CpG at {s['cpg_idx']})"
              f"  avg_sim={s['avg_sim']:.4f}  max_lcs={s['max_lcs']}  "
              f"context={s['context']}")

    print(f"\n{'=' * 90}")
    print("TOP 16 SITES (best matching — definitely kept)")
    print(f"{'=' * 90}")
    for rank, s in enumerate(site_scores[-16:][::-1], 1):
        print(f"  {rank}. {s['label']} (Brick {s['brick_20']}, CpG at {s['cpg_idx']})"
              f"  avg_sim={s['avg_sim']:.4f}  max_lcs={s['max_lcs']}  "
              f"context={s['context']}")

    # ── Heatmap: avg similarity per B site ──
    plot_site_ranking(site_scores, output_path="cpg_site_ranking.png")

    # ── Heatmap: B sites × A bricks (LCS) ──
    plot_cross_heatmap(contexts_a, contexts_b, lcs_matrix, score_matrix,
                       output_path="cpg_cross_heatmap.png")

    return site_scores, score_matrix, lcs_matrix


def plot_site_ranking(site_scores, output_path="cpg_site_ranking.png"):
    """Bar chart of all B sites ranked by average similarity, bottom 16 highlighted."""
    n = len(site_scores)
    avg_sims = [s["avg_sim"] for s in site_scores]
    labels = [s["label"] for s in site_scores]

    fig_h = max(8, n * 0.15)
    fig, ax = plt.subplots(figsize=(12, min(fig_h, 60)))

    max_val = max(avg_sims) if avg_sims else 1
    norm = mcolors.Normalize(vmin=0, vmax=max_val)
    cmap = plt.cm.YlOrRd

    colors = []
    for i in range(n):
        if i < 16:
            colors.append("#D32F2F")  # red for bottom 16
        else:
            colors.append(cmap(norm(avg_sims[i])))

    ax.barh(range(n), avg_sims, color=colors, edgecolor="#555", linewidth=0.3)
    ax.set_yticks(range(n))
    fs = max(2.5, 5 - n // 80)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.invert_yaxis()
    ax.set_xlabel("Average CpG context similarity to Library A")
    ax.set_title("CpG Site Ranking — Red = Bottom 16 (likely discarded)",
                 fontsize=11, fontweight="bold")
    ax.axvline(x=avg_sims[15] if n > 15 else 0, color="red",
               linestyle="--", linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSite ranking chart saved to: {output_path}")


def plot_cross_heatmap(contexts_a, contexts_b, lcs_matrix, score_matrix,
                       output_path="cpg_cross_heatmap.png"):
    """Heatmap: B CpG sites (rows) × A bricks (cols), coloured by LCS length."""
    nb, na = lcs_matrix.shape
    max_len = int(lcs_matrix.max()) or 1

    cell_w = max(0.25, 14 / na)
    cell_h = max(0.06, 25 / nb)
    fig_w = min(na * cell_w + 3, 30)
    fig_h = min(nb * cell_h + 2, 60)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.cm.YlOrRd.copy()
    im = ax.imshow(lcs_matrix, cmap=cmap, vmin=0, vmax=max_len,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(range(na))
    ax.set_yticks(range(nb))
    x_labels = [f"A{c['brick']}" for c in contexts_a]
    y_labels = [c["label"] for c in contexts_b]
    tick_fs = max(2, 5 - max(na, nb) // 60)
    ax.set_xticklabels(x_labels, fontsize=tick_fs, rotation=90)
    ax.set_yticklabels(y_labels, fontsize=tick_fs)
    ax.set_xlabel("Library A bricks (24-nt)", fontsize=9)
    ax.set_ylabel("Library B CpG sites (20-nt)", fontsize=9)
    ax.set_title("CpG Context LCS: Library B sites × Library A bricks",
                 fontsize=11, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.3, pad=0.02, label="LCS length")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Cross-heatmap saved to: {output_path}")


# ──────────────────────────────────────────────
#  FILL IN YOUR SEQUENCES
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # Library A: 38 bricks, 24-nt each, CpG at index 11
    bricks_24 = [
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
        "ACTAACTTATTCGATGCGGCCGAGCTCAAGACCTACGTGGAGGTCTAT",
    ]

    # Library B: 165 bricks, 20-nt each, CpG at index 4 and 14
    # PASTE YOUR 165 SEQUENCES HERE
    bricks_20 = [
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

    if not bricks_20:
        print("Please paste your 165 brick sequences into the bricks_20 list.")
        print("Each brick should be 20-nt with CpG sites at index 4 and 14.")
    else:
        site_scores, score_matrix, lcs_matrix = run_cpg_context_analysis(
            bricks_24=bricks_24,
            bricks_20=bricks_20,
            cpg_index_24=11,
            cpg_indices_20=[4, 14],
            window=8,
            k_range=range(5, 9),
        )
