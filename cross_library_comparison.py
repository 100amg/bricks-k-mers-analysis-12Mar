"""
Cross-Library Brick Comparison
================================
1. Full-sequence k-mer overlap (sense, k=5–10)
2. Combined pairwise LCS with CpG site overlap detection
3. Cross-library k-mer enrichment

Library A: 38 bricks, 24-nt, CpG at index 11
Library B: 175 bricks, 20-nt, CpG at indices 4 and 14
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter, defaultdict
from itertools import product


def b_index_to_label(b_idx, bricks_per_carrier=35):
    """Convert B-index (0-174) to carrier label like L1-12, L3-7."""
    carrier = b_idx // bricks_per_carrier + 1
    brick = b_idx % bricks_per_carrier + 1
    return f"L{carrier}-{brick}"


# ═══════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════

def generate_kmers(sequence: str, k: int) -> list[str]:
    seq = sequence.upper()
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def longest_common_substring(s1: str, s2: str) -> tuple[str, int, int, int]:
    """Returns (substring, length, start_in_s1, start_in_s2)."""
    s1, s2 = s1.upper(), s2.upper()
    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    best_len = 0
    best_end_s1 = 0
    best_end_s2 = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_end_s1 = i
                    best_end_s2 = j
        prev = curr
    start_s1 = best_end_s1 - best_len
    start_s2 = best_end_s2 - best_len
    return s1[start_s1:best_end_s1], best_len, start_s1, start_s2


def lcs_overlaps_cpg(start: int, length: int, cpg_indices: list[int]) -> list[int]:
    """Check if the LCS span [start, start+length) overlaps any CpG site (2 bases)."""
    lcs_range = set(range(start, start + length))
    overlapping = []
    for cpg in cpg_indices:
        if cpg in lcs_range or (cpg + 1) in lcs_range:
            overlapping.append(cpg)
    return overlapping


# ═══════════════════════════════════════════════
#  ANALYSIS 1: Full-sequence k-mer overlap
# ═══════════════════════════════════════════════

def full_kmer_overlap(bricks_a, bricks_b, k_range=range(5, 11)):
    """
    For each k, build k-mer sets for both libraries and find shared k-mers.
    Report which brick pairs share k-mers.
    """
    print(f"\n{'=' * 80}")
    print("ANALYSIS 1: FULL-SEQUENCE K-MER OVERLAP (SENSE)")
    print(f"{'=' * 80}")

    for k in k_range:
        # Build index: kmer -> list of (library, brick_idx)
        kmer_index = defaultdict(list)
        kmers_a_all = set()
        kmers_b_all = set()

        for i, seq in enumerate(bricks_a):
            for km in generate_kmers(seq, k):
                kmer_index[km].append(("A", i))
                kmers_a_all.add(km)

        for i, seq in enumerate(bricks_b):
            for km in generate_kmers(seq, k):
                kmer_index[km].append(("B", i))
                kmers_b_all.add(km)

        shared = kmers_a_all & kmers_b_all

        # Find cross-library pairs
        cross_pairs = defaultdict(set)  # (a_idx, b_idx) -> set of shared kmers
        for km in shared:
            a_bricks = {idx for lib, idx in kmer_index[km] if lib == "A"}
            b_bricks = {idx for lib, idx in kmer_index[km] if lib == "B"}
            for ai in a_bricks:
                for bi in b_bricks:
                    cross_pairs[(ai, bi)].add(km)

        print(f"\n  k={k}:")
        print(f"    Library A unique k-mers: {len(kmers_a_all)}")
        print(f"    Library B unique k-mers: {len(kmers_b_all)}")
        print(f"    Shared k-mers: {len(shared)}")
        print(f"    Cross-library brick pairs with shared k-mers: {len(cross_pairs)}")

        # Top pairs by number of shared k-mers
        if cross_pairs:
            ranked = sorted(cross_pairs.items(), key=lambda x: len(x[1]), reverse=True)
            print(f"\n    Top 10 most overlapping pairs:")
            print(f"    {'A brick':<10}{'B brick':<10}{'Carrier':<10}{'# shared':<10}{'Shared k-mers (first 5)'}")
            print(f"    {'─' * 75}")
            for (ai, bi), kmset in ranked[:10]:
                km_str = ", ".join(sorted(kmset)[:5])
                if len(kmset) > 5:
                    km_str += f" (+{len(kmset) - 5} more)"
                label = b_index_to_label(bi)
                print(f"    A{ai:<9}B{bi:<9}{label:<10}{len(kmset):<10}{km_str}")

    return cross_pairs  # returns last k's pairs


# ═══════════════════════════════════════════════
#  ANALYSIS 2: Pairwise LCS with CpG overlap
# ═══════════════════════════════════════════════

def pairwise_lcs_with_cpg(bricks_a, bricks_b,
                           cpg_a=11, cpg_b_list=[4, 14],
                           output_path="cross_lcs_cpg_heatmap.png"):
    """
    Compute LCS between every (A, B) pair.
    For each pair, report whether the LCS overlaps any CpG site.
    """
    print(f"\n{'=' * 80}")
    print("ANALYSIS 2: PAIRWISE LCS WITH CpG OVERLAP DETECTION")
    print(f"{'=' * 80}")

    na, nb = len(bricks_a), len(bricks_b)
    lcs_lengths = np.zeros((na, nb), dtype=int)
    lcs_data = [[None] * nb for _ in range(na)]

    total = na * nb
    for i, seq_a in enumerate(bricks_a):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing A{i} / {na}...")
        for j, seq_b in enumerate(bricks_b):
            sub, length, start_a, start_b = longest_common_substring(seq_a, seq_b)
            lcs_lengths[i][j] = length

            # Check CpG overlap
            cpg_overlap_a = lcs_overlaps_cpg(start_a, length, [cpg_a])
            cpg_overlap_b = lcs_overlaps_cpg(start_b, length, cpg_b_list)

            lcs_data[i][j] = {
                "sub": sub, "len": length,
                "start_a": start_a, "start_b": start_b,
                "cpg_overlap_a": cpg_overlap_a,
                "cpg_overlap_b": cpg_overlap_b,
                "any_cpg": bool(cpg_overlap_a or cpg_overlap_b),
            }

    # ── Report top pairs ──
    pairs = []
    for i in range(na):
        for j in range(nb):
            d = lcs_data[i][j]
            pairs.append((d["len"], d["sub"], i, j, d))
    pairs.sort(reverse=True)

    print(f"\n  {'Rank':<6}{'Brick':<10}{'Pair':<16}{'LCS':<8}{'Substring':<16}"
          f"{'posA':<8}{'posB':<8}{'CpG_A(11)':<12}{'CpG_B(4,14)':<14}")
    print(f"  {'─' * 100}")
    for rank, (l, sub, ai, bj, d) in enumerate(pairs[:40], 1):
        cpg_a_str = f"YES({','.join(str(c) for c in d['cpg_overlap_a'])})" if d["cpg_overlap_a"] else "-"
        cpg_b_str = f"YES({','.join(str(c) for c in d['cpg_overlap_b'])})" if d["cpg_overlap_b"] else "-"
        label = b_index_to_label(bj)
        print(f"  {rank:<6}{label:<10}A{ai} × B{bj:<8}{l:<8}{sub:<16}"
              f"{d['start_a']:<8}{d['start_b']:<8}{cpg_a_str:<12}{cpg_b_str:<14}")

    # ── Stats: how often does LCS overlap a CpG? ──
    cpg_overlap_count = sum(1 for p in pairs if p[4]["any_cpg"])
    cpg_a_count = sum(1 for p in pairs if p[4]["cpg_overlap_a"])
    cpg_b4_count = sum(1 for p in pairs if 4 in p[4]["cpg_overlap_b"])
    cpg_b14_count = sum(1 for p in pairs if 14 in p[4]["cpg_overlap_b"])

    print(f"\n  CpG overlap statistics (across all {na * nb} pairs):")
    print(f"    LCS overlaps ANY CpG:     {cpg_overlap_count} ({cpg_overlap_count/(na*nb)*100:.1f}%)")
    print(f"    LCS overlaps A CpG(11):   {cpg_a_count} ({cpg_a_count/(na*nb)*100:.1f}%)")
    print(f"    LCS overlaps B CpG(4):    {cpg_b4_count} ({cpg_b4_count/(na*nb)*100:.1f}%)")
    print(f"    LCS overlaps B CpG(14):   {cpg_b14_count} ({cpg_b14_count/(na*nb)*100:.1f}%)")

    # ── For pairs with LCS >= 7, breakdown ──
    long_pairs = [p for p in pairs if p[0] >= 7]
    if long_pairs:
        print(f"\n  Among {len(long_pairs)} pairs with LCS >= 7:")
        lp_cpg = sum(1 for p in long_pairs if p[4]["any_cpg"])
        print(f"    Overlapping any CpG: {lp_cpg} ({lp_cpg/len(long_pairs)*100:.1f}%)")

    # ── Heatmap ──
    max_len = int(lcs_lengths.max()) or 1
    cell_w = max(0.08, 14 / nb)
    cell_h = max(0.25, 10 / na)
    fig_w = min(nb * cell_w + 3, 40)
    fig_h = min(na * cell_h + 2, 18)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.cm.YlOrRd.copy()
    im = ax.imshow(lcs_lengths, cmap=cmap, vmin=0, vmax=max_len,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(range(nb))
    ax.set_yticks(range(na))
    tick_fs = max(2, 5 - max(na, nb) // 50)
    ax.set_xticklabels([f"B{j}" for j in range(nb)], fontsize=tick_fs, rotation=90)
    ax.set_yticklabels([f"A{i}" for i in range(na)], fontsize=tick_fs)
    ax.set_xlabel("175 Bricks (20-nt)", fontsize=9)
    ax.set_ylabel("38 Bricks (24-nt)", fontsize=9)
    ax.set_title("Cross-Library LCS (full sequence)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02, label="LCS length")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Heatmap saved to: {output_path}")

    return lcs_lengths, lcs_data, pairs


# ═══════════════════════════════════════════════
#  ANALYSIS 3: Cross-library k-mer enrichment
# ═══════════════════════════════════════════════

def kmer_enrichment(bricks_a, bricks_b, k_range=range(5, 9),
                    output_path="kmer_enrichment.png"):
    """
    For each k, find k-mers enriched in both libraries vs unique to one.
    """
    print(f"\n{'=' * 80}")
    print("ANALYSIS 3: CROSS-LIBRARY K-MER ENRICHMENT")
    print(f"{'=' * 80}")

    enrichment_data = {}

    for k in k_range:
        # Count k-mer frequencies in each library
        freq_a = Counter()
        freq_b = Counter()
        for seq in bricks_a:
            freq_a.update(generate_kmers(seq, k))
        for seq in bricks_b:
            freq_b.update(generate_kmers(seq, k))

        all_kmers = set(freq_a.keys()) | set(freq_b.keys())
        shared = set(freq_a.keys()) & set(freq_b.keys())
        only_a = set(freq_a.keys()) - set(freq_b.keys())
        only_b = set(freq_b.keys()) - set(freq_a.keys())

        # Enrichment: k-mers that are frequent in BOTH
        shared_by_combined_freq = []
        for km in shared:
            shared_by_combined_freq.append((freq_a[km] + freq_b[km], freq_a[km], freq_b[km], km))
        shared_by_combined_freq.sort(reverse=True)

        enrichment_data[k] = {
            "total": len(all_kmers),
            "shared": len(shared),
            "only_a": len(only_a),
            "only_b": len(only_b),
            "top_shared": shared_by_combined_freq[:20],
            "freq_a": freq_a,
            "freq_b": freq_b,
        }

        print(f"\n  k={k}:")
        print(f"    Total unique k-mers:   {len(all_kmers)}")
        print(f"    Shared (both libs):    {len(shared)} ({len(shared)/len(all_kmers)*100:.1f}%)")
        print(f"    Only in A (24-nt):     {len(only_a)}")
        print(f"    Only in B (20-nt):     {len(only_b)}")

        if shared_by_combined_freq:
            print(f"\n    Top 15 shared k-mers by combined frequency:")
            print(f"    {'K-mer':<15}{'Freq A':<10}{'Freq B':<10}{'Combined'}")
            print(f"    {'─' * 45}")
            for combined, fa, fb, km in shared_by_combined_freq[:15]:
                print(f"    {km:<15}{fa:<10}{fb:<10}{combined}")

    # ── Venn-style bar chart ──
    ks = sorted(enrichment_data.keys())
    shared_counts = [enrichment_data[k]["shared"] for k in ks]
    only_a_counts = [enrichment_data[k]["only_a"] for k in ks]
    only_b_counts = [enrichment_data[k]["only_b"] for k in ks]

    x = np.arange(len(ks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, only_a_counts, width, label="Only in 24-nt bricks", color="#4A90D9")
    ax.bar(x, shared_counts, width, label="Shared (both libraries)", color="#E8913A")
    ax.bar(x + width, only_b_counts, width, label="Only in 20-nt bricks", color="#50B86C")

    for i, k in enumerate(ks):
        ax.text(i - width, only_a_counts[i] + 2, str(only_a_counts[i]),
                ha="center", fontsize=7, color="#4A90D9")
        ax.text(i, shared_counts[i] + 2, str(shared_counts[i]),
                ha="center", fontsize=7, color="#E8913A")
        ax.text(i + width, only_b_counts[i] + 2, str(only_b_counts[i]),
                ha="center", fontsize=7, color="#50B86C")

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("K-mer length")
    ax.set_ylabel("Number of unique k-mers")
    ax.set_title("Cross-Library K-mer Enrichment")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Enrichment chart saved to: {output_path}")

    # ── Shared k-mers containing CG dinucleotide ──
    print(f"\n  Shared k-mers containing 'CG' (potential CpG-context matches):")
    for k in ks:
        cg_shared = [km for km, _ in
                     [(km, None) for _, _, _, km in enrichment_data[k]["top_shared"]]
                     if "CG" in km]
        all_cg_shared = [km for km in
                         (set(enrichment_data[k]["freq_a"]) & set(enrichment_data[k]["freq_b"]))
                         if "CG" in km]
        print(f"    k={k}: {len(all_cg_shared)} shared k-mers contain 'CG'")

    return enrichment_data


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════

if __name__ == "__main__":

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

    # Library B: 175 bricks (20-nt), ordered L1-1..L1-35, L2-1..L2-35, etc.
    # PASTE YOUR 175 SEQUENCES HERE
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
        print("Please paste your 175 brick sequences into the bricks_20 list.")
        print("Order: L1-1..L1-35, L2-1..L2-35, L3-1..L3-35, L4-1..L4-35, L5-1..L5-35")
    else:
        print(f"Library A: {len(bricks_24)} bricks (24-nt)")
        print(f"Library B: {len(bricks_20)} bricks (20-nt)")

        # Analysis 1
        cross_pairs = full_kmer_overlap(bricks_24, bricks_20, k_range=range(5, 11))

        # Analysis 2
        lcs_lengths, lcs_data, lcs_pairs = pairwise_lcs_with_cpg(
            bricks_24, bricks_20,
            cpg_a=11, cpg_b_list=[4, 14],
            output_path="cross_lcs_cpg_heatmap.png",
        )

        # Analysis 3
        enrichment = kmer_enrichment(bricks_24, bricks_20, k_range=range(5, 9),
                                     output_path="kmer_enrichment.png")
