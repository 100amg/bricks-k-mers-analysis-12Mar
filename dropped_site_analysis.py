"""
Dropped Site Analysis
======================
1. Check if the 16 dropped CpG sites overlap heavily with the 38 AHEAD bricks
2. Analyze intrinsic sequence features of dropped vs kept sites to find
   what makes the 16 different

Library A: 38 bricks, 24-nt, CpG at index 11
Library B: 175 bricks, 20-nt, CpG at indices 4 and 14
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════
#  DROPPED SITE DEFINITIONS
# ═══════════════════════════════════════════════

# (B-index, CpG position) for each dropped site
DROPPED_SITES = [
    (11,  4,  "L1-12"),
    (12,  14, "L1-13"),
    (13,  4,  "L1-14"),
    (17,  4,  "L1-18"),
    (34,  4,  "L1-35"),
    (43,  4,  "L2-9"),
    (68,  14, "L2-34"),
    (70,  14, "L3-1"),
    (73,  4,  "L3-4"),
    (76,  4,  "L3-7"),
    (83,  4,  "L3-14"),
    (87,  14, "L3-18"),
    (88,  4,  "L3-19"),
    (111, 4,  "L4-7"),
    (162, 4,  "L5-23"),
    (173, 4,  "L5-34"),
]

DROPPED_SET = {(b, c) for b, c, _ in DROPPED_SITES}


# ═══════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════

def gc_content(seq):
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq) * 100

def longest_homopolymer(seq):
    seq = seq.upper()
    max_run = 1
    current = 1
    max_base = seq[0]
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            if current > max_run:
                max_run = current
                max_base = seq[i]
        else:
            current = 1
    return max_run, max_base

def count_cpg_dinucleotides(seq):
    seq = seq.upper()
    return seq.count("CG")

def self_complementarity_score(seq):
    """Simple self-complement score: count matching bases between seq and its reverse complement."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    seq = seq.upper()
    rc = "".join(comp.get(b, "N") for b in reversed(seq))
    matches = sum(1 for a, b in zip(seq, rc) if a == b)
    return matches / len(seq) * 100

def flanking_gc(seq, cpg_idx, window=3):
    """GC content in the window immediately flanking the CpG (excluding CpG itself)."""
    seq = seq.upper()
    left_start = max(0, cpg_idx - window)
    right_end = min(len(seq), cpg_idx + 2 + window)
    flanking = seq[left_start:cpg_idx] + seq[cpg_idx + 2:right_end]
    if len(flanking) == 0:
        return 0
    return (flanking.count("G") + flanking.count("C")) / len(flanking) * 100

def bases_to_edge(seq, cpg_idx):
    """Distance from CpG to nearest sequence edge."""
    return min(cpg_idx, len(seq) - cpg_idx - 2)

def local_context(seq, cpg_idx, window=4):
    """Extract the immediate context around CpG."""
    seq = seq.upper()
    start = max(0, cpg_idx - window)
    end = min(len(seq), cpg_idx + 2 + window)
    return seq[start:end]

def longest_common_substring(s1, s2):
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


# ═══════════════════════════════════════════════
#  PART 1: Overlap of dropped sites with 38 bricks
# ═══════════════════════════════════════════════

def check_dropped_overlap(bricks_a, bricks_b):
    print("=" * 85)
    print("PART 1: DO THE 16 DROPPED SITES OVERLAP HEAVILY WITH THE 38 AHEAD BRICKS?")
    print("=" * 85)

    results = []
    for b_idx, cpg_pos, label in DROPPED_SITES:
        seq_b = bricks_b[b_idx]
        best_lcs = 0
        best_a = -1
        best_sub = ""
        total_lcs = 0

        for a_idx, seq_a in enumerate(bricks_a):
            sub, length = longest_common_substring(seq_a, seq_b)
            total_lcs += length
            if length > best_lcs:
                best_lcs = length
                best_a = a_idx
                best_sub = sub

        avg_lcs = total_lcs / len(bricks_a)
        results.append({
            "label": label, "b_idx": b_idx, "cpg": cpg_pos,
            "seq": seq_b, "best_lcs": best_lcs, "best_a": best_a,
            "best_sub": best_sub, "avg_lcs": avg_lcs,
        })

    # Also compute average for ALL kept sites for comparison
    kept_best_lcs = []
    for b_idx in range(len(bricks_b)):
        for cpg_pos in [4, 14]:
            if (b_idx, cpg_pos) in DROPPED_SET:
                continue
            seq_b = bricks_b[b_idx]
            best = 0
            for seq_a in bricks_a:
                _, length = longest_common_substring(seq_a, seq_b)
                if length > best:
                    best = length
            kept_best_lcs.append(best)

    dropped_best_lcs = [r["best_lcs"] for r in results]

    print(f"\n  {'Label':<12}{'CpG':<6}{'Best LCS':<10}{'Best A':<10}{'Substring':<16}{'Avg LCS':<10}")
    print(f"  {'─' * 70}")
    for r in sorted(results, key=lambda x: x["best_lcs"], reverse=True):
        print(f"  {r['label']:<12}{r['cpg']:<6}{r['best_lcs']:<10}A{r['best_a']:<9}"
              f"{r['best_sub']:<16}{r['avg_lcs']:<10.2f}")

    print(f"\n  Summary:")
    print(f"    Dropped sites — mean best LCS to any A brick:  {np.mean(dropped_best_lcs):.2f}")
    print(f"    Kept sites    — mean best LCS to any A brick:  {np.mean(kept_best_lcs):.2f}")
    print(f"    Dropped sites — max best LCS:                  {max(dropped_best_lcs)}")
    print(f"    Kept sites    — max best LCS:                  {max(kept_best_lcs)}")
    print(f"\n  → {'YES' if np.mean(dropped_best_lcs) > np.mean(kept_best_lcs) else 'NO'}: "
          f"dropped sites do {'NOT ' if np.mean(dropped_best_lcs) <= np.mean(kept_best_lcs) else ''}"
          f"have higher overlap with AHEAD bricks than kept sites.")


# ═══════════════════════════════════════════════
#  PART 2: Intrinsic feature analysis
# ═══════════════════════════════════════════════

def analyze_intrinsic_features(bricks_b):
    print(f"\n\n{'=' * 85}")
    print("PART 2: WHAT MAKES THE 16 DROPPED SITES DIFFERENT?")
    print("=" * 85)

    # Collect features for all 350 sites
    all_sites = []
    for b_idx in range(len(bricks_b)):
        seq = bricks_b[b_idx].upper()
        for cpg_pos in [4, 14]:
            is_dropped = (b_idx, cpg_pos) in DROPPED_SET
            drop_label = ""
            if is_dropped:
                for bi, ci, li in DROPPED_SITES:
                    if bi == b_idx and ci == cpg_pos:
                        drop_label = li
                        break

            hp_len, hp_base = longest_homopolymer(seq)

            site = {
                "b_idx": b_idx,
                "cpg_pos": cpg_pos,
                "dropped": is_dropped,
                "label": drop_label,
                "seq": seq,
                "gc_full": gc_content(seq),
                "gc_flank_3": flanking_gc(seq, cpg_pos, window=3),
                "gc_flank_5": flanking_gc(seq, cpg_pos, window=5),
                "homopolymer_len": hp_len,
                "homopolymer_base": hp_base,
                "num_cpg": count_cpg_dinucleotides(seq),
                "self_comp": self_complementarity_score(seq),
                "edge_dist": bases_to_edge(seq, cpg_pos),
                "context_4": local_context(seq, cpg_pos, 4),
                "base_before_cpg": seq[cpg_pos - 1] if cpg_pos > 0 else "-",
                "base_after_cpg": seq[cpg_pos + 2] if cpg_pos + 2 < len(seq) else "-",
                "dinuc_before": seq[cpg_pos - 2:cpg_pos] if cpg_pos >= 2 else "-",
                "dinuc_after": seq[cpg_pos + 2:cpg_pos + 4] if cpg_pos + 4 <= len(seq) else "-",
            }
            all_sites.append(site)

    dropped = [s for s in all_sites if s["dropped"]]
    kept = [s for s in all_sites if not s["dropped"]]

    # ── Feature comparison ──
    features = [
        ("GC content (full brick)", "gc_full", "%"),
        ("GC flanking CpG (±3bp)", "gc_flank_3", "%"),
        ("GC flanking CpG (±5bp)", "gc_flank_5", "%"),
        ("Longest homopolymer", "homopolymer_len", "bp"),
        ("# CpG dinucleotides in brick", "num_cpg", ""),
        ("Self-complementarity", "self_comp", "%"),
        ("Distance CpG to nearest edge", "edge_dist", "bp"),
    ]

    print(f"\n  {'Feature':<35}{'Dropped (n=16)':<20}{'Kept (n=334)':<20}{'Difference'}")
    print(f"  {'─' * 90}")
    for name, key, unit in features:
        d_vals = [s[key] for s in dropped]
        k_vals = [s[key] for s in kept]
        d_mean = np.mean(d_vals)
        k_mean = np.mean(k_vals)
        diff = d_mean - k_mean
        sig = "***" if abs(diff) > 1.5 * np.std(k_vals) else "**" if abs(diff) > np.std(k_vals) else "*" if abs(diff) > 0.5 * np.std(k_vals) else ""
        print(f"  {name:<35}{d_mean:>8.2f} {unit:<8}{k_mean:>8.2f} {unit:<8}{diff:>+8.2f} {sig}")

    # ── CpG position breakdown ──
    dropped_cpg4 = sum(1 for s in dropped if s["cpg_pos"] == 4)
    dropped_cpg14 = sum(1 for s in dropped if s["cpg_pos"] == 14)
    kept_cpg4 = sum(1 for s in kept if s["cpg_pos"] == 4)
    kept_cpg14 = sum(1 for s in kept if s["cpg_pos"] == 14)

    print(f"\n  CpG POSITION BIAS:")
    print(f"  {'─' * 50}")
    print(f"    Dropped at CpG4:  {dropped_cpg4}/16  ({dropped_cpg4/16*100:.0f}%)")
    print(f"    Dropped at CpG14: {dropped_cpg14}/16 ({dropped_cpg14/16*100:.0f}%)")
    print(f"    Kept at CpG4:     {kept_cpg4}/334 ({kept_cpg4/334*100:.1f}%)")
    print(f"    Kept at CpG14:    {kept_cpg14}/334 ({kept_cpg14/334*100:.1f}%)")
    drop_rate_4 = dropped_cpg4 / (dropped_cpg4 + kept_cpg4) * 100
    drop_rate_14 = dropped_cpg14 / (dropped_cpg14 + kept_cpg14) * 100
    print(f"\n    Drop rate for CpG4 sites:  {drop_rate_4:.1f}%")
    print(f"    Drop rate for CpG14 sites: {drop_rate_14:.1f}%")
    print(f"    CpG4 is {drop_rate_4/drop_rate_14:.1f}x more likely to be dropped than CpG14")

    # ── Edge distance analysis ──
    print(f"\n  EDGE DISTANCE ANALYSIS:")
    print(f"  {'─' * 50}")
    print(f"    CpG4:  distance to 5' end = 4 bases")
    print(f"    CpG14: distance to 5' end = 14 bases, distance to 3' end = {20 - 14 - 2} bases")
    print(f"    → CpG4 is always only 4bp from the 5' edge")
    print(f"    → CpG14 is 4bp from the 3' edge")
    print(f"    Both are close to an edge, but CpG4 drops 3x more often")

    # ── Flanking base analysis ──
    print(f"\n  FLANKING BASE FREQUENCIES (base immediately before CpG):")
    print(f"  {'─' * 50}")
    for group_name, group in [("Dropped", dropped), ("Kept", kept)]:
        before = Counter(s["base_before_cpg"] for s in group)
        total = sum(before.values())
        print(f"    {group_name}: ", end="")
        for base in "ACGT":
            print(f"{base}={before.get(base, 0)} ({before.get(base, 0)/total*100:.0f}%) ", end="")
        print()

    print(f"\n  FLANKING BASE FREQUENCIES (base immediately after CpG):")
    print(f"  {'─' * 50}")
    for group_name, group in [("Dropped", dropped), ("Kept", kept)]:
        after = Counter(s["base_after_cpg"] for s in group)
        total = sum(after.values())
        print(f"    {group_name}: ", end="")
        for base in "ACGT":
            print(f"{base}={after.get(base, 0)} ({before.get(base, 0)/total*100:.0f}%) ", end="")
        print()

    # ── Dinucleotide context ──
    print(f"\n  DINUCLEOTIDE CONTEXT (2bp before CpG → CG → 2bp after):")
    print(f"  {'─' * 50}")
    print(f"    Dropped sites:")
    for s in dropped:
        print(f"      {s['label']:<12} CpG{s['cpg_pos']:<4} "
              f"..{s['dinuc_before']}[CG]{s['dinuc_after']}..  "
              f"full context: {s['context_4']}")

    # ── Specific motif check: DNMT1 recognition ──
    # DNMT1 prefers hemimethylated CpG in context, but can be affected by
    # flanking sequence. Check for known difficult contexts.
    print(f"\n  DNMT1 CONTEXT ANALYSIS:")
    print(f"  {'─' * 50}")
    print(f"  (DNMT1 methylates hemimethylated CpG. Context matters for efficiency.)")

    # Check trinucleotide context XCG and CGY
    print(f"\n    Trinucleotide XCG (base before CG):")
    for group_name, group in [("Dropped", dropped), ("Kept", kept)]:
        tri_before = Counter()
        for s in group:
            if s["cpg_pos"] > 0:
                tri = s["seq"][s["cpg_pos"] - 1:s["cpg_pos"] + 2]
                tri_before[tri] += 1
        total = sum(tri_before.values())
        print(f"      {group_name}: ", end="")
        for tri, cnt in tri_before.most_common():
            print(f"{tri}={cnt}({cnt/total*100:.0f}%) ", end="")
        print()

    print(f"\n    Trinucleotide CGY (base after CG):")
    for group_name, group in [("Dropped", dropped), ("Kept", kept)]:
        tri_after = Counter()
        for s in group:
            if s["cpg_pos"] + 2 < len(s["seq"]):
                tri = s["seq"][s["cpg_pos"]:s["cpg_pos"] + 3]
                tri_after[tri] += 1
        total = sum(tri_after.values())
        print(f"      {group_name}: ", end="")
        for tri, cnt in tri_after.most_common():
            print(f"{tri}={cnt}({cnt/total*100:.0f}%) ", end="")
        print()

    # ── Carrier distribution ──
    print(f"\n  CARRIER DISTRIBUTION OF DROPS:")
    print(f"  {'─' * 50}")
    carrier_counts = Counter()
    for _, _, label in DROPPED_SITES:
        carrier = label.split("-")[0]  # L1, L2, etc.
        carrier_counts[carrier] += 1
    for carrier in ["L1", "L2", "L3", "L4", "L5"]:
        bar = "█" * carrier_counts.get(carrier, 0)
        print(f"    {carrier}: {carrier_counts.get(carrier, 0):>2} drops  {bar}")
    print(f"    → L3 has the most drops ({carrier_counts['L3']}), L4 the fewest ({carrier_counts['L4']})")

    # ── Homopolymer detail ──
    print(f"\n  HOMOPOLYMER RUNS IN DROPPED vs KEPT:")
    print(f"  {'─' * 50}")
    for group_name, group in [("Dropped", dropped), ("Kept", kept)]:
        runs = [s["homopolymer_len"] for s in group]
        bases = Counter(s["homopolymer_base"] for s in group)
        print(f"    {group_name}: mean longest run = {np.mean(runs):.2f}bp, "
              f"max = {max(runs)}bp")
        print(f"      Homopolymer base: ", end="")
        for b, c in bases.most_common():
            print(f"{b}={c} ", end="")
        print()

    return all_sites, dropped, kept


# ═══════════════════════════════════════════════
#  PART 3: Visualization
# ═══════════════════════════════════════════════

def plot_feature_comparison(all_sites, dropped, kept):
    """Side-by-side feature distributions for dropped vs kept."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Dropped vs Kept CpG Sites — Feature Comparison",
                 fontsize=14, fontweight="bold")

    features = [
        ("GC content (%)", "gc_full"),
        ("GC flanking CpG ±3bp (%)", "gc_flank_3"),
        ("Longest homopolymer (bp)", "homopolymer_len"),
        ("# CpG dinucleotides", "num_cpg"),
        ("Self-complementarity (%)", "self_comp"),
        ("Distance to edge (bp)", "edge_dist"),
    ]

    for idx, (title, key) in enumerate(features):
        ax = axes[idx // 3][idx % 3]
        d_vals = [s[key] for s in dropped]
        k_vals = [s[key] for s in kept]

        bins = np.linspace(min(d_vals + k_vals), max(d_vals + k_vals), 15)
        ax.hist(k_vals, bins=bins, alpha=0.6, color="#4A90D9", label=f"Kept (n={len(kept)})", density=True)
        ax.hist(d_vals, bins=bins, alpha=0.7, color="#D32F2F", label=f"Dropped (n={len(dropped)})", density=True)
        ax.axvline(np.mean(d_vals), color="#D32F2F", linestyle="--", linewidth=1.5)
        ax.axvline(np.mean(k_vals), color="#4A90D9", linestyle="--", linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("dropped_vs_kept_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFeature comparison plot saved to: dropped_vs_kept_features.png")

    # ── CpG position pie chart ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("CpG Position Bias in Dropped Sites", fontsize=13, fontweight="bold")

    d4 = sum(1 for s in dropped if s["cpg_pos"] == 4)
    d14 = sum(1 for s in dropped if s["cpg_pos"] == 14)
    ax1.pie([d4, d14], labels=[f"CpG4\n({d4})", f"CpG14\n({d14})"],
            colors=["#D32F2F", "#FF8A80"], autopct="%1.0f%%", startangle=90)
    ax1.set_title("Dropped sites by CpG position")

    k4 = sum(1 for s in kept if s["cpg_pos"] == 4)
    k14 = sum(1 for s in kept if s["cpg_pos"] == 14)
    ax2.pie([k4, k14], labels=[f"CpG4\n({k4})", f"CpG14\n({k14})"],
            colors=["#4A90D9", "#90CAF9"], autopct="%1.0f%%", startangle=90)
    ax2.set_title("Kept sites by CpG position")

    plt.tight_layout()
    plt.savefig("cpg_position_bias.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"CpG position bias plot saved to: cpg_position_bias.png")


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

    # Library B: 175 bricks, 20-nt
    # PASTE YOUR 175 SEQUENCES HERE (same order as before)
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
    else:
        check_dropped_overlap(bricks_24, bricks_20)
        all_sites, dropped, kept = analyze_intrinsic_features(bricks_20)
        plot_feature_comparison(all_sites, dropped, kept)
