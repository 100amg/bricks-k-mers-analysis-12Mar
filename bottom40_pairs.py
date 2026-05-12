"""
Bottom 40 LCS Pairs
=====================
From all 38 × 175 = 6,650 pairwise comparisons,
find the 40 pairs with the SHORTEST longest common substring.
Check if dropped sites or problematic AHEAD bricks cluster here.
"""

import numpy as np
from collections import Counter


DROPPED_SITES = [
    (11, 4, "L1-12"), (12, 14, "L1-13"), (13, 4, "L1-14"),
    (17, 4, "L1-18"), (34, 4, "L1-35"), (43, 4, "L2-9"),
    (68, 14, "L2-34"), (70, 14, "L3-1"), (73, 4, "L3-4"),
    (76, 4, "L3-7"), (83, 4, "L3-14"), (87, 14, "L3-18"),
    (88, 4, "L3-19"), (111, 4, "L4-7"), (162, 4, "L5-23"),
    (173, 4, "L5-34"),
]
DROPPED_B_INDICES = {b for b, _, _ in DROPPED_SITES}

# AHEAD bricks that gave inconsistent results
PROBLEMATIC_A = {1, 7, 8, 10, 12, 14, 15, 17, 19, 21, 24, 27, 29}


def b_index_to_label(b_idx, bricks_per_carrier=35):
    carrier = b_idx // bricks_per_carrier + 1
    brick = b_idx % bricks_per_carrier + 1
    return f"L{carrier}-{brick}"


def longest_common_substring(s1, s2):
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


def lcs_overlaps_cpg(start, length, cpg_indices):
    lcs_range = set(range(start, start + length))
    overlapping = []
    for cpg in cpg_indices:
        if cpg in lcs_range or (cpg + 1) in lcs_range:
            overlapping.append(cpg)
    return overlapping


def run_analysis(bricks_24, bricks_20):
    na, nb = len(bricks_24), len(bricks_20)
    total = na * nb

    print(f"Computing all {total} pairwise LCS values...")

    all_pairs = []
    for ai in range(na):
        if (ai + 1) % 10 == 0 or ai == 0:
            print(f"  A{ai} / {na}...")
        for bi in range(nb):
            sub, length, start_a, start_b = longest_common_substring(
                bricks_24[ai], bricks_20[bi])

            cpg_a = lcs_overlaps_cpg(start_a, length, [11])
            cpg_b = lcs_overlaps_cpg(start_b, length, [4, 14])

            cpg_b_which = []
            if 4 in cpg_b:
                cpg_b_which.append("CpG4")
            if 14 in cpg_b:
                cpg_b_which.append("CpG14")

            all_pairs.append({
                "ai": ai, "bi": bi,
                "label_b": b_index_to_label(bi),
                "lcs": length, "sub": sub,
                "start_a": start_a, "start_b": start_b,
                "cpg_a": bool(cpg_a),
                "cpg_b": cpg_b_which,
                "b_dropped": bi in DROPPED_B_INDICES,
                "a_problematic": ai in PROBLEMATIC_A,
            })

    # Sort ascending by LCS
    all_pairs.sort(key=lambda x: (x["lcs"], x["sub"]))

    # ═══════════════════════════════════════════
    # BOTTOM 40 PAIRS
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print(f"BOTTOM 40 PAIRS (least similar out of {total} combinations)")
    print(f"{'=' * 120}")
    print(f"\n  {'Rank':<6}{'20-nt Brick':<14}{'24-nt Brick':<14}{'LCS':<6}"
          f"{'Substring':<12}{'CpG@12(24nt)':<14}{'CpG(20nt)':<14}"
          f"{'B dropped?':<14}{'A problematic?'}")
    print(f"  {'─' * 115}")

    for rank, p in enumerate(all_pairs[:60], 1):
        cpg_a_str = "YES" if p["cpg_a"] else "-"
        cpg_b_str = ",".join(p["cpg_b"]) if p["cpg_b"] else "-"
        b_drop = "⚠ DROPPED" if p["b_dropped"] else ""
        a_prob = "⚠ PROBLEMATIC" if p["a_problematic"] else ""
        print(f"  {rank:<6}{p['label_b']:<14}A{p['ai']:<13}{p['lcs']:<6}"
              f"{p['sub']:<12}{cpg_a_str:<14}{cpg_b_str:<14}"
              f"{b_drop:<14}{a_prob}")

    # ═══════════════════════════════════════════
    # TOP 40 PAIRS (for comparison)
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print(f"TOP 40 PAIRS (most similar out of {total} combinations)")
    print(f"{'=' * 120}")
    print(f"\n  {'Rank':<6}{'20-nt Brick':<14}{'24-nt Brick':<14}{'LCS':<6}"
          f"{'Substring':<14}{'CpG@12(24nt)':<14}{'CpG(20nt)':<14}"
          f"{'B dropped?':<14}{'A problematic?'}")
    print(f"  {'─' * 115}")

    for rank, p in enumerate(reversed(all_pairs[-40:]), 1):
        cpg_a_str = "YES" if p["cpg_a"] else "-"
        cpg_b_str = ",".join(p["cpg_b"]) if p["cpg_b"] else "-"
        b_drop = "⚠ DROPPED" if p["b_dropped"] else ""
        a_prob = "⚠ PROBLEMATIC" if p["a_problematic"] else ""
        print(f"  {rank:<6}{p['label_b']:<14}A{p['ai']:<13}{p['lcs']:<6}"
              f"{p['sub']:<14}{cpg_a_str:<14}{cpg_b_str:<14}"
              f"{b_drop:<14}{a_prob}")

    # ═══════════════════════════════════════════
    # ANALYSIS: Do dropped/problematic cluster?
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("CLUSTERING ANALYSIS")
    print(f"{'=' * 120}")

    bottom_40 = all_pairs[:40]
    top_40 = all_pairs[-40:]

    # Count flags
    b40_dropped = sum(1 for p in bottom_40 if p["b_dropped"])
    b40_prob_a = sum(1 for p in bottom_40 if p["a_problematic"])
    b40_both = sum(1 for p in bottom_40 if p["b_dropped"] and p["a_problematic"])

    t40_dropped = sum(1 for p in top_40 if p["b_dropped"])
    t40_prob_a = sum(1 for p in top_40 if p["a_problematic"])
    t40_both = sum(1 for p in top_40 if p["b_dropped"] and p["a_problematic"])

    # Expected rates
    exp_dropped = 16 / 175  # ~9.1%
    exp_prob_a = 13 / 38  # ~34.2%

    print(f"\n  {'Metric':<45}{'Bottom 40':<15}{'Top 40':<15}{'Expected'}")
    print(f"  {'─' * 85}")
    print(f"  {'Pairs involving a DROPPED 20-nt brick':<45}"
          f"{b40_dropped}/40{'':<7}{t40_dropped}/40{'':<7}{exp_dropped*40:.1f}/40")
    print(f"  {'Pairs involving a PROBLEMATIC 24-nt brick':<45}"
          f"{b40_prob_a}/40{'':<7}{t40_prob_a}/40{'':<7}{exp_prob_a*40:.1f}/40")
    print(f"  {'Pairs involving BOTH':<45}"
          f"{b40_both}/40{'':<7}{t40_both}/40{'':<7}{exp_dropped*exp_prob_a*40:.1f}/40")

    # ═══════════════════════════════════════════
    # LCS DISTRIBUTION OVERVIEW
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("LCS LENGTH DISTRIBUTION (all 6,650 pairs)")
    print(f"{'=' * 120}")

    lcs_vals = [p["lcs"] for p in all_pairs]
    print(f"\n  {'LCS':<8}{'Count':<12}{'Cumulative %':<15}{'Dropped pairs':<18}{'Problematic A pairs'}")
    print(f"  {'─' * 65}")
    cumul = 0
    for lcs_val in range(min(lcs_vals), max(lcs_vals) + 1):
        count = lcs_vals.count(lcs_val)
        cumul += count
        d_count = sum(1 for p in all_pairs if p["lcs"] == lcs_val and p["b_dropped"])
        a_count = sum(1 for p in all_pairs if p["lcs"] == lcs_val and p["a_problematic"])
        print(f"  {lcs_val:<8}{count:<12}{cumul/total*100:>6.1f}%{'':<7}"
              f"{d_count:<18}{a_count}")

    # ═══════════════════════════════════════════
    # WHICH A BRICKS APPEAR MOST IN BOTTOM 100?
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("WHICH 24-nt BRICKS APPEAR MOST IN THE BOTTOM 100 PAIRS?")
    print(f"{'=' * 120}")

    bottom_100 = all_pairs[:100]
    a_counts = Counter(p["ai"] for p in bottom_100)
    print(f"\n  {'A brick':<10}{'Count in bottom 100':<22}{'Problematic?'}")
    print(f"  {'─' * 45}")
    for ai, cnt in a_counts.most_common(15):
        prob = "⚠ YES" if ai in PROBLEMATIC_A else ""
        print(f"  A{ai:<9}{cnt:<22}{prob}")

    # Same for B bricks
    print(f"\n{'=' * 120}")
    print("WHICH 20-nt BRICKS APPEAR MOST IN THE BOTTOM 100 PAIRS?")
    print(f"{'=' * 120}")

    b_counts = Counter(p["bi"] for p in bottom_100)
    print(f"\n  {'B brick':<14}{'Count in bottom 100':<22}{'Dropped?'}")
    print(f"  {'─' * 50}")
    for bi, cnt in b_counts.most_common(15):
        drop = "⚠ DROPPED" if bi in DROPPED_B_INDICES else ""
        print(f"  {b_index_to_label(bi):<14}{cnt:<22}{drop}")

    return all_pairs


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
        print("Please paste your 175 brick sequences.")
    else:
        all_pairs = run_analysis(bricks_24, bricks_20)
