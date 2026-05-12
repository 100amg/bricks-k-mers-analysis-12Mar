"""
DNMT1 Flanking Sequence Bias — Based on Adam et al. 2020 (Nat Commun 11:3723)
================================================================================
Key findings from the paper:

MINUS SIDE (upstream of CpG):
  −1 position: G strongly DISFAVORED (forces noncanonical G-G base pair after
               target C flipping, requiring massive conformational rearrangement)
               A/T FAVORED at −1
  −2 position: C DISFAVORED (stabilizes problematic G-G pair at −1 via stacking
               with G at −2 on nontarget strand)
               Combined C(−2)G(−1) before CG is STRONGLY disfavored

PLUS SIDE (downstream of CpG):
  +1 position: G FAVORED (widens minor groove, stabilizes catalytic helix
               in kinked/active conformation)
  +2 position: T strongly FAVORED, A weakly FAVORED
               (R1241/Y1243 contacts with +1/+2 backbone stabilize straight→kinked
               catalytic helix transition needed for activity)

Explicit benchmarks from paper:
  TCCGTA → rank 2/256 (highly favored)
  TACGGA → rank 32/256 (favored)
  GGCGGC → rank 177/256 (disfavored)
  CGCGAG → rank 254/256 (strongly disfavored, couldn't even crystallize)

The paper reports ~100-fold difference between best and worst flanks.
"""

import numpy as np
from collections import Counter


# ═══════════════════════════════════════════════
#  DROPPED SITES
# ═══════════════════════════════════════════════
DROPPED_SITES = [
    (11,  4,  "L1-12"), (12,  14, "L1-13"), (13,  4,  "L1-14"),
    (17,  4,  "L1-18"), (34,  4,  "L1-35"), (43,  4,  "L2-9"),
    (68,  14, "L2-34"), (70,  14, "L3-1"),  (73,  4,  "L3-4"),
    (76,  4,  "L3-7"),  (83,  4,  "L3-14"), (87,  14, "L3-18"),
    (88,  4,  "L3-19"), (111, 4,  "L4-7"),  (162, 4,  "L5-23"),
    (173, 4,  "L5-34"),
]
DROPPED_SET = {(b, c) for b, c, _ in DROPPED_SITES}


# ═══════════════════════════════════════════════
#  DNMT1 SCORING (based on Adam et al. 2020)
# ═══════════════════════════════════════════════

# Per-position preference scores (higher = more favored by DNMT1)
# Derived from Fig 3e (Weblogo of favored/disfavored) and structural data

POS_SCORES = {
    # −2 position: C disfavored, others neutral-to-mild
    -2: {"A": 0.0, "T": 0.0, "G": -0.5, "C": -1.0},

    # −1 position: G strongly disfavored; A, T favored
    -1: {"A": +1.0, "T": +1.0, "G": -2.0, "C": 0.0},

    # +1 position: G favored
    +1: {"A": 0.0, "T": -0.5, "G": +1.5, "C": -0.5},

    # +2 position: T strongly favored, A mildly favored
    +2: {"A": +0.5, "T": +1.0, "G": -0.5, "C": -0.5},
}

# Combined disfavored patterns (additive penalties)
def compute_dnmt1_score(base_m2, base_m1, base_p1, base_p2):
    """
    Compute a DNMT1 preference score for an NNCGNN hexamer.
    Positive = favored, Negative = disfavored.
    Range approximately −4 to +4.
    """
    score = 0
    score += POS_SCORES[-2].get(base_m2, 0)
    score += POS_SCORES[-1].get(base_m1, 0)
    score += POS_SCORES[+1].get(base_p1, 0)
    score += POS_SCORES[+2].get(base_p2, 0)

    # Extra penalty for combined CG at −2/−1 (synergistic disfavor)
    if base_m2 == "C" and base_m1 == "G":
        score -= 1.0

    return score


def extract_context(seq, cpg_idx):
    """Extract flanking bases and hexamer around a CpG."""
    seq = seq.upper()
    m2 = seq[cpg_idx - 2] if cpg_idx >= 2 else "-"
    m1 = seq[cpg_idx - 1] if cpg_idx >= 1 else "-"
    p1 = seq[cpg_idx + 2] if cpg_idx + 2 < len(seq) else "-"
    p2 = seq[cpg_idx + 3] if cpg_idx + 3 < len(seq) else "-"
    hexamer = f"{m2}{m1}CG{p1}{p2}"
    return hexamer, m2, m1, p1, p2


def classify_position(pos_label, base, pos_scores):
    """Return a human-readable verdict for a single position."""
    score = pos_scores.get(base, 0)
    if score <= -1.5:
        return f"{base} ⚠⚠ STRONGLY DISFAVORED"
    elif score <= -0.5:
        return f"{base} ⚠ disfavored"
    elif score >= 1.0:
        return f"{base} ✓✓ FAVORED"
    elif score >= 0.5:
        return f"{base} ✓ mildly favored"
    else:
        return f"{base} — neutral"


# ═══════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════

def run_full_analysis(bricks_20):
    # Build all 350 sites
    all_sites = []
    for b_idx in range(len(bricks_20)):
        seq = bricks_20[b_idx].upper()
        for cpg_pos in [4, 14]:
            is_dropped = (b_idx, cpg_pos) in DROPPED_SET
            label = ""
            if is_dropped:
                for bi, ci, li in DROPPED_SITES:
                    if bi == b_idx and ci == cpg_pos:
                        label = li
                        break

            hexamer, m2, m1, p1, p2 = extract_context(seq, cpg_pos)
            score = compute_dnmt1_score(m2, m1, p1, p2)

            all_sites.append({
                "b_idx": b_idx, "cpg_pos": cpg_pos,
                "dropped": is_dropped, "label": label,
                "seq": seq, "hexamer": hexamer,
                "m2": m2, "m1": m1, "p1": p1, "p2": p2,
                "score": score,
            })

    dropped = [s for s in all_sites if s["dropped"]]
    kept = [s for s in all_sites if not s["dropped"]]

    # ═══════════════════════════════════════════
    # SECTION 1: Detailed per-site breakdown
    # ═══════════════════════════════════════════
    print("=" * 110)
    print("DNMT1 FLANKING SEQUENCE ANALYSIS — ALL 16 DROPPED SITES")
    print("Based on Adam et al. 2020 (Nat Commun 11:3723)")
    print("=" * 110)

    for s in sorted(dropped, key=lambda x: x["score"]):
        print(f"\n  {'─' * 100}")
        print(f"  {s['label']}  CpG{s['cpg_pos']}  |  Hexamer: {s['hexamer']}  |  "
              f"DNMT1 score: {s['score']:+.1f}  |  "
              f"{'⚠ DISFAVORED' if s['score'] < -1 else '⚠ MILDLY DISFAVORED' if s['score'] < 0 else '≈ NEUTRAL' if s['score'] < 1 else '✓ FAVORED'}")
        print(f"  Sequence: ...{s['seq'][max(0,s['cpg_pos']-4):s['cpg_pos']]}[CG]{s['seq'][s['cpg_pos']+2:min(len(s['seq']),s['cpg_pos']+6)]}...")
        print(f"    −2 ({s['m2']}): {classify_position('-2', s['m2'], POS_SCORES[-2]):<40} "
              f"(C disfavored → noncanonical stacking)")
        print(f"    −1 ({s['m1']}): {classify_position('-1', s['m1'], POS_SCORES[-1]):<40} "
              f"(G forces G-G base pair after flip)")
        print(f"    +1 ({s['p1']}): {classify_position('+1', s['p1'], POS_SCORES[+1]):<40} "
              f"(G widens minor groove → kinked helix)")
        print(f"    +2 ({s['p2']}): {classify_position('+2', s['p2'], POS_SCORES[+2]):<40} "
              f"(T/A stabilize active conformation)")
        if s["m2"] == "C" and s["m1"] == "G":
            print(f"    ⚠⚠ COMBINED CG at −2/−1: synergistic disfavor (extra −1.0 penalty)")

    # ═══════════════════════════════════════════
    # SECTION 2: Position-by-position statistics
    # ═══════════════════════════════════════════
    print(f"\n\n{'=' * 110}")
    print("POSITION-BY-POSITION FREQUENCY ANALYSIS")
    print(f"{'=' * 110}")

    for pos, pos_label in [(-2, "−2"), (-1, "−1"), (+1, "+1"), (+2, "+2")]:
        key = {-2: "m2", -1: "m1", +1: "p1", +2: "p2"}[pos]
        print(f"\n  Position {pos_label}:")
        print(f"  {'─' * 80}")

        paper_verdict = {
            -2: "C disfavored (stabilizes noncanonical G-G pair via stacking)",
            -1: "G strongly disfavored (forces G-G pair); A/T favored",
            +1: "G favored (widens minor groove, stabilizes kinked catalytic helix)",
            +2: "T strongly favored, A mildly favored (stabilize active conformation)",
        }
        print(f"  Paper finding: {paper_verdict[pos]}")

        d_counts = Counter(s[key] for s in dropped)
        k_counts = Counter(s[key] for s in kept)
        d_total = sum(d_counts.values())
        k_total = sum(k_counts.values())

        print(f"\n  {'Base':<6}{'Dropped':<18}{'Kept':<18}{'Expected':<12}{'Enrichment':<14}{'DNMT1 effect'}")
        print(f"  {'─' * 80}")

        for base in "ACGT":
            d_n = d_counts.get(base, 0)
            k_n = k_counts.get(base, 0)
            d_pct = d_n / d_total * 100
            k_pct = k_n / k_total * 100
            enrichment = d_pct / k_pct if k_pct > 0 else float('inf')
            sc = POS_SCORES[pos].get(base, 0)
            effect = "DISFAVORED" if sc <= -1.0 else "disfavored" if sc < 0 else "neutral" if sc == 0 else "FAVORED" if sc >= 1.0 else "favored"
            arrow = "↑↑" if enrichment > 1.8 else "↑" if enrichment > 1.3 else "↓↓" if enrichment < 0.5 else "↓" if enrichment < 0.7 else "≈"
            print(f"  {base:<6}{d_n:>2}/16 ({d_pct:>5.1f}%){'':<4}"
                  f"{k_n:>3}/334 ({k_pct:>5.1f}%){'':<3}"
                  f"{25.0:>5.1f}%{'':<5}{enrichment:>4.2f}× {arrow:<4}{effect}")

    # ═══════════════════════════════════════════
    # SECTION 3: Combined disfavor patterns
    # ═══════════════════════════════════════════
    print(f"\n\n{'=' * 110}")
    print("COMBINED DISFAVOR PATTERNS")
    print(f"{'=' * 110}")

    patterns = [
        ("G at −1", lambda s: s["m1"] == "G",
         "Strongest single disfavor — forces noncanonical G-G pairing"),
        ("C at −2", lambda s: s["m2"] == "C",
         "Stabilizes G-G pair via stacking on nontarget strand"),
        ("CG at −2/−1", lambda s: s["m2"] == "C" and s["m1"] == "G",
         "Synergistic: strongest disfavor, CGCGNN rank ≥254/256"),
        ("No G at +1", lambda s: s["p1"] != "G",
         "Missing favorable +1 effect (no minor groove widening)"),
        ("No T/A at +2", lambda s: s["p2"] not in ("T", "A"),
         "Missing favorable +2 effect (no active conformation stabilization)"),
        ("G at −1 AND no G at +1", lambda s: s["m1"] == "G" and s["p1"] != "G",
         "Disfavored upstream + missing downstream compensation"),
        ("Negative DNMT1 score", lambda s: s["score"] < 0,
         "Overall predicted to be disfavored"),
        ("Score ≤ −2", lambda s: s["score"] <= -2,
         "Strongly predicted to be disfavored"),
    ]

    print(f"\n  {'Pattern':<35}{'Dropped':<20}{'Kept':<20}{'Enrich.':<10}{'Verdict'}")
    print(f"  {'─' * 95}")
    for name, test, explanation in patterns:
        d_n = sum(1 for s in dropped if test(s))
        k_n = sum(1 for s in kept if test(s))
        d_pct = d_n / 16 * 100
        k_pct = k_n / 334 * 100
        enrich = d_pct / k_pct if k_pct > 0 else float('inf')
        verdict = "✓✓ YES" if enrich > 1.8 else "✓ yes" if enrich > 1.3 else "✗ no" if enrich < 1.1 else "~ weak"
        print(f"  {name:<35}{d_n:>2}/16 ({d_pct:>5.1f}%){'':<4}"
              f"{k_n:>3}/334 ({k_pct:>5.1f}%){'':<4}{enrich:>5.2f}×{'':<3}{verdict}")
        print(f"  {'':>35}  └ {explanation}")

    # ═══════════════════════════════════════════
    # SECTION 4: Score distribution
    # ═══════════════════════════════════════════
    print(f"\n\n{'=' * 110}")
    print("DNMT1 PREFERENCE SCORE DISTRIBUTION")
    print(f"{'=' * 110}")

    d_scores = [s["score"] for s in dropped]
    k_scores = [s["score"] for s in kept]

    print(f"\n  {'Metric':<30}{'Dropped (n=16)':<20}{'Kept (n=334)':<20}")
    print(f"  {'─' * 65}")
    print(f"  {'Mean score':<30}{np.mean(d_scores):>+8.2f}{'':<12}{np.mean(k_scores):>+8.2f}")
    print(f"  {'Median score':<30}{np.median(d_scores):>+8.2f}{'':<12}{np.median(k_scores):>+8.2f}")
    print(f"  {'Min score':<30}{min(d_scores):>+8.2f}{'':<12}{min(k_scores):>+8.2f}")
    print(f"  {'Max score':<30}{max(d_scores):>+8.2f}{'':<12}{max(k_scores):>+8.2f}")
    print(f"  {'Std dev':<30}{np.std(d_scores):>8.2f}{'':<12}{np.std(k_scores):>8.2f}")

    # Score buckets
    print(f"\n  Score distribution:")
    print(f"  {'Score range':<20}{'Dropped':<15}{'Kept':<15}{'Drop rate'}")
    print(f"  {'─' * 55}")
    for lo, hi, label in [(-5, -2, "≤ −2 (strong disf.)"),
                           (-2, -1, "−2 to −1"),
                           (-1, 0, "−1 to 0"),
                           (0, 1, "0 to +1"),
                           (1, 2, "+1 to +2"),
                           (2, 5, "≥ +2 (strong fav.)")]:
        d_n = sum(1 for s in d_scores if lo <= s < hi)
        k_n = sum(1 for s in k_scores if lo <= s < hi)
        total = d_n + k_n
        rate = d_n / total * 100 if total > 0 else 0
        print(f"  {label:<20}{d_n:>3}{'':<12}{k_n:>3}{'':<12}{rate:>5.1f}%")

    # ═══════════════════════════════════════════
    # SECTION 5: Counterexamples
    # ═══════════════════════════════════════════
    print(f"\n\n{'=' * 110}")
    print("COUNTEREXAMPLES")
    print(f"{'=' * 110}")

    print(f"\n  Dropped sites with FAVORABLE scores (score > 0):")
    counter_fav = [s for s in dropped if s["score"] > 0]
    if counter_fav:
        for s in counter_fav:
            print(f"    {s['label']} CpG{s['cpg_pos']}: {s['hexamer']}  score={s['score']:+.1f}  "
                  f"→ DNMT1 bias does NOT explain this drop")
    else:
        print(f"    None — all dropped sites have neutral or disfavored scores")

    print(f"\n  Kept sites with STRONGLY DISFAVORED scores (score ≤ −2):")
    counter_disf = [s for s in kept if s["score"] <= -2]
    print(f"    {len(counter_disf)} kept sites have score ≤ −2 but were NOT dropped")
    if counter_disf:
        for s in counter_disf[:10]:
            print(f"    B{s['b_idx']} CpG{s['cpg_pos']}: {s['hexamer']}  score={s['score']:+.1f}")
        if len(counter_disf) > 10:
            print(f"    ... and {len(counter_disf) - 10} more")

    # ═══════════════════════════════════════════
    # SECTION 6: Final verdict
    # ═══════════════════════════════════════════
    print(f"\n\n{'=' * 110}")
    print("FINAL VERDICT")
    print(f"{'=' * 110}")

    explained_by_dnmt1 = sum(1 for s in dropped if s["score"] < 0)
    strongly_explained = sum(1 for s in dropped if s["score"] <= -2)
    not_explained = sum(1 for s in dropped if s["score"] >= 0)

    print(f"\n  Of 16 dropped sites:")
    print(f"    {strongly_explained:>2} strongly disfavored by DNMT1 (score ≤ −2)")
    print(f"    {explained_by_dnmt1 - strongly_explained:>2} mildly disfavored (−2 < score < 0)")
    print(f"    {not_explained:>2} NOT explained by DNMT1 bias (score ≥ 0)")
    print(f"\n  Combined with CpG4 positional bias (3× higher drop rate at CpG4):")

    both = sum(1 for s in dropped if s["cpg_pos"] == 4 and s["score"] < 0)
    pos_only = sum(1 for s in dropped if s["cpg_pos"] == 4 and s["score"] >= 0)
    ctx_only = sum(1 for s in dropped if s["cpg_pos"] == 14 and s["score"] < 0)
    neither = sum(1 for s in dropped if s["cpg_pos"] == 14 and s["score"] >= 0)

    print(f"    CpG4 + disfavored context: {both:>2}  (double risk — most likely to fail)")
    print(f"    CpG4 + neutral/good context: {pos_only:>2}  (position alone)")
    print(f"    CpG14 + disfavored context: {ctx_only:>2}  (context alone)")
    print(f"    CpG14 + neutral/good context: {neither:>2}  (unexplained)")

    return all_sites, dropped, kept


if __name__ == "__main__":
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
        run_full_analysis(bricks_20)
