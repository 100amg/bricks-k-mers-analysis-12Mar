"""
K-mer Library Generator & Cross-Primer Matcher
================================================
Generates k-mer libraries (k=5–10) for a list of primer sequences,
finds all matching k-mers between different primers, and reports
match counts and locations.
"""

from collections import Counter, defaultdict
from itertools import combinations


def generate_kmers(sequence: str, k: int) -> list[tuple[str, int]]:
    """Return all k-mers in a sequence as (kmer, start_position) tuples."""
    sequence = sequence.upper()
    return [(sequence[i:i + k], i) for i in range(len(sequence) - k + 1)]


def build_kmer_index(primers: list[str], k: int) -> dict:
    """
    Build an inverted index: kmer -> list of (primer_index, position).
    Only k-mers that appear in 2+ different primers are interesting.
    """
    index = defaultdict(list)
    for primer_idx, seq in enumerate(primers):
        for kmer, pos in generate_kmers(seq, k):
            index[kmer].append((primer_idx, pos))
    return index


def find_cross_primer_matches(primers: list[str], k_range: range) -> dict:
    """
    For each k value, find all k-mers shared between different primers.

    Returns
    -------
    dict  keyed by k, each value is a dict with:
        - "matches": list of match records
        - "match_count": total number of pairwise matches
        - "kmer_library": {primer_index: [list of kmers]}
    """
    results = {}

    for k in k_range:
        index = build_kmer_index(primers, k)
        matches = []

        for kmer, locations in index.items():
            # Group locations by primer
            primers_with_kmer = defaultdict(list)
            for primer_idx, pos in locations:
                primers_with_kmer[primer_idx].append(pos)

            # Only care about k-mers found in more than one distinct primer
            involved_primers = list(primers_with_kmer.keys())
            if len(involved_primers) < 2:
                continue

            # Record every cross-primer pair
            for p1, p2 in combinations(involved_primers, 2):
                for pos1 in primers_with_kmer[p1]:
                    for pos2 in primers_with_kmer[p2]:
                        matches.append({
                            "kmer": kmer,
                            "primer_a": p1,
                            "pos_a": pos1,
                            "primer_b": p2,
                            "pos_b": pos2,
                        })

        # Build per-primer k-mer library for reference
        kmer_library = {}
        all_unique_kmers = set()
        kmer_freq = Counter()  # total occurrences across all primers
        for idx, seq in enumerate(primers):
            kmers = [km for km, _ in generate_kmers(seq, k)]
            kmer_library[idx] = kmers
            all_unique_kmers.update(kmers)
            kmer_freq.update(kmers)

        unique_shared = {m["kmer"] for m in matches}

        results[k] = {
            "matches": matches,
            "match_count": len(matches),
            "kmer_library": kmer_library,
            "total_unique_kmers": len(all_unique_kmers),
            "unique_shared_kmers": len(unique_shared),
            "kmer_freq": kmer_freq,
        }

    return results


def print_report(primers: list[str], results: dict) -> None:
    """Pretty-print the full analysis."""
    print("=" * 70)
    print("K-MER CROSS-PRIMER MATCH REPORT")
    print("=" * 70)

    print(f"\nPrimers analysed: {len(primers)}")
    for i, seq in enumerate(primers):
        print(f"  Primer {i}: {seq}  (len {len(seq)})")

    for k in sorted(results):
        data = results[k]
        print(f"\n{'─' * 70}")
        print(f"  k = {k}   |   Total matching pairs: {data['match_count']}   |   Unique k-mers: {data['total_unique_kmers']}")
        print(f"{'─' * 70}")

        # Show library sizes
        for idx in sorted(data["kmer_library"]):
            n = len(data["kmer_library"][idx])
            print(f"    Primer {idx}: {n} k-mers")

        if data["matches"]:
            # Group matches by k-mer for cleaner output
            by_kmer = defaultdict(list)
            for m in data["matches"]:
                by_kmer[m["kmer"]].append(m)

            print(f"\n    Shared k-mers ({len(by_kmer)} unique):")
            for kmer, group in sorted(by_kmer.items()):
                print(f"\n      '{kmer}'")
                for m in group:
                    print(
                        f"        Primer {m['primer_a']}[{m['pos_a']}:{m['pos_a']+k}]"
                        f"  <->  Primer {m['primer_b']}[{m['pos_b']}:{m['pos_b']+k}]"
                    )
        else:
            print("\n    No cross-primer matches found.")

        # Primer-pair overlap: count unique shared k-mers per pair
        pair_shared = defaultdict(set)
        for m in data["matches"]:
            pair_key = (m["primer_a"], m["primer_b"])
            pair_shared[pair_key].add(m["kmer"])

        # Filter to pairs with > 2 shared k-mers, sort descending
        heavy_pairs = [
            (p, kmers) for p, kmers in pair_shared.items() if len(kmers) > 2
        ]
        heavy_pairs.sort(key=lambda x: len(x[1]), reverse=True)

        if heavy_pairs:
            print(f"\n    Primer pairs with >2 shared k-mers:")
            print(f"    {'Pair':<20}{'# Shared':<10}{'K-mers'}")
            print(f"    {'─' * 60}")
            for (p1, p2), kmers in heavy_pairs:
                kmer_str = ", ".join(sorted(kmers))
                print(f"    Primer {p1} & {p2:<10}{len(kmers):<10}{kmer_str}")
        else:
            print(f"\n    No primer pairs share more than 2 k-mers.")

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for k in sorted(results):
        unique_shared = results[k]["unique_shared_kmers"]
        total_unique = results[k]["total_unique_kmers"]
        print(
            f"  k={k:>2}  ->  {results[k]['match_count']:>4} matching pairs  "
            f"({unique_shared} unique shared k-mers / {total_unique} unique total k-mers)"
        )

    # Top 10 most common k-mers for k = 5, 6, 7
    print(f"\n{'=' * 70}")
    print("TOP 10 MOST FREQUENT K-MERS  (k = 5, 6, 7)")
    print(f"{'=' * 70}")
    for k in [5, 6, 7]:
        if k not in results:
            continue
        top10 = results[k]["kmer_freq"].most_common(10)

        # Build kmer -> set of primer indices
        kmer_to_primers = defaultdict(set)
        for idx, kmers in results[k]["kmer_library"].items():
            for km in kmers:
                kmer_to_primers[km].add(idx)

        print(f"\n  k = {k}:")
        print(f"  {'Rank':<6}{'K-mer':<15}{'Count':<8}{'Shared?':<10}{'Present in primers'}")
        print(f"  {'─' * 60}")
        shared_set = {m["kmer"] for m in results[k]["matches"]}
        for rank, (kmer, count) in enumerate(top10, 1):
            flag = "yes" if kmer in shared_set else "-"
            primer_list = ", ".join(str(p) for p in sorted(kmer_to_primers[kmer]))
            print(f"  {rank:<6}{kmer:<15}{count:<8}{flag:<10}{primer_list}")


def plot_kmer_bar_chart(results: dict, output_path: str = "kmer_bar_chart.png") -> None:
    """Bar chart: total unique k-mers (blue) with shared sub-bar (orange)."""
    import matplotlib.pyplot as plt

    ks = sorted(results.keys())
    total_unique = [results[k]["total_unique_kmers"] for k in ks]
    unique_shared = [results[k]["unique_shared_kmers"] for k in ks]

    x = range(len(ks))
    bar_width = 0.55

    fig, ax = plt.subplots(figsize=(8, 5))

    # Blue bars for total unique
    bars_total = ax.bar(x, total_unique, bar_width,
                        color="#4A90D9", label="Total unique k-mers")
    # Orange bars (overlaid) for shared unique
    bars_shared = ax.bar(x, unique_shared, bar_width,
                         color="#E8913A", label="Uniquely shared k-mers")

    # Labels on top of each bar
    for i, (t, s) in enumerate(zip(total_unique, unique_shared)):
        ax.text(i, t + 0.8, str(t), ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#4A90D9")
        if s > 0:
            ax.text(i, s + 0.5, str(s), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#E8913A")

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("K-mer length")
    ax.set_ylabel("Number of unique k-mers")
    ax.set_title("K-mer Analysis: Total vs Shared Unique K-mers")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(total_unique) * 1.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nBar chart saved to: {output_path}")


# ──────────────────────────────────────────────
#  Example usage — swap in your own primers here
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

    results = find_cross_primer_matches(primers, k_range=range(5, 11))
    print_report(primers, results)
    plot_kmer_bar_chart(results, output_path="kmer_bar_chart.png")