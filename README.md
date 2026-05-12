# Cross-library CpG Context and Motif Analysis

Pipeline for analyzing sequence motifs, CpG-context similarity, dropped methylation sites, and DNMT1 flanking-sequence bias across 24-nt bricks and 20-nt bricks libraries. 

## Repository Structure

```bash id="8u8y7f"
repo/
├── k_mers_gen_comparison.py
├── longest_common_substring.py
├── cross_library_comparison.py
├── histograph.py
├── dropped_site_analysis.py
├── finding_bad_kmers.py
├── cpg_context_analysis.py
├── dnmt1_full_analysis.py
├── final.py
├── bottom40_pairs.py
└── README.md
```

## Required Input Structure

```bash id="0z5d71"
repo/
│
├── k_mers_gen_comparison.py
├── longest_common_substring.py
├── cross_library_comparison.py
├── histograph.py
├── dropped_site_analysis.py
├── finding_bad_kmers.py
├── cpg_context_analysis.py
├── dnmt1_full_analysis.py
├── final.py
├── bottom40_pairs.py
│
└── README.md
```

### Notes

* No external input files are required.
* The 24-nt AHEAD library and 20-nt PANDA library sequences are embedded directly inside the scripts.
* Dropped CpG site lists and sequence contexts are also hardcoded within the analysis scripts.

## Scripts

| Script                        | Purpose                               |
| ----------------------------- | ------------------------------------- |
| `k_mers_gen_comparison.py`    | Shared k-mer discovery                |
| `longest_common_substring.py` | Pairwise LCS similarity analysis      |
| `cross_library_comparison.py` | Cross-library overlap analysis        |
| `histograph.py`               | LCS distribution statistics           |
| `dropped_site_analysis.py`    | Dropped-site feature analysis         |
| `finding_bad_kmers.py`        | Failure-associated k-mer enrichment   |
| `cpg_context_analysis.py`     | CpG-context similarity analysis       |
| `dnmt1_full_analysis.py`      | DNMT1 flanking-sequence bias analysis |
| `final.py`                    | Relationship visualization            |
| `bottom40_pairs.py`           | Weakest-overlap pair analysis         |

## Workflow

### Phase 1 — Overlap Discovery

```bash id="kdmh23"
python k_mers_gen_comparison.py
python longest_common_substring.py
python cross_library_comparison.py
python histograph.py
```

### Phase 2 — Dropped Site Analysis

```bash id="h7smq6"
python dropped_site_analysis.py
python finding_bad_kmers.py
python cpg_context_analysis.py
```

### Phase 3 — DNMT1 Bias Analysis

```bash id="7a9f1x"
python dnmt1_full_analysis.py
python final.py
python bottom40_pairs.py
```

## Main Outputs

* k-mer overlap reports
* LCS heatmaps and networks
* CpG-context similarity plots
* Dropped-site enrichment plots
* DNMT1 motif analyses
* Cross-library comparison visualizations

## Requirements

* Python 3.10+
* numpy
* pandas
* matplotlib
* scipy
* networkx
* seaborn

## Full Documentation

[Google Docs Documentation](https://docs.google.com/document/d/1elhBuNKWhIGg_5GrwhJjYMKeGe0ZsA9TWPzZ9Zf_OMQ/edit?tab=t.0)
