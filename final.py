# problematic_A = {1, 7, 8, 10, 12, 14, 15, 17, 19, 21, 24, 27, 29}
# with open("/Volumes/Amishi_SSD/bio_data/12Mar/24bits_primer_analysis/k-mer_analysis.txt", "r") as f:


import re
import matplotlib.pyplot as plt

# -----------------------------
# LOAD FILE
# -----------------------------
with open("/Volumes/Amishi_SSD/bio_data/12Mar/24bits_primer_analysis/k-mer_analysis.txt", "r") as f:
     text = f.read()

# -----------------------------
# EXTRACT DATA
# -----------------------------
pattern = re.compile(
    r'\d+\s+(L\d+-\d+)\s+A(\d+)\s×\sB\d+\s+(\d+)\s+\S+\s+(\d+)\s+(\d+)\s+(YES\(11\)|-)\s+(YES\(\d+\)|-)'
)

data = []

for match in pattern.findall(text):
    b_brick = match[0]
    a_brick = int(match[1])
    lcs = int(match[2])
    posB = int(match[4])
    cpg = (match[5] != '-') or (match[6] != '-')

    data.append((a_brick, b_brick, lcs, posB, cpg))

# -----------------------------
# DROPPED SITES (POSITION-AWARE)
# -----------------------------
DROPPED_SITES = [
    (11, 4, "L1-12"), (12, 14, "L1-13"), (13, 4, "L1-14"),
    (17, 4, "L1-18"), (34, 4, "L1-35"), (43, 4, "L2-9"),
    (68, 14, "L2-34"), (70, 14, "L3-1"), (73, 4, "L3-4"),
    (76, 4, "L3-7"), (83, 4, "L3-14"), (87, 14, "L3-18"),
    (88, 4, "L3-19"), (111, 4, "L4-7"), (162, 4, "L5-23"),
    (173, 4, "L5-34"),
]

dropped_lookup = set((brick, pos) for (_, pos, brick) in DROPPED_SITES)

# -----------------------------
# PROBLEMATIC BRICKS
# -----------------------------
problematic_A = {1, 7, 8, 10, 12, 14, 15, 17, 19, 21, 24, 27, 29}
# -----------------------------
# PREPARE SCATTER DATA
# -----------------------------
x, y, colors = [], [], []

for a, b, lcs, posB, cpg in data:

    is_dropped = (b, posB) in dropped_lookup

    if a in problematic_A and is_dropped:
        color = "purple"
    elif a in problematic_A:
        color = "red"
    elif is_dropped:
        color = "blue"
    else:
        color = "gray"

    x.append(lcs)
    y.append(int(cpg))
    colors.append(color)

# -----------------------------
# SCATTER PLOT
# -----------------------------
plt.figure()

plt.scatter(x, y, c=colors, alpha=0.7)

plt.xlabel("LCS")
plt.ylabel("CpG overlap (0/1)")
plt.title("Position-aware Scatter: Dropped vs Problematic")

# Legend
import matplotlib.patches as mpatches
plt.legend(handles=[
    mpatches.Patch(color='red', label='Problematic'),
    mpatches.Patch(color='blue', label='Dropped (position-specific)'),
    mpatches.Patch(color='purple', label='Both'),
    mpatches.Patch(color='gray', label='Other')
])

plt.show()