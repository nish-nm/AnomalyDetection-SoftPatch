import pandas as pd
import matplotlib.pyplot as plt

# Data for Set 1 and Set 2
data = {
    "Row Names": ["mvtec_bottle", "mvtec_cable", "mvtec_capsule", "mvtec_carpet", "Mean"],
    "image_auroc_set1": [1.0, 0.994, 0.958, 0.989, 0.985],
    "pixel_auroc_set1": [0.984, 0.984, 0.989, 0.991, 0.981],
    "mean_set1": [0.9922381731781944, 0.9889079413671432, 0.9731978121952651, 0.9899421178033816, 0.9828523930283735],
    "image_auroc_set2": [1.0, 0.993, 0.959, 0.998, 0.987],
    "pixel_auroc_set2": [0.915, 0.970, 0.989, 0.981, 0.964],
    "mean_set2": [0.9577486917491659, 0.9818849360165824, 0.9739019493128883, 0.9896594873862884, 0.9757987661162312]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

# Image AUROC
ax.plot(df["Row Names"], df["image_auroc_set1"], marker='o', label="Image AUROC (Set 1)", color='blue')
ax.plot(df["Row Names"], df["image_auroc_set2"], marker='o', label="Image AUROC (Set 2)", color='blue', linestyle='--')

# Pixel AUROC
ax.plot(df["Row Names"], df["pixel_auroc_set1"], marker='s', label="Pixel AUROC (Set 1)", color='green')
ax.plot(df["Row Names"], df["pixel_auroc_set2"], marker='s', label="Pixel AUROC (Set 2)", color='green', linestyle='--')

# Mean AUROC
ax.plot(df["Row Names"], df["mean_set1"], marker='^', label="Mean AUROC (Set 1)", color='red')
ax.plot(df["Row Names"], df["mean_set2"], marker='^', label="Mean AUROC (Set 2)", color='red', linestyle='--')

# Titles and labels
ax.set_title("Comparison of AUROC Scores between Set 1 and Set 2", fontsize=14)
ax.set_xlabel("Categories", fontsize=12)
ax.set_ylabel("AUROC", fontsize=12)
ax.legend()

# Display the plot
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("auroc_comparison.png")
plt.show()
