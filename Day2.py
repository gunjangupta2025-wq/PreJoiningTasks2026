import matplotlib.pyplot as plt
import seaborn as sns

# Set clean structural aesthetics using seaborn styling engine
sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 4))
# Simulating timeline data tracking multiple groups
plt.plot(df_time["Date"], df_time["Sales_A"], label="Region A", color="royalblue", linewidth=2)
plt.plot(df_time["Date"], df_time["Sales_B"], label="Region B", color="crimson", linewidth=2, linestyle="--")

plt.title("Regional Sales Performance Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Timeline Metrics")
plt.ylabel("Gross Sales ($)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
# hue handles categorization grouping; palette sets styling colors
sns.scatterplot(data=df, x="Engine_Size", y="Horsepower", hue="Vehicle_Type", palette="viridis", s=100)

plt.title("Engine Displacement vs. Output Performance", fontsize=13)
plt.xlabel("Displacement Volume (Liters)")
plt.ylabel("Horsepower Rating")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
# kde=True overlays an estimated Kernel Density curve tracing the shape of the data distribution
sns.histplot(data=df, x="Exam_Scores", bins=20, kde=True, color="teal")

plt.title("Student Performance Frequency Distribution", fontsize=13)
plt.xlabel("Scores Achieved")
plt.ylabel("Student Count")
plt.show()

# Create a figure grid of 1 Row, 2 Columns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

# Plot 1 on Left Axis (0)
sns.boxplot(data=df, x="Category", y="Value", ax=axes[0], palette="pastel")
axes[0].set_title("Value Spread Across Categories")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

# Plot 2 on Right Axis (1)
sns.barplot(data=df, x="Category", y="Value", ax=axes[1], estimator=np.mean, color="salmon")
axes[1].set_title("Mean Evaluation Comparison")

# Figure structural level details
plt.suptitle("Comprehensive Operational Performance Analysis", fontsize=16, fontweight="bold")
plt.tight_layout()  # Automatically adjusts bounds to avoid overlapping text

# Save the current active figure canvas asset safely before calling plt.show()
plt.savefig("analytics_performance_report.png", dpi=300, bbox_inches="tight")
plt.close() # Safely releases backend system memory footprints