import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# EMPATHY STRATEGY NAMES MAPPING
EMPATHY_STRATEGY_NAMES = {
    -1: "General Sympathy",
    0: "Detailed Validation",
    1: "Acknowledging Difficulty",
    2: "Offering Support",
    3: "Recognizing Deep Pain",
    4: "Validating Feelings",
    5: "Celebrating Joy",
    6: "Recognizing Strength",
    7: "Encouraging Self-Care",
    8: "Offering Wisdom",
    9: "Finding Silver Linings",
    10: "Inspiring Hope"
}


class EmpathyStrategyAnalyzer:
    def __init__(self):
        # Load data
        cluster_df = pd.read_excel('responses_with_cluster_chains.xlsx')
        original_df = pd.read_csv('StoryDBfull_24012025_Public.csv')

        self.data = cluster_df.merge(original_df, on=['SubID', 'StudyNum'], how='left')

        # Create combined "support" rating from three support-related measures
        support_columns = ['Deg_Keep_con_1', 'Deg_Could_help_1', 'Deg_Helped_1']

        # Calculate average of the three support measures for each row
        valid_support_data = []
        for _, row in self.data.iterrows():
            support_values = []
            for col in support_columns:
                if col in row and pd.notna(row[col]):
                    support_values.append(row[col])

            if len(support_values) > 0:
                valid_support_data.append(np.mean(support_values))
            else:
                valid_support_data.append(np.nan)

        # Add the new "support" column to the dataframe
        self.data['support'] = valid_support_data

        # Updated rating columns (removed individual support measures, added combined "support")
        self.rating_columns = [
            'EmpathyQ_1', 'cognitive', 'affective', 'motivational', 'general_empathy',
            'PosRes', 'NegEmotions', 'PosEmotions',
            'Authenticity_1', 'support', 'AidedBy', 'Feel_aboutAI_1'
        ]

    def get_strategy_name(self, cluster_id, truncate_length=12):
        """Get strategy name for a cluster ID, with optional truncation"""
        if cluster_id in EMPATHY_STRATEGY_NAMES:
            name = EMPATHY_STRATEGY_NAMES[cluster_id]
            if truncate_length and len(name) > truncate_length:
                return name[:truncate_length] + "..."
            return name
        return f"Unknown {cluster_id}"

    def get_cluster_color(self, cluster_id):
        """Get a consistent color for each cluster"""
        # Define a color palette for clusters
        colors = [
            '#FF6B6B',  # Red - C0: Detailed Validation
            '#4ECDC4',  # Teal - C1: Acknowledging Difficulty
            '#45B7D1',  # Blue - C2: Offering Support
            '#96CEB4',  # Green - C3: Recognizing Deep Pain
            '#FFEAA7',  # Yellow - C4: Validating Feelings
            '#DDA0DD',  # Plum - C5: Celebrating Joy
            '#98D8C8',  # Mint - C6: Recognizing Strength
            '#F7DC6F',  # Light Yellow - C7: Encouraging Self-Care
            '#BB8FCE',  # Light Purple - C8: Offering Wisdom
            '#85C1E9',  # Light Blue - C9: Finding Silver Linings
            '#F8C471',  # Orange - C10: Inspiring Hope
        ]

        if cluster_id is None or cluster_id == -1:
            return '#CCCCCC'  # Gray for noise/N/A

        return colors[cluster_id % len(colors)]

    def parse_chains(self):
        """Extract individual clusters for ANOVA analysis"""
        self.cluster_ratings = {rating: defaultdict(list) for rating in self.rating_columns}

        for _, row in self.data.iterrows():
            try:
                chain_raw = row['cluster_chain_list']
                if pd.isna(chain_raw):
                    continue

                if isinstance(chain_raw, str):
                    chain = eval(chain_raw)
                else:
                    chain = chain_raw

                clean_chain = [c for c in chain if c != -1]
                if len(clean_chain) == 0:
                    continue

                for rating_col in self.rating_columns:
                    if rating_col in row and pd.notna(row[rating_col]):
                        rating_value = row[rating_col]

                        # Scale PosRes from 0-100 to 0-10 for consistency
                        if rating_col == 'PosRes':
                            rating_value = rating_value / 10.0

                        # Individual clusters
                        for cluster in clean_chain:
                            self.cluster_ratings[rating_col][cluster].append(rating_value)
            except:
                continue

    def perform_anova_analysis(self):
        """Perform ANOVA analysis for each rating dimension"""
        self.anova_results = {}
        self.top_clusters_per_rating = {}

        for rating_col in self.rating_columns:
            # Filter clusters with sufficient data (at least 5 samples)
            valid_clusters = {}
            for cluster, ratings in self.cluster_ratings[rating_col].items():
                if len(ratings) >= 5:
                    valid_clusters[cluster] = ratings

            if len(valid_clusters) < 2:
                print(f"⚠️  Insufficient data for ANOVA in {rating_col}")
                self.anova_results[rating_col] = {
                    'f_statistic': None,
                    'p_value': None,
                    'effect_size': None,
                    'cluster_means': {},
                    'cluster_stds': {},
                    'cluster_counts': {}
                }
                self.top_clusters_per_rating[rating_col] = []
                continue

            # Prepare data for ANOVA
            groups = list(valid_clusters.values())
            cluster_ids = list(valid_clusters.keys())

            # Perform one-way ANOVA
            f_statistic, p_value = stats.f_oneway(*groups)

            # Calculate effect size (eta-squared)
            total_sum_sq = sum(np.var(group, ddof=1) * (len(group) - 1) for group in groups)
            between_sum_sq = sum(
                len(group) * (np.mean(group) - np.mean(np.concatenate(groups))) ** 2 for group in groups)
            eta_squared = between_sum_sq / (between_sum_sq + total_sum_sq) if (between_sum_sq + total_sum_sq) > 0 else 0

            # Calculate cluster statistics
            cluster_means = {cluster_id: np.mean(ratings) for cluster_id, ratings in valid_clusters.items()}
            cluster_stds = {cluster_id: np.std(ratings, ddof=1) for cluster_id, ratings in valid_clusters.items()}
            cluster_counts = {cluster_id: len(ratings) for cluster_id, ratings in valid_clusters.items()}

            # Get top 5 clusters by mean rating
            sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)
            top_5_clusters = sorted_clusters[:5]

            self.anova_results[rating_col] = {
                'f_statistic': f_statistic,
                'p_value': p_value,
                'effect_size': eta_squared,
                'cluster_means': cluster_means,
                'cluster_stds': cluster_stds,
                'cluster_counts': cluster_counts
            }

            self.top_clusters_per_rating[rating_col] = top_5_clusters

    def create_detailed_summary(self):
        """Create a detailed summary of ANOVA results"""
        print("\n" + "=" * 100)
        print("🔬 ANOVA ANALYSIS: EMPATHY STRATEGY PERFORMANCE BY RATING DIMENSION")
        print("=" * 100)

        for rating_col in self.rating_columns:
            result = self.anova_results[rating_col]

            print(f"\n📊 {rating_col.upper()}")
            print("-" * 60)

            if result['f_statistic'] is not None:
                print(f"   F-statistic: {result['f_statistic']:.3f}")
                print(f"   p-value: {result['p_value']:.6f}")
                print(f"   Effect size (η²): {result['effect_size']:.3f}")

                significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if \
                result['p_value'] < 0.05 else "ns"
                print(f"   Significance: {significance}")

                effect_interpretation = "Large" if result['effect_size'] > 0.14 else "Medium" if result[
                                                                                                     'effect_size'] > 0.06 else "Small"
                print(f"   Effect size: {effect_interpretation}")

                print(f"\n   🏆 TOP 5 STRATEGIES:")
                for rank, (cluster_id, mean_rating) in enumerate(self.top_clusters_per_rating[rating_col], 1):
                    strategy_name = self.get_strategy_name(cluster_id, truncate_length=None)
                    std = result['cluster_stds'][cluster_id]
                    count = result['cluster_counts'][cluster_id]
                    print(f"      {rank}. C{cluster_id} ({strategy_name}): {mean_rating:.3f} ± {std:.3f} (n={count})")
            else:
                print("   ⚠️  Insufficient data for analysis")

    def create_visualizations(self):
        """Create grouped bar chart showing top 5 clusters for each rating dimension"""
        # Parse the data
        self.parse_chains()

        # Perform ANOVA analysis
        self.perform_anova_analysis()

        # Create detailed summary
        self.create_detailed_summary()

        # Prepare data for visualization
        rating_labels = []
        cluster_data = {i: [] for i in range(5)}  # Top 5 positions
        cluster_colors = {i: [] for i in range(5)}
        cluster_labels = {i: [] for i in range(5)}

        for rating_col in self.rating_columns:
            # Clean up rating column name for display
            clean_name = rating_col.replace('_', '\n').replace('1', '')
            rating_labels.append(clean_name)

            top_clusters = self.top_clusters_per_rating[rating_col]

            # Fill data for top 5 positions
            for pos in range(5):
                if pos < len(top_clusters):
                    cluster_id, mean_rating = top_clusters[pos]
                    cluster_data[pos].append(mean_rating)
                    cluster_colors[pos].append(self.get_cluster_color(cluster_id))
                    strategy_name = self.get_strategy_name(cluster_id, truncate_length=8)
                    cluster_labels[pos].append(f"C{cluster_id}")
                else:
                    # No data for this position
                    cluster_data[pos].append(0)
                    cluster_colors[pos].append('#F0F0F0')
                    cluster_labels[pos].append('')

        # Create the visualization
        fig, ax = plt.subplots(figsize=(24, 10))

        # Set up the grouped bar chart
        x = np.arange(len(rating_labels))
        width = 0.15  # Width of individual bars
        positions = [x + i * width for i in range(-2, 3)]  # 5 positions centered around x

        # Prepare error bar data
        error_data = {i: [] for i in range(5)}

        for rating_col in self.rating_columns:
            top_clusters = self.top_clusters_per_rating[rating_col]
            result = self.anova_results[rating_col]

            for pos in range(5):
                if pos < len(top_clusters):
                    cluster_id, mean_rating = top_clusters[pos]
                    std_error = result['cluster_stds'][cluster_id] / np.sqrt(result['cluster_counts'][cluster_id])
                    error_data[pos].append(std_error)
                else:
                    error_data[pos].append(0)

        bars_list = []
        for pos in range(5):
            bars = ax.bar(positions[pos], cluster_data[pos], width,
                          yerr=error_data[pos], capsize=3,
                          color=cluster_colors[pos], alpha=0.8,
                          edgecolor='black', linewidth=0.5,
                          error_kw={'elinewidth': 1, 'capthick': 1},
                          label=f'Rank {pos + 1}')
            bars_list.append(bars)

            # Add cluster labels on bars
            for bar, label, error in zip(bars, cluster_labels[pos], error_data[pos]):
                if label:  # Only add label if there's data
                    height = bar.get_height()
                    if height > 0:
                        label_height = height + error + height * 0.02
                        ax.text(bar.get_x() + bar.get_width() / 2., label_height,
                                label, ha='center', va='bottom', fontsize=8, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # Customize the plot
        ax.set_title(
            'Top 5 Empathy Strategies by Rating Dimension\n(ANOVA Analysis)',
            fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Rating Dimensions\n(Parameter boxes show F-statistic, p-value, effect size)', fontsize=14,
                      fontweight='bold')
        ax.set_ylabel('Mean Rating Score ± Standard Error (0-10 scale)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(rating_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title='Ranking', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add ANOVA parameter boxes for each dimension
        y_max = max([max(cluster_data[pos]) for pos in range(5) if cluster_data[pos]]) if any(
            cluster_data[pos] for pos in range(5)) else 10

        for i, rating_col in enumerate(self.rating_columns):
            result = self.anova_results[rating_col]

            # Get ANOVA parameters for this dimension
            if result['p_value'] is not None:
                f_stat = f"F={result['f_statistic']:.1f}" if result['f_statistic'] is not None else "F=N/A"

                if result['p_value'] < 0.001:
                    p_val = "p<0.001"
                elif result['p_value'] < 0.01:
                    p_val = "p<0.01"
                elif result['p_value'] < 0.05:
                    p_val = "p<0.05"
                else:
                    p_val = f"p={result['p_value']:.3f}"

                eta_sq = f"η²={result['effect_size']:.2f}" if result['effect_size'] is not None else "η²=N/A"
            else:
                f_stat, p_val, eta_sq = "F=N/A", "p=N/A", "η²=N/A"

            # Position the info box below each group of bars
            box_x = x[i]
            box_y = -y_max * 0.25  # Below the x-axis

            # Create info text
            info_text = f"{f_stat}\n{p_val}\n{eta_sq}"

            # Determine box color based on significance
            if "p<0.001" in p_val:
                box_color = '#FF6B6B'  # Red for highly significant
                alpha = 0.8
            elif "p<0.01" in p_val:
                box_color = '#FFA500'  # Orange for significant
                alpha = 0.7
            elif "p<0.05" in p_val:
                box_color = '#FFFF00'  # Yellow for marginally significant
                alpha = 0.6
            else:
                box_color = '#D3D3D3'  # Gray for not significant
                alpha = 0.5

            # Add the info box
            ax.text(box_x, box_y, info_text, ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=alpha, edgecolor='black'),
                    family='monospace', fontweight='bold')

        # Adjust y-axis limits to accommodate info boxes
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0] - y_max * 0.4, current_ylim[1])

        plt.tight_layout()
        plt.savefig('empathy_strategy_anova_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print additional insights
        print(f"\n💡 KEY INSIGHTS FROM ANOVA ANALYSIS:")
        print("=" * 60)

        # Explain ANOVA method
        print(f"\n📚 UNDERSTANDING ANOVA (Analysis of Variance):")
        print("-" * 60)
        print("ANOVA tests whether group means significantly differ from each other.")
        print("It measures how much the variation BETWEEN groups exceeds variation WITHIN groups.\n")

        print("🔍 Key Parameters:")
        print("• F-statistic: Ratio of between-group variance to within-group variance")
        print("  - Higher F = groups differ more than expected by chance")
        print("  - F > 1.0 suggests some group differences exist")
        print("  - F >> 1.0 (like 10+) indicates strong group differences")

        print("\n• p-value: Probability these results occurred by random chance")
        print("  - p < 0.05: Statistically significant (*)")
        print("  - p < 0.01: Highly significant (**)")
        print("  - p < 0.001: Very highly significant (***)")
        print("  - p ≥ 0.05: Not significant (ns)")

        print("\n• Effect size (η² - eta squared): Practical significance")
        print("  - η² = 0.01-0.06: Small effect")
        print("  - η² = 0.06-0.14: Medium effect")
        print("  - η² > 0.14: Large effect")
        print("  - Shows what % of variance is explained by strategy differences")

        print("\n• Error bars: Standard Error of the Mean (SEM)")
        print("  - Shows precision of each group's mean estimate")
        print("  - Smaller bars = more precise/reliable estimate")
        print("  - SEM = Standard Deviation / √(sample size)")

        print("\n• Support dimension: Combined measure")
        print("  - Average of: Degree Helped, Could Help, Keep Connection")
        print("  - Represents overall perceived supportiveness of empathy strategy")

        print(f"\n📊 HOW TO READ YOUR RESULTS:")
        print("-" * 60)
        print("✅ SIGNIFICANT results (*, **, ***) mean empathy strategies truly differ")
        print("❌ NON-SIGNIFICANT (ns) means strategies perform similarly")
        print("🎯 LARGE effect sizes (η² > 0.14) show meaningful practical differences")
        print("📏 ERROR BARS show how confident we are in each strategy's performance")

        # Find most significant results
        significant_results = []
        for rating_col in self.rating_columns:
            result = self.anova_results[rating_col]
            if result['p_value'] is not None and result['p_value'] < 0.05:
                significant_results.append((rating_col, result['p_value'], result['effect_size']))

        significant_results.sort(key=lambda x: x[1])  # Sort by p-value

        print(f"\n🌟 MOST SIGNIFICANT DIFFERENCES:")
        for rating_col, p_val, effect_size in significant_results[:5]:
            effect_interp = "Large" if effect_size > 0.14 else "Medium" if effect_size > 0.06 else "Small"
            print(f"   {rating_col}: p={p_val:.6f}, η²={effect_size:.3f} ({effect_interp} effect)")

        # Find most consistently top-performing strategies
        cluster_rankings = defaultdict(list)
        for rating_col in self.rating_columns:
            for rank, (cluster_id, _) in enumerate(self.top_clusters_per_rating[rating_col]):
                cluster_rankings[cluster_id].append(rank + 1)

        print(f"\n🏆 MOST CONSISTENTLY HIGH-PERFORMING STRATEGIES:")
        avg_rankings = {cluster_id: np.mean(ranks) for cluster_id, ranks in cluster_rankings.items() if len(ranks) >= 3}
        sorted_avg_rankings = sorted(avg_rankings.items(), key=lambda x: x[1])

        for cluster_id, avg_rank in sorted_avg_rankings[:5]:
            strategy_name = self.get_strategy_name(cluster_id, truncate_length=None)
            appearances = len(cluster_rankings[cluster_id])
            print(
                f"   C{cluster_id} ({strategy_name}): Avg rank {avg_rank:.1f} (appears in top 5 of {appearances} dimensions)")

        print(f"\n✅ SCALE NORMALIZATION:")
        print(f"   PosRes values have been scaled from 0-100 to 0-10 for direct comparison")
        print(f"   All dimensions now use consistent 0-10 scaling")

        print(f"\n🔗 SUPPORT DIMENSION CREATION:")
        print(f"   'support' = average of (Deg_Helped_1, Deg_Could_help_1, Deg_Keep_con_1)")
        print(f"   This combines three related support measures into one comprehensive metric")


def main():
    print("🔬 EMPATHY STRATEGY ANOVA PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Analyzing empathy strategies using ANOVA across multiple rating dimensions...")
    print("📏 Scale normalization: PosRes (0-100) scaled to (0-10) for consistency")
    print("🔗 Combined metric: 'support' = average of (Deg_Helped_1, Deg_Could_help_1, Deg_Keep_con_1)")
    print(f"Rating dimensions: EmpathyQ_1, cognitive, affective, motivational, general_empathy,")
    print(f"                  PosRes, NegEmotions, PosEmotions, Authenticity_1,")
    print(f"                  support, AidedBy, Feel_aboutAI_1")

    analyzer = EmpathyStrategyAnalyzer()
    analyzer.create_visualizations()


if __name__ == "__main__":
    main()