def add_bar_labels(ax, fmt="{:.0f}", skip_zero=True, eps=1e-9):
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h) or (skip_zero and abs(h) < eps):
            continue
        ax.annotate(fmt.format(h),
                    (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=5)


def plot_target_distribution(df):
    if "default_flag_customer" not in df.columns:
        return
    counts = df["default_flag_customer"].value_counts().sort_index()
    labels = ["Non-default", "Default"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    sns.barplot(x=labels, y=values, ax=ax, palette=[DARK_GREEN, ORANGE])
    ax.set_title("Customer Default Distribution", fontsize=8)
    ax.set_xlabel("Default Flag", fontsize=6)
    ax.set_ylabel("Count", fontsize=6)
    ax.tick_params(axis="both", labelsize=5)
    add_bar_labels(ax)
    plt.tight_layout()
    st.pyplot(fig)


def plot_default_rate_by_category(df, col, max_categories=8):
    if col not in df.columns or "default_flag_customer" not in df.columns:
        return
    top = df[col].value_counts().head(max_categories).index
    tmp = df[df[col].isin(top)].copy()
    rate_df = (
        tmp.groupby(col)["default_flag_customer"]
        .mean()
        .reset_index()
        .rename(columns={"default_flag_customer": "default_rate"})
        .sort_values("default_rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    sns.barplot(x=col, y="default_rate", data=rate_df, color=ORANGE, ax=ax)
    ax.set_title(f"Default Rate by {col}", fontsize=8)
    ax.set_ylabel("Default Rate", fontsize=6)
    ax.set_xlabel(col, fontsize=6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=5)
    ax.tick_params(axis="y", labelsize=5)
    add_bar_labels(ax, fmt="{:.1%}")
    plt.tight_layout()
    st.pyplot(fig)


def plot_pd_histogram(scored_df):
    if "pd" not in scored_df.columns:
        return
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    counts, _, patches = ax.hist(scored_df["pd"], bins=20, color=ORANGE, edgecolor="k")
    ax.set_title("Predicted PD Distribution", fontsize=8)
    ax.set_xlabel("PD", fontsize=6)
    ax.set_ylabel("Count", fontsize=6)
    ax.tick_params(axis="both", labelsize=5)
    for count, patch in zip(counts, patches):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width() / 2, count + 0.5,
                    f"{int(count)}", ha="center", va="bottom", fontsize=5)
    plt.tight_layout()
    st.pyplot(fig)


def plot_correlation_heatmap(df_fe, max_features=15):
    num_cols = [c for c in df_fe.select_dtypes(include=[np.number]).columns if "id" not in c.lower()]
    if len(num_cols) < 2:
        return
    if len(num_cols) > max_features:
        num_cols = num_cols[:max_features]
    corr = df_fe[num_cols].corr()
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(corr, cmap=HEATMAP_CMAP, square=True, linewidths=0.3,
                cbar_kws={"shrink": 0.5}, ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=8)
    ax.tick_params(axis="both", labelsize=5)
    plt.tight_layout()
    st.pyplot(fig)


def plot_default_rate_deciles(dec_table):
    if "default_rate" not in dec_table.columns:
        return
    d = dec_table.sort_values("decile")
    x = d["decile"].astype(str)
    fig, ax1 = plt.subplots(figsize=(3.8, 2.6))
    bars = ax1.bar(x, d["default_rate"], color=ORANGE)
    ax1.set_xlabel("Decile", fontsize=6)
    ax1.set_ylabel("Default Rate", fontsize=6)
    ax1.tick_params(axis="both", labelsize=5)
    for p in bars:
        h = p.get_height()
        if not np.isnan(h):
            ax1.text(p.get_x() + p.get_width() / 2, h + 0.005,
                     f"{h:.1%}", ha="center", va="bottom", fontsize=5)
    if "cum_default_rate" in d.columns:
        ax2 = ax1.twinx()
        ax2.plot(x, d["cum_default_rate"], marker="o", color=DARK_GREEN,
                 linewidth=1.2, markersize=2)
        ax2.set_ylabel("Cumulative Default Rate", fontsize=6)
        ax2.tick_params(axis="y", labelsize=5)
    ax1.set_title("Default Rate and Cumulative Default Rate by Decile", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)


def plot_top_bottom_decile_feature_means(scored_full):
    if "decile" not in scored_full.columns:
        return
    features = ["worst_slik_score", "pay_ratio_total", "late_ratio", "n_defaulted_loans"]
    features = [f for f in features if f in scored_full.columns]
    if not features:
        return
    top = scored_full[scored_full["decile"] == 10]
    bottom = scored_full[scored_full["decile"] == 1]
    if top.empty or bottom.empty:
        return
    mean_top = top[features].mean()
    mean_bottom = bottom[features].mean()
    plot_df = pd.DataFrame({
        "Feature": features,
        "Decile 1 (lowest PD)": mean_bottom.values,
        "Decile 10 (highest PD)": mean_top.values,
    }).melt(id_vars="Feature", var_name="Group", value_name="Value")
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    sns.barplot(data=plot_df, x="Feature", y="Value", hue="Group", ax=ax,
                palette={"Decile 1 (lowest PD)": DARK_GREEN, "Decile 10 (highest PD)": ORANGE})
    ax.set_title("Key Behavioural Features: Decile 1 vs Decile 10", fontsize=8)
    ax.set_xlabel("Feature", fontsize=6)
    ax.set_ylabel("Mean Value", fontsize=6)
    ax.tick_params(axis="x", labelsize=5, rotation=45)
    ax.tick_params(axis="y", labelsize=5)
    ax.legend(title="Group", fontsize=5.5, title_fontsize=6)
    add_bar_labels(ax, fmt="{:.2f}", skip_zero=True)
    plt.tight_layout()
    st.pyplot(fig)
