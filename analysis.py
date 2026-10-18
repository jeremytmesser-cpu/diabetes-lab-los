import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Lab Tests and Length of Stay in Diabetes Hospitalizations

        ## Research question

        > **How is the number of lab procedures performed during a hospital stay related to the length of that stay, and does this relationship differ across admission types (Elective, Emergency, Urgent)?**

        **Columns used:** `num_lab_procedures` (numeric), `time_in_hospital` (numeric, days), `admission_type` (categorical).

        **Why this question:** hospitals order tests as a stay unfolds, so I expect more lab tests
        to go with longer stays. But an elective admission is planned and an emergency is not, so
        the *strength* of that link might differ by how the patient arrived. The question uses two
        numeric columns plus one categorical column, as the assignment requires.

        **Why this notebook is organised the way it is:** every code cell is preceded by a short
        markdown cell explaining *why* that code exists, and the code carries comments for the
        individual lines.
        """
    )
    return


@app.cell
def _():
    # GitHub repository for this project:
    # https://github.com/jeremytmesser-cpu/diabetes-lab-los

    # Why this cell: the assignment requires the repo link to be a comment in the notebook.
    # It sits next to the imports so anyone opening the notebook sees it first.
    import marimo as mo
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    return mo, np, os, pd, plt, sns, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load and check the data

        **Why:** before analysing anything I need to confirm the file loads, how big it is, and
        that the three columns I care about have no missing values or odd types. A correlation
        computed on silently-missing data would be misleading.
        """
    )
    return


@app.cell
def _(mo, pd):
    # Relative path so the notebook works from the repo root on any machine.
    DATA_PATH = "data/diabetes_130.csv"
    df = pd.read_csv(DATA_PATH)

    # Only the columns that matter for the research question; used whenever rows are displayed.
    RELEVANT = ["admission_type", "time_in_hospital", "num_lab_procedures"]

    mo.vstack(
        [
            mo.md(
                f"**{df.shape[0]:,} stays** and {df.shape[1]} columns in the full file. "
                f"Missing values in the 3 relevant columns: **{int(df[RELEVANT].isna().sum().sum())}**."
            ),
            df[RELEVANT].head(),
        ]
    )
    return RELEVANT, df


@app.cell
def _(RELEVANT, df):
    # Why: summary statistics show the scale and spread of the two numeric columns,
    # and the category counts show whether any admission type is too small to analyse.
    # (Printing counts here also checks the spelling of the category labels used later.)
    numeric_summary = df[["time_in_hospital", "num_lab_procedures"]].describe().round(2)
    admission_counts = df["admission_type"].value_counts()
    print(admission_counts.to_string())
    numeric_summary
    return admission_counts, numeric_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Data operations

        ### 2a. Filtering by column values

        **Why:** long stays are the expensive, high-stakes cases. Filtering to stays of
        7 or more days shows how many there are and which admission types they come from,
        which tells me whether "long stay" is a meaningful group worth comparing against short ones.
        """
    )
    return


@app.cell
def _(RELEVANT, df, mo):
    # Boolean mask on a numeric column: keep stays of a week or longer.
    long_stays = df[df["time_in_hospital"] >= 7]

    # Combining two conditions with & (each in parentheses): long AND emergency admissions.
    long_emergency = df[(df["admission_type"] == "Emergency") & (df["time_in_hospital"] >= 7)]

    mo.vstack(
        [
            mo.md(
                f"**{len(long_stays):,}** of {len(df):,} stays "
                f"({len(long_stays) / len(df):.1%}) lasted 7+ days; "
                f"**{len(long_emergency):,}** of those were emergency admissions "
                f"({len(long_emergency) / len(long_stays):.1%} of long stays)."
            ),
            long_stays[RELEVANT].head(),
        ]
    )
    return long_emergency, long_stays


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 2b. Sorting the dataset

        **Why:** sorting surfaces the extremes. Sorting by stay length and breaking ties by lab
        count shows whether the very longest stays also have the most lab work, which is a quick
        sanity check on the direction of the relationship before I quantify it.
        """
    )
    return


@app.cell
def _(RELEVANT, df):
    # Sort by days (longest first), then by lab procedures (most first) to break ties.
    # Keep only the relevant columns and the top 10 rows.
    longest_stays = (
        df.sort_values(["time_in_hospital", "num_lab_procedures"], ascending=[False, False])[RELEVANT]
        .head(10)
        .reset_index(drop=True)
    )
    longest_stays
    return (longest_stays,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 2c. Grouping by a categorical variable

        **Why:** the research question is about *admission type*, so I need the basic picture
        per group first: how many stays, how long they last, and how much lab work they get.
        If the groups already differ a lot in these averages, that affects how to read the
        correlations later.
        """
    )
    return


@app.cell
def _(df):
    # One row per admission type; named aggregations give readable column names.
    admission_summary = (
        df.groupby("admission_type")
        .agg(
            stays=("time_in_hospital", "size"),
            mean_days=("time_in_hospital", "mean"),
            median_days=("time_in_hospital", "median"),
            mean_lab_procedures=("num_lab_procedures", "mean"),
            median_lab_procedures=("num_lab_procedures", "median"),
        )
        .round(2)
        .sort_values("mean_days")
    )
    admission_summary
    return (admission_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 2d. Adding computed columns

        **Why:** two derived columns help answer the question.

        * `stay_length` bins the days into Short (1-3), Medium (4-6) and Long (7+). Stay length is a
          whole number that is heavily bunched, so bins make a clean boxplot comparison.
        * `labs_per_day` = lab procedures / days. If more lab tests only meant "more days to run
          tests", this rate would stay flat. Checking it shows whether testing is front-loaded.

        I put the new columns in a new DataFrame `data` so the original `df` stays untouched.
        """
    )
    return


@app.cell
def _(RELEVANT, df, mo, pd):
    # assign() returns a copy with the new columns, leaving df unchanged.
    data = df.assign(
        # pd.cut: right-closed bins (0,3], (3,6], (6,14] -> 1-3, 4-6, 7+ days
        stay_length=pd.cut(
            df["time_in_hospital"],
            bins=[0, 3, 6, df["time_in_hospital"].max()],
            labels=["Short (1-3 days)", "Medium (4-6 days)", "Long (7+ days)"],
        ),
        # time_in_hospital is never 0 (minimum is 1), so this division is safe.
        labs_per_day=df["num_lab_procedures"] / df["time_in_hospital"],
    )

    # Mean lab rate per stay-length bin: does testing intensity change with stay length?
    labs_rate_by_stay = (
        data.groupby("stay_length", observed=True)
        .agg(stays=("labs_per_day", "size"), mean_labs=("num_lab_procedures", "mean"), mean_labs_per_day=("labs_per_day", "mean"))
        .round(2)
    )

    mo.vstack(
        [
            data[RELEVANT + ["stay_length", "labs_per_day"]].head().round(2),
            mo.md("Mean lab work by stay-length bin:"),
            labs_rate_by_stay,
        ]
    )
    return data, labs_rate_by_stay


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Visualizations

        **Why a shared setup cell:** using one colour per admission type everywhere means a reader
        never has to re-learn the legend. The palette is colour-blind safe (Okabe-Ito). Figures are
        also saved to `figures/` so they can go straight into the slides.
        """
    )
    return


@app.cell
def _(os, plt, sns):
    # One colour per admission type, reused in every chart.
    COLORS = {"Elective": "#0072B2", "Emergency": "#D55E00", "Urgent": "#009E73"}
    ORDER = ["Elective", "Emergency", "Urgent"]
    NEUTRAL = "#6B7280"

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})

    os.makedirs("figures", exist_ok=True)  # make sure the output folder exists
    return COLORS, NEUTRAL, ORDER


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 3a. Histogram: how long do stays last?

        **Why:** a correlation coefficient assumes a roughly sensible spread of values. The
        histogram shows stay length is right-skewed (many short stays, a long tail), which is
        why I also report a rank-based (Spearman) correlation later as a robustness check.
        """
    )
    return


@app.cell
def _(NEUTRAL, data, np, plt):
    # Bins centred on whole numbers (0.5-1.5, 1.5-2.5, ...) because days are integers.
    fig_hist, ax_hist = plt.subplots(figsize=(9, 5))
    ax_hist.hist(data["time_in_hospital"], bins=np.arange(0.5, 15.5, 1), color=NEUTRAL, edgecolor="white")

    _median = data["time_in_hospital"].median()
    ax_hist.axvline(_median, color="black", linestyle="--", linewidth=1.5)
    ax_hist.annotate(f"Median: {_median:.0f} days", xy=(_median, ax_hist.get_ylim()[1] * 0.92),
                     xytext=(8, 0), textcoords="offset points", fontsize=13)

    ax_hist.set_title("Most stays are short; a few last two weeks")
    ax_hist.set_xlabel("Length of stay (days)")
    ax_hist.set_ylabel("Number of stays")
    fig_hist.tight_layout()
    fig_hist.savefig("figures/01_histogram_stay_length.png", dpi=200)
    fig_hist
    return ax_hist, fig_hist


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 3b. Boxplot: lab procedures by stay-length bin

        **Why:** a boxplot compares the *whole distribution* of lab procedures (median, spread,
        outliers) across short, medium and long stays, using the binned column from step 2d.
        It tests the relationship without assuming a straight line.
        """
    )
    return


@app.cell
def _(NEUTRAL, data, plt, sns):
    fig_box, ax_box = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=data,
        x="stay_length",
        y="num_lab_procedures",
        color=NEUTRAL,
        width=0.55,
        fliersize=3,
        ax=ax_box,
    )
    ax_box.set_title("Longer stays come with more lab procedures")
    ax_box.set_xlabel("Length of stay")
    ax_box.set_ylabel("Lab procedures per stay")
    fig_box.tight_layout()
    fig_box.savefig("figures/02_boxplot_labs_by_stay_length.png", dpi=200)
    fig_box
    return ax_box, fig_box


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 3c. Scatter plot: lab procedures vs. days in hospital

        **Why:** this is the core picture of the research question: two numeric columns plotted
        against each other, coloured by the third (admission type). I add one fitted line per
        group so the *slope and tightness* can be compared by eye. Days are whole numbers, so I add
        a tiny vertical jitter to the *plotted* points only (not to the fitted lines or any
        statistics) so that identical points do not hide each other.
        """
    )
    return


@app.cell
def _(COLORS, ORDER, data, np, plt):
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    _rng = np.random.default_rng(42)  # fixed seed -> the same jitter every run

    for _group in ORDER:
        _g = data[data["admission_type"] == _group]
        _jitter = _rng.uniform(-0.2, 0.2, size=len(_g))
        ax_scatter.scatter(
            _g["num_lab_procedures"], _g["time_in_hospital"] + _jitter,
            s=12, alpha=0.30, color=COLORS[_group], linewidths=0,
        )
        # Least-squares line for this group, computed on the real (un-jittered) values.
        _slope, _intercept = np.polyfit(_g["num_lab_procedures"], _g["time_in_hospital"], 1)
        _x = np.array([_g["num_lab_procedures"].min(), _g["num_lab_procedures"].max()])
        ax_scatter.plot(_x, _slope * _x + _intercept, color=COLORS[_group], linewidth=3, label=_group)

    ax_scatter.set_title("More lab procedures go with longer stays", fontsize=17)
    ax_scatter.set_xlabel("Lab procedures per stay")
    ax_scatter.set_ylabel("Length of stay (days)")
    ax_scatter.legend(title="Admission type", frameon=False)
    fig_scatter.tight_layout()
    fig_scatter.savefig("figures/03_scatter_labs_vs_stay.png", dpi=200)
    fig_scatter
    return ax_scatter, fig_scatter


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Quantifying the relationship

        **Why:** the plots suggest a positive relationship, but a number makes it comparable
        and testable. For each admission type I compute:

        * **Pearson r** and its 95% confidence interval (via the Fisher z-transform), because the
          scatter plot looks roughly linear, so r is an appropriate measure;
        * **r²**, the share of variation in stay length that lab count accounts for;
        * the **slope**, in extra days per additional lab procedure.

        I also compute Spearman's rho overall, since stay length is skewed (see the histogram);
        if it agrees with Pearson, the skew is not driving the result.
        """
    )
    return


@app.cell
def _(data, np, pd, stats):
    _records = []
    for _name, _g in data.groupby("admission_type"):
        _x, _y = _g["num_lab_procedures"], _g["time_in_hospital"]
        _r, _p = stats.pearsonr(_x, _y)
        _n = len(_g)
        # 95% CI: transform r with arctanh (roughly normal), add +/-1.96 standard errors, transform back.
        _se = 1 / np.sqrt(_n - 3)
        _lo, _hi = np.tanh(np.arctanh(_r) + np.array([-1.96, 1.96]) * _se)
        _records.append(
            {
                "admission_type": _name,
                "n": _n,
                "r": _r,
                "r_squared": _r**2,
                "ci_low": _lo,
                "ci_high": _hi,
                "slope_days_per_lab": stats.linregress(_x, _y).slope,
                "p_value": _p,
            }
        )
    group_corr = pd.DataFrame(_records).set_index("admission_type")

    # Overall (all admission types pooled) for context and the Spearman robustness check.
    overall_r, overall_p = stats.pearsonr(data["num_lab_procedures"], data["time_in_hospital"])
    spearman_rho, _ = stats.spearmanr(data["num_lab_procedures"], data["time_in_hospital"])

    group_corr.round(4)
    return group_corr, overall_p, overall_r, spearman_rho


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Why this next test:** the three r values differ (about 0.46 vs. 0.32), but with a few
        hundred stays per group I need to know whether the gap is bigger than chance. Comparing two
        correlations from independent groups uses a z-test on their Fisher-transformed values.
        """
    )
    return


@app.cell
def _(group_corr, np, pd, stats):
    def compare_correlations(a, b):
        """Two-sided z-test that the correlations in groups a and b are equal."""
        _se = np.sqrt(1 / (group_corr.loc[a, "n"] - 3) + 1 / (group_corr.loc[b, "n"] - 3))
        _z = (np.arctanh(group_corr.loc[a, "r"]) - np.arctanh(group_corr.loc[b, "r"])) / _se
        return _z, 2 * (1 - stats.norm.cdf(abs(_z)))

    _pairs = [("Elective", "Emergency"), ("Elective", "Urgent"), ("Emergency", "Urgent")]
    _rows = []
    for _a, _b in _pairs:
        _z, _p = compare_correlations(_a, _b)
        _rows.append({"comparison": f"{_a} vs. {_b}", "z": _z, "p_value": _p})
    pairwise_tests = pd.DataFrame(_rows).set_index("comparison")
    pairwise_tests.round(4)
    return compare_correlations, pairwise_tests


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 3d. Group-difference chart: correlation by admission type

        **Why:** this is the visual answer to the second half of the question. Each bar is the
        correlation for one admission type, with its 95% confidence interval as the error bar.
        Intervals that do not overlap point to a real difference between groups.
        """
    )
    return


@app.cell
def _(COLORS, ORDER, group_corr, plt):
    fig_groups, ax_groups = plt.subplots(figsize=(9, 5.5))
    _vals = group_corr.loc[ORDER]
    _err_low = _vals["r"] - _vals["ci_low"]
    _err_high = _vals["ci_high"] - _vals["r"]

    ax_groups.bar(
        ORDER, _vals["r"], color=[COLORS[g] for g in ORDER], width=0.55,
        yerr=[_err_low, _err_high], capsize=7, error_kw={"linewidth": 2, "ecolor": "#111827"},
    )
    # Label each bar with its r value just above the error bar.
    for _i, _g in enumerate(ORDER):
        ax_groups.text(_i, _vals.loc[_g, "ci_high"] + 0.015, f"r = {_vals.loc[_g, 'r']:.2f}",
                       ha="center", fontsize=14, fontweight="bold")

    ax_groups.set_ylim(0, 0.62)
    ax_groups.set_title("Elective stays show the tightest lab-to-stay link", fontsize=17)
    ax_groups.set_xlabel("Admission type (error bars: 95% confidence interval)")
    ax_groups.set_ylabel("Correlation (r): lab procedures vs. days")
    fig_groups.tight_layout()
    fig_groups.savefig("figures/04_group_correlation_by_admission.png", dpi=200)
    fig_groups
    return ax_groups, fig_groups


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Conclusions

        **Why:** the notebook should end by answering the research question in plain language,
        with the numbers pulled from the computed results above (not typed by hand), and by being
        honest about what the analysis cannot show.
        """
    )
    return


@app.cell
def _(group_corr, labs_rate_by_stay, mo, overall_p, overall_r, pairwise_tests, spearman_rho):
    _e, _m, _u = (group_corr.loc[g] for g in ["Elective", "Emergency", "Urgent"])
    _short, _long = labs_rate_by_stay.iloc[0], labs_rate_by_stay.iloc[-1]
    _p_em = pairwise_tests.loc["Elective vs. Emergency", "p_value"]
    _p_eu = pairwise_tests.loc["Elective vs. Urgent", "p_value"]
    _p_mu = pairwise_tests.loc["Emergency vs. Urgent", "p_value"]

    mo.md(
        f"""
        **Answer to the research question**

        1. **Overall relationship:** more lab procedures go with longer stays
           (Pearson r = {overall_r:.2f}, p < 0.001; Spearman rho = {spearman_rho:.2f}, so the
           skewed stay-length distribution is not driving the result). Lab count accounts for
           roughly {overall_r**2:.0%} of the variation in stay length.
        2. **Differences by admission type:** the relationship is positive in every group, but
           it is strongest for **Elective** admissions (r = {_e['r']:.2f}, r² = {_e['r_squared']:.2f})
           and weaker for **Emergency** (r = {_m['r']:.2f}) and **Urgent** (r = {_u['r']:.2f}).
           Elective differs from Emergency (p = {_p_em:.4f}) and from Urgent (p = {_p_eu:.4f});
           Emergency and Urgent are indistinguishable (p = {_p_mu:.2f}).
        3. **Size of the effect:** about {_e['slope_days_per_lab']:.3f} extra days per additional
           lab test for elective stays versus about {_m['slope_days_per_lab']:.3f} for emergency stays.
        4. **Testing is front-loaded:** short stays average {_short['mean_labs_per_day']:.1f} lab
           procedures per day but long stays only {_long['mean_labs_per_day']:.1f}, even though
           long stays have more tests in total ({_long['mean_labs']:.0f} vs. {_short['mean_labs']:.0f}).
           Extra days add tests, but far fewer per day than the first days do.

        **Limits to keep in mind**

        * **Correlation is not causation.** Lab tests probably do not *cause* longer stays.
          A sicker patient plausibly needs both more tests and more days, and a longer stay
          also gives more time to order tests. This data cannot separate those explanations.
        * **Most of the variation is unexplained.** Even the strongest group has r² around
          {_e['r_squared']:.2f}, so lab count alone is a weak predictor of any one patient's stay.
        * **Dataset caveat:** this is a 3,000-stay simplified extract of the original
          130-hospital dataset, and it records no severity measure, so severity cannot be controlled for.
        """
    )
    return


if __name__ == "__main__":
    app.run()
