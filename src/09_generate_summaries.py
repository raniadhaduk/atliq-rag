"""
Step 8: Generate summary documents - pre-compute aggregate facts
(counts, totals, averages) from tabular data and save them as their
own small markdown document.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")


def summarize_hr_csv():
    csv_path = DATA_DIR / "hr" / "hr_data.csv"

    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return

    df = pd.read_csv(csv_path)

    total_employees = len(df)
    dept_counts = df["department"].value_counts().to_dict()
    avg_salary = df["salary"].mean()
    avg_attendance = df["attendance_pct"].mean()
    avg_performance = df["performance_rating"].mean()
    location_counts = df["location"].value_counts().to_dict()

    lines = [
        "# HR Data - Summary Statistics",
        "",
        "**Department:** HR",
        "**Access Level:** HR Team, C-Level Executives only",
        "**Classification:** Confidential",
        "",
        "This document contains pre-computed aggregate statistics from the "
        "full HR employee dataset, for questions about totals, counts, and averages.",
        "",
        f"## Total Employee Count",
        f"There are **{total_employees}** total employees in the HR dataset.",
        "",
        "## Employee Count by Department",
    ]

    for dept, count in dept_counts.items():
        lines.append(f"- {dept}: {count} employees")

    lines += [
        "",
        "## Salary Statistics",
        f"- Average salary across all employees: {avg_salary:,.2f}",
        f"- Minimum salary: {df['salary'].min():,.2f}",
        f"- Maximum salary: {df['salary'].max():,.2f}",
        "",
        "## Performance & Attendance",
        f"- Average attendance percentage: {avg_attendance:.2f}%",
        f"- Average performance rating: {avg_performance:.2f} (out of 5)",
        "",
        "## Employee Count by Location",
    ]

    for loc, count in location_counts.items():
        lines.append(f"- {loc}: {count} employees")

    summary_text = "\n".join(lines)

    output_path = DATA_DIR / "hr" / "hr_summary_stats.md"
    output_path.write_text(summary_text, encoding="utf-8")

    print(f"  Wrote summary to: {output_path}")
    print(f"  Total employees computed: {total_employees}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating summary documents from tabular data...")
    print("=" * 60)

    summarize_hr_csv()

    print("\nDone. Re-run the ingestion + chunking + embedding pipeline")
    print("(python src\\03_embed_and_store.py) to pick up the new summary doc.")