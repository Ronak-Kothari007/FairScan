"""Generate a project overview PDF for FairScan."""
from datetime import datetime
from pathlib import Path

from fpdf import FPDF, XPos, YPos

OUT = Path("exports/fairscan_project_overview.pdf")

NEXT = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)

PRIMARY = (35, 50, 95)
ACCENT = (61, 214, 140)
WARN = (245, 196, 81)
DANGER = (255, 75, 110)
MUTED = (110, 116, 134)
TEXT = (28, 30, 38)


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*MUTED)
        self.cell(0, 8, "FairScan - Project Overview", **NEXT)
        self.set_draw_color(220, 222, 230)
        self.line(10, 18, 200, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(
            0, 8,
            f"FairScan - Generated {datetime.now():%Y-%m-%d %H:%M}  |  Page {self.page_no()}",
            align="C",
        )


def h1(pdf: Doc, text: str):
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 12, text, **NEXT)


def h2(pdf: Doc, text: str):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 8, text, **NEXT)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.6)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 30, pdf.get_y())
    pdf.ln(3)


def body(pdf: Doc, text: str):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*TEXT)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(1)


def bullet(pdf: Doc, text: str):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*TEXT)
    pdf.cell(5)
    pdf.cell(4, 5.5, "-")
    x, y = pdf.get_x(), pdf.get_y()
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(0.5)


def kv(pdf: Doc, label: str, value: str):
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(45, 6, label, new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*TEXT)
    avail = pdf.w - pdf.r_margin - pdf.get_x()
    pdf.multi_cell(avail, 6, value, **NEXT)


def code(pdf: Doc, text: str):
    pdf.set_fill_color(245, 246, 250)
    pdf.set_text_color(40, 44, 60)
    pdf.set_font("Courier", "", 9)
    for line in text.split("\n"):
        pdf.cell(0, 5, " " + line, fill=True, **NEXT)
    pdf.ln(2)


def severity_chip(pdf: Doc, label: str, color):
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(50, 7, "  " + label, fill=True, **NEXT)
    pdf.ln(1)


def cover(pdf: Doc):
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(*PRIMARY)
    pdf.cell(0, 16, "FairScan", **NEXT)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 8, "AI Bias Detection Dashboard", **NEXT)
    pdf.ln(8)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(1.2)
    pdf.line(10, pdf.get_y(), 60, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*TEXT)
    pdf.multi_cell(
        0, 6,
        "A Streamlit application that audits tabular machine-learning models "
        "for fairness. It trains a logistic-regression classifier, computes "
        "fairness metrics with fairlearn, visualizes group-level approval "
        "rates before and after mitigation, and exports a one-click PDF "
        "audit report."
    )
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, f"Generated {datetime.now():%B %d, %Y}", **NEXT)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = Doc(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)

    cover(pdf)

    # --- Overview ----------------------------------------------------------
    pdf.add_page()
    h1(pdf, "Overview")
    body(
        pdf,
        "FairScan helps teams catch unfair model behavior before it ships. "
        "Point it at a dataset, choose a sensitive attribute (e.g. sex, "
        "race) and a target outcome, and it trains a logistic-regression "
        "model, measures three fairness metrics, runs a mitigation pass, "
        "and produces an exportable audit report."
    )

    h2(pdf, "Three core features")
    bullet(pdf, "Dataset audit: trains LogisticRegression and computes three fairlearn metrics, "
                "shown as color-coded severity cards (high/moderate/fair).")
    bullet(pdf, "Visual bias report: grouped Plotly bar chart comparing approval rates per group "
                "before and after ThresholdOptimizer mitigation, plus pills for any flagged biases.")
    bullet(pdf, "One-click PDF report: fpdf2-generated audit summary with verdict, metrics, "
                "and prioritized recommendations.")

    h2(pdf, "Stack")
    kv(pdf, "Language", "Python 3.11")
    kv(pdf, "App framework", "Streamlit (dark theme)")
    kv(pdf, "ML", "scikit-learn (LogisticRegression, ColumnTransformer, OneHotEncoder)")
    kv(pdf, "Fairness", "fairlearn (metrics + ThresholdOptimizer post-processing)")
    kv(pdf, "Charts", "Plotly (grouped bar chart, dark template)")
    kv(pdf, "PDF export", "fpdf2")
    kv(pdf, "Data handling", "pandas, numpy")

    # --- Fairness metrics --------------------------------------------------
    pdf.add_page()
    h1(pdf, "Fairness Metrics")
    body(
        pdf,
        "FairScan computes three complementary fairlearn metrics on the "
        "test split. Each one captures a different fairness failure mode."
    )

    h2(pdf, "1. Demographic Parity Difference")
    body(pdf, "Maximum gap in positive-prediction rate across groups. "
              "Answers: 'Does the model approve some groups more than others, regardless of merit?'")

    h2(pdf, "2. Equalized Odds Difference")
    body(pdf, "Maximum gap in true-positive and false-positive rates across groups. "
              "Answers: 'Are the model's mistakes distributed unevenly across groups?'")

    h2(pdf, "3. Calibration (group selection rate spread)")
    body(pdf, "Spread of group-level selection rates. "
              "Answers: 'Is the model behaving consistently across groups when conditions are equal?'")

    h2(pdf, "Severity thresholds")
    severity_chip(pdf, "> 0.10  -  HIGH RISK", DANGER)
    severity_chip(pdf, "0.05 - 0.10  -  NEEDS REVIEW", WARN)
    severity_chip(pdf, "< 0.05  -  FAIR", ACCENT)
    body(pdf, "Overall verdict: any HIGH metric -> 'High Risk'. Any MODERATE -> 'Needs Review'. "
              "Otherwise -> 'Fair'.")

    h2(pdf, "Mitigation")
    body(pdf, "FairScan re-trains predictions through fairlearn's ThresholdOptimizer with a "
              "demographic-parity constraint. The grouped bar chart shows approval rates per "
              "group before vs. after, and the dashboard reports the accuracy/fairness tradeoff "
              "(accuracy delta and DPD delta).")

    # --- Target column handling -------------------------------------------
    pdf.add_page()
    h1(pdf, "Target Column Handling")
    body(
        pdf,
        "Any column in the dataset can be used as the prediction target. "
        "FairScan auto-detects the column's shape and exposes the right "
        "control:"
    )
    bullet(pdf, "Already 0/1 numeric -> used as-is.")
    bullet(pdf, "Two-value categorical (e.g. income <=50K vs >50K) -> 'positive class' dropdown.")
    bullet(pdf, "Numeric (int or float) with many values (e.g. capital_gain, hours_per_week, age) "
                "-> cutoff slider with live class-balance preview.")
    bullet(pdf, "Numbers stored as strings ('5', '10') -> auto-coerced to numeric, gets the slider.")
    bullet(pdf, "Categorical with 3+ categories - including 30+ (e.g. native_country) "
                "-> 'positive class' dropdown showing value counts and share for each option, "
                "so you can pick one with enough samples.")

    h2(pdf, "Sensitive-attribute handling")
    bullet(pdf, "Tiny groups (<30 rows) are dropped before training so metrics aren't dominated by noise.")
    bullet(pdf, "If ThresholdOptimizer can't fit (e.g. only one class per group), the app reports the "
                "error and shows pre-mitigation rates instead of crashing.")

    # --- Default dataset --------------------------------------------------
    pdf.add_page()
    h1(pdf, "Default Dataset")
    body(
        pdf,
        "On first run, FairScan loads the UCI Adult Income dataset from "
        "archive.ics.uci.edu. The result is cached in Streamlit so subsequent "
        "runs are instant. If the network fetch fails, the app falls back to "
        "a synthetic biased dataset so the demo always works."
    )
    h2(pdf, "Custom data")
    body(pdf, "Users can upload any CSV from the sidebar. After upload, every column is selectable "
              "as either the sensitive attribute or the target.")

    h2(pdf, "Per-group breakdown table")
    body(pdf, "Beneath the chart, FairScan renders a per-group table with sample size, base rate "
              "(actual positives in the data), model approval rate before and after mitigation, and "
              "true/false positive rates. Comparing base rate vs. approval rate is the single most "
              "useful diagnostic - it shows whether the model is amplifying real disparities or "
              "inventing them.")

    # --- Project layout ---------------------------------------------------
    pdf.add_page()
    h1(pdf, "Project Layout")
    code(pdf,
         "FairScan/\n"
         "  app.py                          # main Streamlit application\n"
         "  requirements.txt                # Python dependencies\n"
         "  .streamlit/config.toml          # dark theme + server config\n"
         "  replit.md                       # project notes\n"
         "  artifacts/\n"
         "    api-server/\n"
         "      .replit-artifact/\n"
         "        artifact.toml             # registers FairScan service\n"
         "    mockup-sandbox/               # design canvas (unrelated)\n"
         "  scripts/\n"
         "    build_project_pdf.py          # this PDF generator\n"
         "  exports/\n"
         "    fairscan_project_overview.pdf # output\n")

    h2(pdf, "Key file: app.py")
    body(pdf, "Single-file Streamlit app (~890 lines) organized into:")
    bullet(pdf, "Sidebar - dataset source, sensitive attribute, target column, positive-class control.")
    bullet(pdf, "run_audit() - cached function that trains LogisticRegression, computes fairlearn "
                "metrics, runs ThresholdOptimizer, and builds the per-group breakdown.")
    bullet(pdf, "Main panel - severity metric cards, grouped Plotly chart, accuracy/fairness tradeoff "
                "metrics, per-group table, flagged-bias pills, and PDF download button.")
    bullet(pdf, "build_pdf() - fpdf2 audit-report builder used by the download button.")

    # --- Deployment -------------------------------------------------------
    pdf.add_page()
    h1(pdf, "Deployment")
    body(
        pdf,
        "FairScan runs in a pnpm monorepo. Deployment configuration lives "
        "inside each registered artifact's artifact.toml file, not in .replit. "
        "FairScan is registered as a 'web' service inside the api-server artifact:"
    )
    bullet(pdf, "previewPath = '/' (the Streamlit app owns the root URL).")
    bullet(pdf, "kind = 'web' (the publisher recognizes this as deployable).")
    bullet(pdf, "Two services: 'FairScan' (Streamlit, port 5000, path /) and "
                "'API Server' (Node Express, port 8080, path /api - currently unused, kept for "
                "future expansion).")

    h2(pdf, "Production run command")
    code(pdf,
         "streamlit run ../../app.py \\\n"
         "  --server.port 5000 \\\n"
         "  --server.address 0.0.0.0 \\\n"
         "  --server.headless true \\\n"
         "  --browser.gatherUsageStats false")
    body(pdf, "The path is ../../app.py because the artifact's working directory is "
              "artifacts/api-server/.")

    h2(pdf, "Deployment target")
    bullet(pdf, "Type: autoscale (scales to zero when idle, ideal for a demo dashboard).")
    bullet(pdf, "TLS, hosting, and health checks are handled by the platform.")
    bullet(pdf, "Output URL: a .replit.app subdomain (or a custom domain if configured).")

    # --- Issues fixed -----------------------------------------------------
    pdf.add_page()
    h1(pdf, "Notable Fixes Along the Way")

    h2(pdf, "Capital-gain (and similar) targets refused to run")
    body(pdf, "Cause: the sidebar only treated FLOAT columns as continuous. capital_gain is "
              "integer-typed with hundreds of distinct values, so it fell through to a "
              "'too many unique values' error. "
              "Fix: any numeric column - integer or float - that isn't already 0/1 now gets a "
              "cutoff slider with a live class-balance preview.")

    h2(pdf, "Categorical columns with 30+ categories rejected")
    body(pdf, "Cause: native_country (~42 categories) hit the same dead end. "
              "Fix: categorical targets with any number of unique values now get a positive-class "
              "dropdown that shows row counts and share for each option, so you can pick one "
              "with enough samples.")

    h2(pdf, "Publisher said 'nothing to publish'")
    body(pdf, "Cause: the artifact was registered with kind = 'api', which the publisher does "
              "not list as a deployable kind. "
              "Fix: changed the registered kind to 'web' so the publisher recognizes it.")

    h2(pdf, "Streamlit deprecation warnings spamming logs")
    body(pdf, "Cause: use_container_width=True/False is deprecated in Streamlit 1.56; fpdf2 "
              "deprecated the ln= and dest= parameters. "
              "Fix: switched to width='stretch' / 'content' across all sites; replaced ln=True "
              "with new_x=XPos.LMARGIN, new_y=YPos.NEXT and removed dest='S'.")

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
