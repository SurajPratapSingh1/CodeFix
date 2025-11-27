#!/usr/bin/env python3
"""
db_analyzer.py
Automated SQLite database analyzer -> visualizations -> PDF report -> email sender.

Usage:
    python3 db_analyzer.py --db data.db --email recipient@example.com --team "Team Name"
Requires:
    export SMTP_SERVER="smtp.gmail.com"
    export SMTP_PORT="587"
    export SENDER_EMAIL="yourgmail@gmail.com"
    export SENDER_PASSWORD="YOUR_16_CHAR_APP_PASSWORD"
"""

import os
import sys
import argparse
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
import mimetypes
from PIL import Image as PILImage

# ReportLab for PDF creation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table as RLTable,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import warnings
from pandas.errors import ParserWarning

# Silence datetime inference warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ParserWarning)


# ---------- Helper utilities ----------

def ensure_outdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def save_fig(fig, path, dpi=200):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def find_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    rows = cur.fetchall()
    return [r[0] for r in rows]

def get_table_schema(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}');")
    cols = cur.fetchall()
    return [{"cid": c[0], "name": c[1], "type": c[2], "notnull": c[3], "dflt": c[4], "pk": c[5]} for c in cols]

def try_parse_datetime_series(s):
    """Try to parse a series (or array-like) into datetimes. Return parsed Series if good, else None."""
    try:
        ser = pd.to_datetime(s, errors="coerce")
        non_null = ser.notnull().sum()
        # heuristics: at least 3 parsed or >=3% non-null
        if non_null >= max(3, int(len(s) * 0.03)):
            return ser
    except Exception:
        pass
    return None

# ---------- Analysis functions ----------

def analyze_table(conn, table):
    df = pd.read_sql_query(f"SELECT * FROM '{table}';", conn)
    info = {}
    info["table"] = table
    info["rows"] = len(df)
    info["cols"] = len(df.columns)
    info["columns"] = []
    info["null_counts"] = df.isnull().sum().to_dict()
    info["duplicate_count"] = int(df.duplicated().sum())
    for col in df.columns:
        col_series = df[col]
        col_dtype = str(col_series.dtype)
        try:
            top = col_series.dropna().value_counts().head(5).to_dict()
        except Exception:
            top = {}
        col_summary = {
            "name": col,
            "dtype": col_dtype,
            "nulls": int(col_series.isnull().sum()),
            "unique": int(col_series.nunique(dropna=True)),
            "top": top,
        }
        if pd.api.types.is_numeric_dtype(col_series):
            if not col_series.dropna().empty:
                col_summary["min"] = float(col_series.min())
                col_summary["max"] = float(col_series.max())
                col_summary["mean"] = float(col_series.mean())
                col_summary["std"] = float(col_series.std())
            else:
                col_summary["min"] = col_summary["max"] = col_summary["mean"] = col_summary["std"] = None
        info["columns"].append(col_summary)
    info["sample"] = df.head(5).to_dict(orient="records")
    info["df"] = df
    return info

# ---------- Visualizations ----------

def create_distribution_chart(info, outdir):
    df = info["df"]
    table = info["table"]
    if df.empty:
        return None
    cat_col = None
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_object_dtype(s) or s.dtype == "bool" or (s.nunique(dropna=True) <= 20 and s.nunique(dropna=True) > 1):
            cat_col = c
            break
    if cat_col is None:
        # fallback: choose first column with >1 unique values
        for c in df.columns:
            if df[c].nunique(dropna=True) > 1:
                cat_col = c
                break
    if cat_col is None:
        return None

    cnt = df[cat_col].fillna("NULL").value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=cnt.values, y=cnt.index, ax=ax)
    ax.set_title(f"Top values in '{cat_col}' (table: {table})")
    ax.set_xlabel("Count")
    ax.set_ylabel(cat_col)
    path = os.path.join(outdir, f"{table}_distribution_{sanitize_filename(cat_col)}.png")
    save_fig(fig, path)
    return path, f"Distribution of {cat_col} in {table}"

def create_trend_chart(infos, outdir):
    best = None
    for info in infos:
        df = info["df"]
        for c in df.columns:
            ser = df[c]
            if ser.dropna().empty:
                continue
            parsed = try_parse_datetime_series(ser.astype(str))
            if parsed is not None:
                if best is None or info["rows"] > best["info"]["rows"]:
                    best = {"info": info, "col": c, "parsed": parsed}
    if best:
        df = best["info"]["df"].copy()
        ser = pd.to_datetime(df[best["col"]], errors="coerce")
        df["_parsed_date"] = ser
        grp = df.dropna(subset=["_parsed_date"]).set_index("_parsed_date").resample("M").size()
        if not grp.empty:
            fig, ax = plt.subplots(figsize=(10,4))
            grp.plot(ax=ax, marker="o")
            ax.set_title(f"Trend over time for '{best['col']}' ({best['info']['table']})")
            ax.set_ylabel("Record count per month")
            ax.set_xlabel("Month")
            path = os.path.join(outdir, f"{best['info']['table']}_trend_{sanitize_filename(best['col'])}.png")
            save_fig(fig, path)
            return path, f"Trend by {best['col']} in {best['info']['table']}"
    # fallback
    tables = [info["table"] for info in infos]
    counts = [info["rows"] for info in infos]
    if len(tables) == 0:
        return None
    fig, ax = plt.subplots(figsize=(10,4))
    sns.barplot(x=counts, y=tables, ax=ax)
    ax.set_title("Row counts per table (fallback trend)")
    ax.set_xlabel("Row count")
    ax.set_ylabel("Table")
    path = os.path.join(outdir, "tables_row_counts_trend.png")
    save_fig(fig, path)
    return path, "Row counts per table"

def create_relationship_chart(info, outdir):
    df = info["df"]
    table = info["table"]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        numdf = df[numeric_cols].dropna()
        if numdf.shape[0] >= 2:
            corr = numdf.corr()
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax, cbar_kws={"shrink":.8})
            ax.set_title(f"Correlation heatmap ({table})")
            path = os.path.join(outdir, f"{table}_correlation_heatmap.png")
            save_fig(fig, path)
            return path, f"Correlation heatmap for numeric columns in {table}"
    if len(numeric_cols) >= 2:
        xcol, ycol = numeric_cols[:2]
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(x=df[xcol], y=df[ycol], ax=ax)
        ax.set_title(f"Scatter: {xcol} vs {ycol} ({table})")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        path = os.path.join(outdir, f"{table}_scatter_{sanitize_filename(xcol)}_vs_{sanitize_filename(ycol)}.png")
        save_fig(fig, path)
        return path, f"Scatter {xcol} vs {ycol} in {table}"
    return None

def sanitize_filename(s):
    return "".join([c if c.isalnum() or c in "-_." else "_" for c in str(s)])

# ---------- PDF helpers ----------

def add_image_to_story(story, img_path, max_width_cm=16, max_height_cm=10):
    """
    Add an image to the ReportLab story, preserving aspect ratio and scaling to fit max dimensions.
    """
    if not os.path.exists(img_path):
        return
    try:
        with PILImage.open(img_path) as pil:
            w_px, h_px = pil.size
            dpi = pil.info.get("dpi", (72,72))[0] if pil.info.get("dpi") else 72
            # convert px to cm (1 inch = 2.54 cm; px/dpi inches)
            w_in = w_px / dpi
            h_in = h_px / dpi
            w_cm = w_in * 2.54
            h_cm = h_in * 2.54
    except Exception:
        # fallback to some default size
        w_cm, h_cm = max_width_cm, max_height_cm

    max_w = max_width_cm * cm
    max_h = max_height_cm * cm
    # compute scale preserving aspect ratio
    scale = min(max_w / (w_cm*cm) if w_cm>0 else 1, max_h / (h_cm*cm) if h_cm>0 else 1)
    if scale <= 0:
        scale = 1.0
    draw_w = (w_cm*cm) * scale
    draw_h = (h_cm*cm) * scale
    im = RLImage(img_path, width=draw_w, height=draw_h)
    story.append(im)
    story.append(Spacer(1, 8))

# ---------- Report creation (PDF) ----------

def create_pdf_report(out_pdf_path, summary, table_infos, chart_records, team_name="AI CODEFIX Team"):
    doc = SimpleDocTemplate(out_pdf_path, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("Title", parent=styles["Title"], alignment=1, fontSize=20, leading=22)
    story.append(Paragraph("Database Analysis Report", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Team:</b> {team_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated:</b> {summary['analysis_date']}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # DATABASE SUMMARY
    story.append(Paragraph("<b>=== DATABASE SUMMARY ===</b>", styles["Heading3"]))
    story.append(Paragraph(f"- Total Tables: {summary['total_tables']}", styles["Normal"]))
    story.append(Paragraph(f"- Total Records (all tables sum): {summary['total_records']}", styles["Normal"]))
    story.append(Paragraph(f"- Database file: {summary.get('db_file','N/A')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # KEY INSIGHTS
    story.append(Paragraph("<b>=== KEY INSIGHTS ===</b>", styles["Heading3"]))
    if summary.get("insights"):
        for i, ins in enumerate(summary["insights"], start=1):
            story.append(Paragraph(f"{i}. {ins}", styles["Normal"]))
    else:
        story.append(Paragraph("No automated insights found.", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Charts
    story.append(Paragraph("<b>Charts</b>", styles["Heading3"]))
    for cr in chart_records:
        story.append(Paragraph(f"<b>{cr.get('title','Chart')}</b>", styles["Normal"]))
        img_path = cr.get("path")
        if img_path and os.path.exists(img_path):
            add_image_to_story(story, img_path)
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # Detailed per-table analysis
    story.append(Paragraph("<b>Detailed per-table analysis</b>", styles["Heading2"]))
    for info in table_infos:
        story.append(Paragraph(f"Table: <b>{info['table']}</b> (rows: {info['rows']}, cols: {info['cols']})", styles["Heading4"]))
        # columns summary table
        rows = [["Column", "Dtype", "Nulls", "Unique", "Top values (sample)"]]
        for c in info["columns"]:
            top_sample = ", ".join([f"{k} ({v})" for k, v in list(c["top"].items())[:3]]) if c.get("top") else ""
            rows.append([c["name"], c.get("dtype", ""), str(c.get("nulls","")), str(c.get("unique","")), top_sample])
        t = RLTable(rows, repeatRows=1, colWidths=[4*cm, 3*cm, 2*cm, 2*cm, 6*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f0f0")),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ]))
        story.append(t)
        story.append(Spacer(1,8))
        # sample rows
        story.append(Paragraph("Sample rows (up to 5):", styles["Normal"]))
        try:
            sample_df = pd.DataFrame(info.get("sample", []))
            if not sample_df.empty:
                data = [list(sample_df.columns)] + sample_df.fillna("").astype(str).values.tolist()
                t2 = RLTable(data, colWidths=None)
                t2.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey)]))
                story.append(t2)
        except Exception:
            pass
        story.append(Spacer(1,12))

    story.append(PageBreak())
    story.append(Paragraph("<b>Recommendations</b>", styles["Heading2"]))
    recs = [
        "Address columns with high null counts: drop/define defaults or improve ETL.",
        "Investigate duplicate rows in tables with duplicate_count > 0.",
        "If trend charts use timestamp columns, ensure timestamps are stored in ISO format for robust analysis.",
        "Consider adding foreign key constraints where relationships exist.",
    ]
    for r in recs:
        story.append(Paragraph(f"- {r}", styles["Normal"]))

    story.append(Spacer(1,12))
    story.append(Paragraph("Best regards,", styles["Normal"]))
    story.append(Paragraph(team_name, styles["Normal"]))

    # Build PDF
    try:
        doc.build(story)
    except Exception as e:
        print("ERROR: Failed to build PDF:", e)
        raise

# ---------- Email sending ----------

def send_email_with_attachments(smtp_server, smtp_port, sender_email, sender_password, recipient_email, subject, body_text, attach_paths, sender_name="DB Analyst Bot"):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((sender_name, sender_email))
    msg["To"] = recipient_email
    msg.set_content(body_text)

    for path in attach_paths:
        if not path or not os.path.exists(path):
            continue
        ctype, encoding = mimetypes.guess_type(path)
        if ctype is None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(path, "rb") as f:
            data = f.read()
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(path))

    try:
        smtp_port = int(smtp_port)
        # Use SSL for 465, STARTTLS for 587, else plain connect
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
            server.login(sender_email, sender_password)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.ehlo()
            if smtp_port == 587:
                server.starttls()
                server.ehlo()
            server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)

# ---------- Main flow ----------

def main():
    parser = argparse.ArgumentParser(description="Automated DB Analyzer -> PDF -> Email")
    parser.add_argument("--db", required=True, help="Path to SQLite database file (data.db)")
    parser.add_argument("--email", required=True, help="Recipient email address")
    parser.add_argument("--team", default="AI CODEFIX Team", help="Team name for report footer")
    parser.add_argument("--outdir", default="output", help="Output directory for report and charts")
    parser.add_argument("--no-email", action="store_true", help="Don't send email; just save files")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    dbpath = args.db
    if not os.path.exists(dbpath):
        print(f"ERROR: Database file not found: {dbpath}")
        sys.exit(1)

    conn = sqlite3.connect(dbpath)
    try:
        tables = find_tables(conn)
        infos = []
        total_records = 0
        for t in tables:
            try:
                info = analyze_table(conn, t)
                infos.append(info)
                total_records += info["rows"]
            except Exception as e:
                print(f"Warning: failed analyzing table {t}: {e}")

        summary = {
            "analysis_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "total_tables": len(tables),
            "total_records": total_records,
            "db_file": os.path.abspath(dbpath),
        }

        # Generate charts
        chart_records = []
        # Chart 1
        for info in infos:
            try:
                res = create_distribution_chart(info, outdir)
                if res:
                    chart_records.append({"path": res[0], "title": res[1]})
                    break
            except Exception:
                pass

        # Chart 2
        try:
            res = create_trend_chart(infos, outdir)
            if res:
                chart_records.append({"path": res[0], "title": res[1]})
        except Exception as e:
            print("Warning: trend chart failed:", e)

        # Chart 3
        sorted_infos = sorted(infos, key=lambda x: x["rows"], reverse=True)
        for info in sorted_infos:
            try:
                res = create_relationship_chart(info, outdir)
                if res:
                    chart_records.append({"path": res[0], "title": res[1]})
                    break
            except Exception:
                pass

        # ensure at least 3 charts if possible
        if len(chart_records) < 3:
            for info in infos:
                try:
                    res = create_distribution_chart(info, outdir)
                    if res and all(r["path"] != res[0] for r in chart_records):
                        chart_records.append({"path": res[0], "title": res[1]})
                        if len(chart_records) >= 3:
                            break
                except Exception:
                    pass

        # Insights
        insights = []
        if summary["total_tables"] == 0:
            insights.append("Database contains no user tables.")
        else:
            if infos:
                largest = max(infos, key=lambda x: x["rows"])
                insights.append(f"Largest table is '{largest['table']}' with {largest['rows']} rows.")
                dup_tables = [f"{i['table']} ({i['duplicate_count']})" for i in infos if i.get("duplicate_count",0) > 0]
                if dup_tables:
                    insights.append(f"Duplicate rows detected in: {', '.join(dup_tables)}.")
                null_issues = []
                for i in infos:
                    for c in i["columns"]:
                        if c.get("nulls",0) > max(5, 0.1 * max(1, i["rows"])):
                            null_issues.append(f"{i['table']}.{c['name']}({c['nulls']})")
                if null_issues:
                    insights.append(f"Columns with many nulls: {', '.join(null_issues[:5])}...")
                else:
                    insights.append("No major null-value issues detected in top columns.")
            else:
                insights.append("No detailed table info available.")
        summary["insights"] = insights

        # Create PDF
        pdf_path = os.path.join(outdir, f"db_analysis_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf")
        create_pdf_report(pdf_path, summary, infos, chart_records, team_name=args.team)

        # Email body
        body_lines = [
            f"Dear Recipient,",
            "",
            "Please find the automated database analysis report attached.",
            "",
            "=== DATABASE SUMMARY ===",
            f"- Total Tables: {summary['total_tables']}",
            f"- Total Records: {summary['total_records']}",
            f"- Analysis Date: {summary['analysis_date']}",
            "",
            "=== KEY INSIGHTS ===",
        ]
        for i, ins in enumerate(insights, start=1):
            body_lines.append(f"{i}. {ins}")
        body_lines.append("")
        body_lines.append("Attached: " + ", ".join([os.path.basename(pdf_path)] + [os.path.basename(c["path"]) for c in chart_records if c.get("path")]))
        body_lines.append("")
        body_lines.append("Best regards,")
        body_lines.append(args.team)
        body_text = "\n".join(body_lines)

        print(f"Report saved to: {pdf_path}")
        for c in chart_records:
            print(f"Chart: {c['title']} -> {c['path']}")

        if args.no_email:
            print("Skipping email ( --no-email )")
        else:
            SMTP_SERVER = os.environ.get("SMTP_SERVER")
            SMTP_PORT = os.environ.get("SMTP_PORT")
            SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
            SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
            if not all([SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD]):
                print("ERROR: Missing SMTP configuration in environment variables. Set SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD.")
                print("Report saved locally; you can send it manually.")
            else:
                subject = f"Database Analysis Report - {args.team}"
                attachments = [pdf_path] + [c["path"] for c in chart_records if c.get("path")]
                succ, err = send_email_with_attachments(SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD, args.email, subject, body_text, attachments, sender_name=args.team)
                if succ:
                    print(f"Email sent to {args.email}")
                else:
                    print(f"Failed to send email: {err}")
                    print("Report saved locally; you can send it manually.")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
