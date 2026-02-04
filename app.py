from __future__ import annotations

import io
import os
import sys
import sqlite3
import threading
import webbrowser
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from flask import Flask, render_template, request, redirect, url_for, send_file, abort, session

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

APP_TITLE = "E-Broiler Dashboard"
AGE_DAYS = 35  # enter data once per 35 days
SECONDS_PER_DAY = 86400
DEFAULT_USERS = [
    ("SACHIN", "SACHIN2001@", "user", "User Login"),
    ("HASINTHA", "HASINTHA2007@", "master", "Master Creator Login"),
]


# -----------------------------
# EXE-safe path helper
# -----------------------------
def get_data_path(filename: str) -> str:
    """
    When running as an EXE (PyInstaller), store DB next to the EXE.
    When running as normal python, store DB in the project folder.
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.abspath(".")
    return os.path.join(base, filename)


DB_FILE = get_data_path("broiler.db")

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "ebroiler-dashboard-local-secret")


def now_ts() -> int:
    return int(datetime.now().timestamp())


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_ts(ts: Optional[int]) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")


def seconds_to_days(seconds: Optional[int]) -> Optional[float]:
    if seconds is None:
        return None
    return round(max(int(seconds), 0) / SECONDS_PER_DAY, 2)


def get_db_user(username: str) -> Optional[sqlite3.Row]:
    with db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username=?;",
            (username.upper(),),
        ).fetchone()


def is_license_expired(user_row: sqlite3.Row) -> bool:
    if user_row["role"] == "master":
        return False
    exp = user_row["license_expires_at"]
    return exp is not None and int(exp) <= now_ts()


def get_current_user() -> Optional[Dict[str, Any]]:
    username = session.get("username")
    if not username:
        return None

    user = get_db_user(username)
    if not user:
        session.clear()
        return None

    if is_license_expired(user):
        session.clear()
        session["login_error"] = "Your license has expired. Contact the master account."
        return None

    exp = user["license_expires_at"]
    remaining = (int(exp) - now_ts()) if exp is not None else None
    return {
        "username": user["username"],
        "role": user["role"],
        "role_label": user["role_label"],
        "license_expires_at": exp,
        "license_remaining_days": seconds_to_days(remaining),
    }


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not get_current_user():
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapped


def master_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if not user:
            return redirect(url_for("login"))
        if user["role"] != "master":
            abort(403)
        return fn(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_auth_context() -> Dict[str, Any]:
    user = get_current_user()
    return {
        "current_user": user,
        "can_edit": bool(user and user["role"] == "master"),
        "can_manage_users": bool(user and user["role"] == "master"),
    }


# -----------------------------
# DB
# -----------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_no TEXT NOT NULL,
            out_date TEXT NOT NULL,

            input_birds INTEGER NOT NULL,
            cull_birds INTEGER NOT NULL,

            feed_bb_kg REAL NOT NULL,
            feed_bf_kg REAL NOT NULL,

            total_weight_kg REAL NOT NULL,

            total_cost_lkr REAL,
            price_per_kg_lkr REAL,

            profit_per_bird_lkr REAL,
            created_by TEXT,
            created_at TEXT,
            updated_by TEXT,
            updated_at TEXT
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            role_label TEXT NOT NULL,
            license_expires_at INTEGER,
            created_by TEXT,
            created_at TEXT NOT NULL,
            updated_by TEXT,
            updated_at TEXT
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_out_date ON batches(out_date);")

        # Safe migration (if old DB existed)
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(batches);").fetchall()]
        if "profit_per_bird_lkr" not in cols:
            try:
                conn.execute("ALTER TABLE batches ADD COLUMN profit_per_bird_lkr REAL;")
            except Exception:
                pass
        if "created_by" not in cols:
            conn.execute("ALTER TABLE batches ADD COLUMN created_by TEXT;")
        if "created_at" not in cols:
            conn.execute("ALTER TABLE batches ADD COLUMN created_at TEXT;")
        if "updated_by" not in cols:
            conn.execute("ALTER TABLE batches ADD COLUMN updated_by TEXT;")
        if "updated_at" not in cols:
            conn.execute("ALTER TABLE batches ADD COLUMN updated_at TEXT;")

        # Seed default accounts if they do not exist.
        for username, password, role, role_label in DEFAULT_USERS:
            exists = conn.execute(
                "SELECT 1 FROM users WHERE username=?;",
                (username,),
            ).fetchone()
            if not exists:
                conn.execute(
                    """
                    INSERT INTO users (
                        username, password, role, role_label,
                        license_expires_at, created_by, created_at, updated_by, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        password,
                        role,
                        role_label,
                        None,
                        "SYSTEM",
                        now_iso(),
                        "SYSTEM",
                        now_iso(),
                    ),
                )


# -----------------------------
# Helpers
# -----------------------------
def safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


@dataclass
class KPI:
    mortality_birds: int
    mortality_pct: float          # 0..1
    livability_pct: float         # 0..100
    total_feed_kg: float
    avg_feed_per_bird_kg: float
    avg_weight_per_bird_kg: float
    fcr: float
    pi: float                     # should be ~250..400+
    score: float
    profit_per_bird_lkr: Optional[float]


def compute_kpi(r: sqlite3.Row) -> KPI:
    start = int(r["input_birds"])
    cull = int(r["cull_birds"])
    bb = float(r["feed_bb_kg"])
    bf = float(r["feed_bf_kg"])
    tw = float(r["total_weight_kg"])
    profit_per_bird = r["profit_per_bird_lkr"]

    mort_birds = max(start - cull, 0)
    mort_pct = safe_div(mort_birds, start)
    liv_pct = safe_div(cull, start) * 100.0

    total_feed = bb + bf
    avg_feed = safe_div(total_feed, cull)
    avg_w = safe_div(tw, cull)

    fcr = safe_div(avg_feed, avg_w) if avg_w else 0.0

    # ✅ PI FIX (EPEF style): (Livability% × AvgW × 100) / (AgeDays × FCR)
    pi = safe_div(liv_pct * avg_w * 100.0, (AGE_DAYS * fcr)) if fcr else 0.0

    # Simple score (for sorting / quick performance view)
    # Benchmarks you can tune
    w_bench = 2.2
    fcr_bench = 1.55
    mort_bench = 0.05
    pi_bench = 350.0

    w_part = min(1.0, safe_div(avg_w, w_bench)) * 30
    fcr_part = min(1.0, safe_div(fcr_bench, fcr)) * 30 if fcr else 0
    mort_part = min(1.0, safe_div(mort_bench, mort_pct)) * 20 if mort_pct else 20
    pi_part = min(1.0, safe_div(pi, pi_bench)) * 20

    score = round(w_part + fcr_part + mort_part + pi_part, 1)

    return KPI(
        mortality_birds=mort_birds,
        mortality_pct=mort_pct,
        livability_pct=liv_pct,
        total_feed_kg=total_feed,
        avg_feed_per_bird_kg=avg_feed,
        avg_weight_per_bird_kg=avg_w,
        fcr=fcr,
        pi=pi,
        score=score,
        profit_per_bird_lkr=float(profit_per_bird) if profit_per_bird is not None else None,
    )


def fetch_batches_latest_first() -> List[sqlite3.Row]:
    with db() as conn:
        return conn.execute("SELECT * FROM batches ORDER BY out_date DESC, id DESC;").fetchall()


def fetch_batches_oldest_first() -> List[sqlite3.Row]:
    with db() as conn:
        return conn.execute("SELECT * FROM batches ORDER BY out_date ASC, id ASC;").fetchall()


def get_batch(batch_id: int) -> sqlite3.Row:
    with db() as conn:
        r = conn.execute("SELECT * FROM batches WHERE id=?;", (batch_id,)).fetchone()
    if not r:
        abort(404)
    return r


def parse_date(value: str) -> Optional[date]:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def get_previous_batch(current: sqlite3.Row) -> Optional[sqlite3.Row]:
    with db() as conn:
        return conn.execute(
            """
            SELECT * FROM batches
            WHERE (out_date < ?) OR (out_date = ? AND id < ?)
            ORDER BY out_date DESC, id DESC
            LIMIT 1;
            """,
            (current["out_date"], current["out_date"], current["id"]),
        ).fetchone()


def build_chart_payload(rows_oldest_first: List[sqlite3.Row]) -> Dict[str, Any]:
    labels, avg_w, fcr, mort, pi = [], [], [], [], []
    for r in rows_oldest_first:
        k = compute_kpi(r)
        labels.append(str(r["batch_no"]))
        avg_w.append(round(k.avg_weight_per_bird_kg, 3))
        fcr.append(round(k.fcr, 3))
        mort.append(round(k.mortality_pct * 100.0, 2))
        pi.append(round(k.pi, 1))
    return {"labels": labels, "avg_w": avg_w, "fcr": fcr, "mort": mort, "pi": pi}


def dashboard_stats(rows_latest_first: List[sqlite3.Row]) -> Dict[str, Any]:
    if not rows_latest_first:
        return {"total": 0, "best_score": "-", "best_fcr": "-", "best_pi": "-", "lowest_mort": "-"}

    best_score = None
    best_fcr = None
    best_pi = None
    lowest_mort = None

    for r in rows_latest_first:
        k = compute_kpi(r)
        if best_score is None or k.score > best_score[0]:
            best_score = (k.score, r["batch_no"])
        if best_fcr is None or k.fcr < best_fcr[0]:
            best_fcr = (k.fcr, r["batch_no"])
        if best_pi is None or k.pi > best_pi[0]:
            best_pi = (k.pi, r["batch_no"])
        if lowest_mort is None or k.mortality_pct < lowest_mort[0]:
            lowest_mort = (k.mortality_pct, r["batch_no"])

    return {
        "total": len(rows_latest_first),
        "best_score": best_score[1],
        "best_fcr": best_fcr[1],
        "best_pi": best_pi[1],
        "lowest_mort": lowest_mort[1],
    }


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
@login_required
def dashboard():
    rows_latest = fetch_batches_latest_first()
    rows_oldest = list(reversed(rows_latest))

    # trend vs previous batch (chronological)
    prev_by_id: Dict[int, Optional[sqlite3.Row]] = {}
    prev = None
    for r in rows_oldest:
        prev_by_id[r["id"]] = prev
        prev = r

    enriched = []
    for r in rows_latest:
        k = compute_kpi(r)
        prev_r = prev_by_id.get(r["id"])
        trend = None
        if prev_r:
            pk = compute_kpi(prev_r)
            trend = {
                "d_w": round(k.avg_weight_per_bird_kg - pk.avg_weight_per_bird_kg, 3),
                "d_fcr": round(k.fcr - pk.fcr, 3),
                "d_mort": round((k.mortality_pct - pk.mortality_pct) * 100.0, 2),
                "d_pi": round(k.pi - pk.pi, 1),
            }
        enriched.append({"row": r, "kpi": k, "trend": trend})

    chart = build_chart_payload(rows_oldest) if rows_latest else None
    stats = dashboard_stats(rows_latest)

    return render_template(
        "dashboard.html",
        title=APP_TITLE,
        rows=enriched,
        stats=stats,
        chart=chart,
    )


@app.route("/about")
@login_required
def about():
    return render_template(
        "about.html",
        title=f"About - {APP_TITLE}",
    )


@app.route("/new", methods=["GET", "POST"])
@login_required
def new_batch():
    actor = get_current_user()
    if request.method == "GET":
        return render_template("new_batch.html", title=APP_TITLE)

    out_date = request.form.get("out_date", "").strip()
    batch_no = request.form.get("batch_no", "").strip()

    def to_int(name: str) -> int:
        v = request.form.get(name, "0") or "0"
        return int(float(v))

    def to_float(name: str) -> float:
        v = request.form.get(name, "0") or "0"
        return float(v)

    input_birds = to_int("input_birds")
    cull_birds = to_int("cull_birds")
    feed_bb_kg = to_float("feed_bb_kg")
    feed_bf_kg = to_float("feed_bf_kg")
    total_weight_kg = to_float("total_weight_kg")

    profit_per_bird_lkr = request.form.get("profit_per_bird_lkr") or None
    total_cost_lkr = request.form.get("total_cost_lkr") or None
    price_per_kg_lkr = request.form.get("price_per_kg_lkr") or None

    if not out_date or not batch_no:
        return render_template("new_batch.html", title=APP_TITLE, error="Date and Batch Number are required.")

    if input_birds <= 0 or cull_birds <= 0 or total_weight_kg <= 0 or (feed_bb_kg + feed_bf_kg) <= 0:
        return render_template("new_batch.html", title=APP_TITLE, error="Please enter valid numeric values.")

    with db() as conn:
        conn.execute("""
        INSERT INTO batches (
            batch_no, out_date,
            input_birds, cull_birds,
            feed_bb_kg, feed_bf_kg,
            total_weight_kg,
            total_cost_lkr, price_per_kg_lkr,
            profit_per_bird_lkr,
            created_by, created_at, updated_by, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            batch_no, out_date,
            input_birds, cull_birds,
            feed_bb_kg, feed_bf_kg,
            total_weight_kg,
            float(total_cost_lkr) if total_cost_lkr not in (None, "") else None,
            float(price_per_kg_lkr) if price_per_kg_lkr not in (None, "") else None,
            float(profit_per_bird_lkr) if profit_per_bird_lkr not in (None, "") else None,
            actor["username"] if actor else "UNKNOWN",
            now_iso(),
            actor["username"] if actor else "UNKNOWN",
            now_iso(),
        ))

    return redirect(url_for("dashboard"))


@app.route("/edit/<int:batch_id>", methods=["GET", "POST"])
@master_required
def edit_batch(batch_id: int):
    actor = get_current_user()
    r = get_batch(batch_id)
    if request.method == "GET":
        return render_template("edit_batch.html", title=APP_TITLE, r=r)

    out_date = request.form.get("out_date", "").strip()
    batch_no = request.form.get("batch_no", "").strip()

    def to_int(name: str) -> int:
        v = request.form.get(name, "0") or "0"
        return int(float(v))

    def to_float(name: str) -> float:
        v = request.form.get(name, "0") or "0"
        return float(v)

    input_birds = to_int("input_birds")
    cull_birds = to_int("cull_birds")
    feed_bb_kg = to_float("feed_bb_kg")
    feed_bf_kg = to_float("feed_bf_kg")
    total_weight_kg = to_float("total_weight_kg")

    profit_per_bird_lkr = request.form.get("profit_per_bird_lkr") or None
    total_cost_lkr = request.form.get("total_cost_lkr") or None
    price_per_kg_lkr = request.form.get("price_per_kg_lkr") or None

    if not out_date or not batch_no:
        return render_template("edit_batch.html", title=APP_TITLE, r=r, error="Date and Batch Number are required.")

    if input_birds <= 0 or cull_birds <= 0 or total_weight_kg <= 0 or (feed_bb_kg + feed_bf_kg) <= 0:
        return render_template("edit_batch.html", title=APP_TITLE, r=r, error="Please enter valid numeric values.")

    with db() as conn:
        conn.execute("""
        UPDATE batches SET
            batch_no=?,
            out_date=?,
            input_birds=?,
            cull_birds=?,
            feed_bb_kg=?,
            feed_bf_kg=?,
            total_weight_kg=?,
            total_cost_lkr=?,
            price_per_kg_lkr=?,
            profit_per_bird_lkr=?,
            updated_by=?,
            updated_at=?
        WHERE id=?;
        """, (
            batch_no,
            out_date,
            input_birds,
            cull_birds,
            feed_bb_kg,
            feed_bf_kg,
            total_weight_kg,
            float(total_cost_lkr) if total_cost_lkr not in (None, "") else None,
            float(price_per_kg_lkr) if price_per_kg_lkr not in (None, "") else None,
            float(profit_per_bird_lkr) if profit_per_bird_lkr not in (None, "") else None,
            actor["username"] if actor else "UNKNOWN",
            now_iso(),
            batch_id,
        ))

    return redirect(url_for("view_batch", batch_id=batch_id))


@app.post("/delete/<int:batch_id>")
@master_required
def delete_batch(batch_id: int):
    with db() as conn:
        conn.execute("DELETE FROM batches WHERE id=?;", (batch_id,))
    return redirect(url_for("dashboard"))


@app.route("/batch/<int:batch_id>")
@login_required
def view_batch(batch_id: int):
    r = get_batch(batch_id)
    k = compute_kpi(r)
    prev = get_previous_batch(r)
    days_since_prev = None
    if prev:
        curr_d = parse_date(r["out_date"])
        prev_d = parse_date(prev["out_date"])
        if curr_d and prev_d:
            days_since_prev = (curr_d - prev_d).days
    return render_template("batch.html", title=APP_TITLE, r=r, k=k, prev=prev, days_since_prev=days_since_prev)


@app.route("/pdf/<int:batch_id>")
@login_required
def batch_pdf(batch_id: int):
    r = get_batch(batch_id)
    k = compute_kpi(r)
    prev = get_previous_batch(r)
    days_since_prev = None
    if prev:
        curr_d = parse_date(r["out_date"])
        prev_d = parse_date(prev["out_date"])
        if curr_d and prev_d:
            days_since_prev = (curr_d - prev_d).days

    start = int(r["input_birds"])
    cull = int(r["cull_birds"])
    bb = float(r["feed_bb_kg"])
    bf = float(r["feed_bf_kg"])
    tw = float(r["total_weight_kg"])
    profit_per_bird = r["profit_per_bird_lkr"]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def header(text: str, y: float):
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, y, text)

    def section(text: str, y: float):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, text)
        c.setLineWidth(1)
        c.setStrokeGray(0.85)
        c.line(40, y - 6, width - 40, y - 6)
        c.setStrokeGray(0)

    def line(label: str, value: str, y: float):
        c.setFont("Helvetica", 11)
        c.drawString(50, y, label)
        c.setFont("Helvetica-Bold", 11)
        c.drawRightString(width - 50, y, value)

    y = height - 60
    header("BROILER BATCH REPORT (OFFLINE)", y)

    y -= 35
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Batch: {r['batch_no']}   |   Date: {r['out_date']}   |   Age: {AGE_DAYS} days")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Added by: {r['created_by'] or '-'}   |   Updated by: {r['updated_by'] or '-'}")
    if prev and days_since_prev is not None:
        y -= 16
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Previous batch: {prev['batch_no']}   |   Date: {prev['out_date']}   |   Gap: {days_since_prev} days")

    y -= 30
    section("KPI Snapshot", y)
    y -= 22
    line("Avg Weight / bird (kg)", f"{k.avg_weight_per_bird_kg:.3f}", y); y -= 18
    line("FCR", f"{k.fcr:.2f}", y); y -= 18
    line("Mortality %", f"{k.mortality_pct*100:.2f}%", y); y -= 18
    line("PI", f"{k.pi:.0f}", y); y -= 18
    line("Score", f"{k.score:.1f}/100", y)

    y -= 26
    section("Birds Summary", y)
    y -= 22
    line("Input birds (start)", f"{start:,}", y); y -= 18
    line("Cull birds (final out)", f"{cull:,}", y); y -= 18
    line("Mortality birds (auto)", f"{k.mortality_birds:,}", y); y -= 18
    line("Livability %", f"{k.livability_pct:.2f}%", y)

    y -= 26
    section("Feed Summary", y)
    y -= 22
    line("Feed BB (kg)", f"{bb:,.0f}", y); y -= 18
    line("Feed BF (kg)", f"{bf:,.0f}", y); y -= 18
    line("Total feed (kg)", f"{k.total_feed_kg:,.0f}", y); y -= 18
    line("Avg feed / bird (kg)", f"{k.avg_feed_per_bird_kg:.3f}", y)

    y -= 26
    section("Weight + Finance", y)
    y -= 22
    line("Total weight (kg)", f"{tw:,.0f}", y); y -= 18
    line("Profit per bird (LKR) [manual]", f"{float(profit_per_bird):,.2f}" if profit_per_bird is not None else "-", y)

    y -= 28
    c.setFont("Helvetica-Oblique", 9)
    c.setFillGray(0.45)
    c.drawCentredString(width / 2, y, "Generated offline • Keep folder backups.")
    y -= 14
    c.drawCentredString(width / 2, y, "© 2026 E-Broiler Dashboard | Developed by Hasintha Arunalu | RTX Technologies. All rights reserved.")
    c.setFillGray(0)

    c.showPage()
    c.save()

    buf.seek(0)
    fname = f"broiler_batch_{r['batch_no']}_{r['out_date']}.pdf"
    return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=fname)


# -----------------------------
# One-click EXE behavior
# -----------------------------
def open_browser():
    port = int(os.environ.get("PORT", "5000"))
    webbrowser.open(f"http://127.0.0.1:{port}")


@app.route("/login", methods=["GET", "POST"])
def login():
    user = get_current_user()
    if user:
        return redirect(url_for("dashboard"))

    if request.method == "GET":
        return render_template(
            "login.html",
            title=f"Login - {APP_TITLE}",
            error=session.pop("login_error", None),
        )

    username = request.form.get("username", "").strip().upper()
    password = request.form.get("password", "")
    info = get_db_user(username)

    if info and password == info["password"]:
        if is_license_expired(info):
            return render_template(
                "login.html",
                title=f"Login - {APP_TITLE}",
                error="License expired. Please contact master.",
            )
        session["username"] = username
        session["role"] = info["role"]
        return redirect(url_for("dashboard"))

    return render_template(
        "login.html",
        title=f"Login - {APP_TITLE}",
        error="Invalid username or password.",
    )


@app.route("/users", methods=["GET", "POST"])
@master_required
def manage_users():
    actor = get_current_user()
    error = None
    success = None

    if request.method == "POST":
        action = request.form.get("action", "").strip()
        username = request.form.get("username", "").strip().upper()

        if not username:
            error = "Username is required."
        else:
            if action == "create_user":
                password = request.form.get("password", "")
                day_raw = request.form.get("license_days", "").strip()
                if not password:
                    error = "Password is required."
                elif not day_raw:
                    error = "License days is required."
                else:
                    try:
                        license_days = float(day_raw)
                    except ValueError:
                        license_days = 0
                    if license_days <= 0:
                        error = "License days must be greater than 0."
                    elif get_db_user(username):
                        error = "Username already exists."
                    else:
                        license_seconds = int(license_days * SECONDS_PER_DAY)
                        exp = now_ts() + license_seconds
                        with db() as conn:
                            conn.execute(
                                """
                                INSERT INTO users (
                                    username, password, role, role_label, license_expires_at,
                                    created_by, created_at, updated_by, updated_at
                                ) VALUES (?, ?, 'user', 'User Login', ?, ?, ?, ?, ?);
                                """,
                                (
                                    username,
                                    password,
                                    exp,
                                    actor["username"] if actor else "UNKNOWN",
                                    now_iso(),
                                    actor["username"] if actor else "UNKNOWN",
                                    now_iso(),
                                ),
                            )
                        success = f"User {username} created."

            elif action == "extend_license":
                day_raw = request.form.get("license_days", "").strip()
                if not day_raw:
                    error = "License days is required."
                else:
                    try:
                        add_days = float(day_raw)
                    except ValueError:
                        add_days = 0
                    target = get_db_user(username)
                    if add_days <= 0:
                        error = "License days must be greater than 0."
                    elif not target:
                        error = "User not found."
                    elif target["role"] == "master":
                        error = "Master accounts do not use license expiry."
                    else:
                        add_seconds = int(add_days * SECONDS_PER_DAY)
                        current_exp = int(target["license_expires_at"]) if target["license_expires_at"] is not None else now_ts()
                        base = max(current_exp, now_ts())
                        new_exp = base + add_seconds
                        with db() as conn:
                            conn.execute(
                                """
                                UPDATE users
                                SET license_expires_at=?, updated_by=?, updated_at=?
                                WHERE username=?;
                                """,
                                (
                                    new_exp,
                                    actor["username"] if actor else "UNKNOWN",
                                    now_iso(),
                                    username,
                                ),
                            )
                        success = f"License extended for {username}."

            elif action == "update_password":
                new_password = request.form.get("password", "")
                target = get_db_user(username)
                if not new_password:
                    error = "New password is required."
                elif not target:
                    error = "User not found."
                else:
                    with db() as conn:
                        conn.execute(
                            """
                            UPDATE users
                            SET password=?, updated_by=?, updated_at=?
                            WHERE username=?;
                            """,
                            (
                                new_password,
                                actor["username"] if actor else "UNKNOWN",
                                now_iso(),
                                username,
                            ),
                        )
                    success = f"Password updated for {username}."

            else:
                error = "Unknown action."

    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY CASE WHEN role='master' THEN 0 ELSE 1 END, username ASC;"
        ).fetchall()
    users = []
    for u in rows:
        exp = u["license_expires_at"]
        rem = (int(exp) - now_ts()) if exp is not None else None
        users.append(
            {
                "username": u["username"],
                "role": u["role"],
                "role_label": u["role_label"],
                "license_expires_at": exp,
                "license_expires_at_text": format_ts(exp) if exp is not None else None,
                "license_remaining_days": seconds_to_days(rem),
                "created_by": u["created_by"],
                "created_at": u["created_at"],
                "updated_by": u["updated_by"],
                "updated_at": u["updated_at"],
            }
        )

    return render_template(
        "users.html",
        title=f"Users - {APP_TITLE}",
        users=users,
        error=error,
        success=success,
    )


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.post("/shutdown")
def shutdown():
    # Works for the built-in Werkzeug server used by this app.
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        abort(500)
    func()
    return "Shutting down..."


# Ensure DB exists for local and deployed runs.
init_db()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))

    # Auto-open browser only for local runs.
    if os.environ.get("OPEN_BROWSER", "1") == "1" and host in {"127.0.0.1", "localhost"}:
        threading.Timer(1.2, open_browser).start()

    app.run(host=host, port=port, debug=False)
