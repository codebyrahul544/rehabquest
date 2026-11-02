"""Progress report: matplotlib charts (PNG) + an HTML page with therapist-style comments.

Charts (one measure per panel, never a dual axis):
  1. score over time for each game      4. finger success rate (weakest finger)
  2. left vs right arm reach            5. shoulder abduction range (L / R)
  3. facial symmetry trend              6. finger open-close range
"""
from __future__ import annotations

import base64
import html
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order, one colour per game - never cycled).
GAME_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100", 5: "#e87ba4"}
LEFT_COLOR, RIGHT_COLOR = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GAME_NAMES = {1: "Bubble Garden", 2: "Lantern Lighter", 3: "Smile Sunshine", 4: "Mirror Dancer", 5: "Spell Caster"}


def load_sessions(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / "sessions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["extra"] = df["extra"].apply(lambda s: json.loads(s) if isinstance(s, str) and s.startswith("{") else {})
    df["passed"] = df["passed"].astype(int).astype(bool)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["session"] = df.groupby("game_id").cumcount() + 1
    return df


def _style(ax, title, ylabel, xlabel="Session"):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2, fontsize=11, transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _line(ax, x, y, color, label=None, direct=True):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=label)
    if direct and label and len(x):
        ax.annotate(label, (x.iloc[-1], y.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)


def _xi(ax):
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def build_figure(df: pd.DataFrame, png_path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    (a1, a2), (a3, a4), (a5, a6) = axes

    # 1. score per game
    _style(a1, "Score over time, per game", "Score")
    if df.empty:
        _empty(a1, "No sessions yet - play a game first!")
    else:
        for gid in sorted(df["game_id"].unique()):
            d = df[df["game_id"] == gid]
            _line(a1, d["session"], d["score"], GAME_COLORS.get(int(gid), INK2), GAME_NAMES.get(int(gid)), False)
        a1.legend(frameon=False, fontsize=9, loc="best")
        _xi(a1)

    # 2. left vs right reach (Game 1)
    _style(a2, "Arm reach: left vs right (Bubble Garden)", "Average reach height (% of screen)")
    g1 = df[df["game_id"] == 1] if not df.empty else df
    if g1.empty:
        _empty(a2, "Play Bubble Garden to see left/right reach")
    else:
        rl = g1["extra"].apply(lambda e: e.get("reach_left", 0) or None)
        rr = g1["extra"].apply(lambda e: e.get("reach_right", 0) or None)
        x = g1["session"]
        w = 0.38
        a2.bar(x - w / 2 - 0.01, rl.fillna(0), w, color=LEFT_COLOR, label="Left arm")
        a2.bar(x + w / 2 + 0.01, rr.fillna(0), w, color=RIGHT_COLOR, label="Right arm")
        a2.legend(frameon=False, fontsize=9)
        a2.set_ylim(0, 100)
        _xi(a2)

    # 3. facial symmetry
    _style(a3, "Facial symmetry (Smile Sunshine)", "Mean symmetry (%)")
    g3 = df[df["game_id"] == 3] if not df.empty else df
    if g3.empty:
        _empty(a3, "Play Smile Sunshine to track facial symmetry")
    else:
        _line(a3, g3["session"], g3["rom_value"], GAME_COLORS[3])
        a3.axhline(55, color=INK2, linestyle="--", linewidth=1)
        a3.text(g3["session"].iloc[0], 56.5, "pass threshold 55%", fontsize=8, color=INK2)
        a3.set_ylim(0, 105)
        _xi(a3)

    # 4. finger weakness (Game 5)
    _style(a4, "Finger success rate (Spell Caster)", "Success (%)", "Finger")
    g5 = df[df["game_id"] == 5] if not df.empty else df
    if g5.empty:
        _empty(a4, "Play Spell Caster to find your weakest finger")
    else:
        tot = {f: [0, 0] for f in ("thumb", "index", "middle", "ring", "pinky")}
        for e in g5["extra"]:
            for f, s in (e.get("fingers") or {}).items():
                if f in tot:
                    tot[f][0] += s.get("ok", 0)
                    tot[f][1] += s.get("tries", 0)
        names = [f for f in tot if tot[f][1]]
        rates = [100 * tot[f][0] / tot[f][1] for f in names]
        bars = a4.bar(names, rates, color=GAME_COLORS[5], width=0.6)
        if rates and max(rates) - min(rates) >= 0.5:     # only flag a weakest finger when there is one
            k = rates.index(min(rates))
            bars[k].set_hatch("//")
            bars[k].set_edgecolor(SURFACE)
            a4.annotate("weakest", (k, rates[k]), xytext=(0, 6), textcoords="offset points", ha="center",
                        fontsize=9, color=INK)
        a4.set_ylim(0, 110)

    # 5. shoulder range (Game 4)
    _style(a5, "Shoulder lift range (Mirror Dancer)", "Max shoulder abduction (deg)")
    g4 = df[df["game_id"] == 4] if not df.empty else df
    if g4.empty:
        _empty(a5, "Play Mirror Dancer to track shoulder range")
    else:
        _line(a5, g4["session"], g4["extra"].apply(lambda e: e.get("abd_left_max", 0)), LEFT_COLOR, "Left", False)
        _line(a5, g4["session"], g4["extra"].apply(lambda e: e.get("abd_right_max", 0)), RIGHT_COLOR, "Right", False)
        a5.legend(frameon=False, fontsize=9, loc="lower right")
        a5.set_ylim(0, 190)
        _xi(a5)

    # 6. finger range (Game 2)
    _style(a6, "Finger open-close range (Lantern Lighter)", "Range (open - fist, x hand size)")
    g2 = df[df["game_id"] == 2] if not df.empty else df
    if g2.empty:
        _empty(a6, "Play Lantern Lighter to track finger range")
    else:
        _line(a6, g2["session"], g2["rom_value"], GAME_COLORS[2])
        a6.set_ylim(0, max(1.5, float(g2["rom_value"].max()) * 1.2))
        _xi(a6)

    fig.suptitle("RehabQuest - Progress Report", x=0.02, ha="left", fontsize=20, fontweight="bold", color=INK)
    fig.text(0.02, 0.955, time.strftime("Generated %d %b %Y %H:%M") + "   -   practice aid only, "
             "share with your physiotherapist", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, facecolor=SURFACE)
    plt.close(fig)


def _change(series: pd.Series):
    """(first-third mean, last-third mean, % change) or None if too little data."""
    s = series.dropna()
    if len(s) < 2:
        return None
    k = max(1, len(s) // 3)
    a, b = float(s.iloc[:k].mean()), float(s.iloc[-k:].mean())
    if a == 0:
        return None
    return a, b, 100 * (b - a) / abs(a)


def _window_label(df):
    span = (df["timestamp"].max() - df["timestamp"].min()).days if not df.empty else 0
    return "this week" if span <= 7 else "since you started"


def insights(df: pd.DataFrame) -> list:
    """Plain-language, therapist-style comments."""
    out = []
    if df.empty:
        return ["No sessions recorded yet. Play Bubble Garden to begin your journey!"]
    when = _window_label(df)
    out.append(f"You have completed {len(df)} sessions ({int(df['passed'].sum())} passed) "
               f"across {df['game_id'].nunique()} games, for {df['duration_s'].sum() / 60:.0f} minutes of exercise.")
    g1 = df[df["game_id"] == 1]
    if not g1.empty:
        c = _change(g1["rom_value"])
        if c:
            verb = "improved" if c[2] >= 0 else "dipped"
            out.append(f"Your shoulder reach {verb} {abs(c[2]):.0f}% {when} "
                       f"(from {c[0]:.0f}% to {c[1]:.0f}% of screen height).")
        left = sum(e.get("pops_left", 0) for e in g1["extra"])
        right = sum(e.get("pops_right", 0) for e in g1["extra"])
        if left + right >= 10:
            share = 100 * left / (left + right)
            if 40 <= share <= 60:
                out.append(f"Nice balance: your left hand did {share:.0f}% of the bubble popping.")
            else:
                weak = "left" if share < 40 else "right"
                out.append(f"Your {weak} hand did only {min(share, 100 - share):.0f}% of the popping - "
                           f"try the '{weak} hand' setting to give it extra practice.")
    g2 = df[df["game_id"] == 2]
    if not g2.empty:
        c = _change(g2["rom_value"])
        if c:
            trend = "stayed steady" if abs(c[2]) < 2 else f"changed {c[2]:+.0f}%"
            out.append(f"Finger opening range {trend} {when} ({c[0]:.2f} -> {c[1]:.2f} hand-lengths).")
        best = max((e.get("best_open_hold_s", 0) for e in g2["extra"]), default=0)
        out.append(f"Your longest open-hand hold is {best:.1f} seconds - good grip endurance.")
    g3 = df[df["game_id"] == 3]
    if not g3.empty:
        c = _change(g3["rom_value"])
        last = float(g3["rom_value"].iloc[-1])
        out.append(f"Your latest facial symmetry is {last:.0f}%" +
                   (f", a {c[2]:+.0f}% change {when}." if c else "."))
        sym = {}
        for e in g3["extra"]:
            for k, v in (e.get("symmetry") or {}).items():
                if v is not None:
                    sym.setdefault(k, []).append(v)
        if sym:
            worst = min(sym, key=lambda k: sum(sym[k]) / len(sym[k]))
            names = {"smile": "the big smile", "brows": "eyebrow raises", "eyes": "eye closing",
                     "pucker": "the kiss/pucker", "jaw": "mouth opening"}
            out.append(f"Least even movement: {names.get(worst, worst)}. Practise it slowly in a mirror.")
    g4 = df[df["game_id"] == 4]
    if not g4.empty:
        e = g4["extra"].iloc[-1]
        l, r = e.get("abd_left_max", 0), e.get("abd_right_max", 0)
        out.append(f"Latest shoulder lift: left {l:.0f} deg, right {r:.0f} deg"
                   + (f" - your {'left' if l < r else 'right'} side is {abs(l - r):.0f} deg behind." if abs(l - r) > 15
                      else " - nicely balanced."))
        post = [x.get("posture", 0) for x in g4["extra"]]
        out.append(f"Average posture score {100 * sum(post) / len(post):.0f}% (level shoulders, head centred).")
    g5 = df[df["game_id"] == 5]
    if not g5.empty:
        weak = g5["extra"].iloc[-1].get("weakest_finger")
        if weak and weak != "-":
            out.append(f"Your {weak} finger is currently the weakest - spend extra time on thumb-to-{weak} taps.")
    return out


def generate_report(data_dir: Path, out_dir: Path, demo: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_sessions(data_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    png = out_dir / f"report_{stamp}.png"
    html_path = out_dir / f"report_{stamp}.html"
    build_figure(df, png)
    comments = insights(df)
    b64 = base64.b64encode(png.read_bytes()).decode()
    rows = ""
    if not df.empty:
        for _, r in df.iloc[::-1].head(40).iterrows():
            rows += ("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                r["timestamp"].strftime("%Y-%m-%d %H:%M") if pd.notna(r["timestamp"]) else "",
                r["game_name"], r["difficulty"], int(r["score"]), f"{100 * r['accuracy']:.0f}%",
                f"{r['rom_value']:.1f}", r["rom_label"], f"{r['duration_s']:.0f}s",
                "Passed" if r["passed"] else "Not yet", "*" * int(r["stars"]))) + "</tr>")
    summary_cards = ""
    if not df.empty:
        for gid, name in GAME_NAMES.items():
            d = df[df["game_id"] == gid]
            best = int(d["score"].max()) if not d.empty else 0
            passed = "Passed" if (not d.empty and d["passed"].any()) else ("Not yet" if not d.empty else "Not played")
            summary_cards += (f'<div class="card"><div class="dot" style="background:{GAME_COLORS[gid]}"></div>'
                              f'<b>{gid}. {name}</b><span>{passed}</span><span>best {best}</span>'
                              f'<span>{len(d)} sessions</span></div>')
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RehabQuest Report</title>
<style>
:root{{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{margin:0 0 4px}} .sub{{color:var(--ink2);margin:0 0 20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:16px 0}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:2px}}
.card span{{color:var(--ink2);font-size:14px}} .dot{{width:12px;height:12px;border-radius:50%}}
.note{{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;padding:14px 18px}}
.note li{{margin:6px 0}} img{{width:100%;border-radius:12px;border:1px solid var(--line);margin:16px 0}}
.tablewrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--ink2)}}
.disc{{color:var(--ink2);font-size:14px;margin-top:24px}}
</style></head><body><main>
<h1>RehabQuest progress report</h1>
<p class="sub">Generated {time.strftime('%d %B %Y, %H:%M')}{' - DEMO DATA (virtual patient)' if demo else ''}</p>
<div class="cards">{summary_cards}</div>
<h2>What your therapist would say</h2>
<div class="note"><ul>{''.join(f'<li>{html.escape(c)}</li>' for c in comments)}</ul></div>
<h2>Charts</h2>
<img alt="Progress charts: score per game, left vs right reach, facial symmetry, finger success, shoulder range, finger range" src="data:image/png;base64,{b64}">
<h2>Session log</h2>
<div class="tablewrap"><table><thead><tr><th>Date</th><th>Game</th><th>Level</th><th>Score</th><th>Accuracy</th>
<th>Range</th><th>Range measure</th><th>Time</th><th>Result</th><th>Stars</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="disc">RehabQuest is a practice aid. It does not diagnose or replace a physiotherapist, occupational therapist or doctor.
Measurements come from a webcam and are approximate. Stop if anything hurts.</p>
</main></body></html>"""
    html_path.write_text(doc, encoding="utf-8")
    return html_path, png
