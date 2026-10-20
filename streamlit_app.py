import math
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st

st.set_page_config(
    page_title="เรียนรู้กราฟฟังก์ชันลอการิทึม",
    page_icon="📉",
    layout="wide",
)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

# ชื่อ, ค่าต่ำสุด, ค่าสูงสุด, step, คำอธิบาย
PARAMS = {
    "alpha": ("α", -5.0, 5.0, 0.1, "ยืด/หด และสะท้อนในแนวตั้ง"),
    "a": ("a", 0.1, 10.0, 0.1, "ฐาน (a > 0, a ≠ 1)"),
    "c": ("c", -5.0, 5.0, 0.1, "เลื่อนแนวตั้ง"),
}

EFFECTS = {
    "alpha": "|α| ยืด (|α| > 1) หรือหด (|α| < 1) กราฟในแนวตั้งโดยวัดจากแกน X และถ้า α < 0 กราฟจะสะท้อนกับแกน X "
             "จุด (1, c) อยู่ที่เดิมเสมอ ส่วนจุด (a, α + c) ขยับขึ้นลงตาม α",
    "a": "ถ้า a > 1 ค่า a ยิ่งมาก กราฟยิ่งแบน (โตช้าลง) ส่วนถ้า 0 < a < 1 กราฟจะกลายเป็นฟังก์ชันลด "
         "ทุกกราฟยังผ่านจุด (1, c) เหมือนกัน",
    "c": "c เลื่อนกราฟขึ้นหรือลงทั้งเส้น จุด (1, 0) จะย้ายไปอยู่ที่ (1, c) "
         "ส่วนเส้นกำกับแนวตั้ง x = 0 ไม่เปลี่ยน",
}


# ===================== ส่วนคณิตศาสตร์ =====================
def f(x, alpha, a, c):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = alpha * np.log(x) / np.log(a) + c
    return np.where(np.isfinite(y) & (x > 0), y, np.nan)


def f_inv(x, alpha, a, c):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y = np.power(a, (np.asarray(x, dtype=float) - c) / alpha)
    return np.where(np.isfinite(y), y, np.nan)


def fmt(v, nd=3):
    v = round(float(v), nd)
    if v == 0:
        v = 0.0
    return f"{v:g}"


def is_e(a):
    return math.isclose(a, math.e, rel_tol=1e-9)


def base_txt(a):
    return "e" if is_e(a) else fmt(a)


def log_term(a):
    return r"\ln x" if is_e(a) else rf"\log_{{{fmt(a)}}} x"


def latex_eq(p):
    alpha, a, c = p["alpha"], p["a"], p["c"]
    coef = "" if alpha == 1 else ("-" if alpha == -1 else f"{fmt(alpha)}\\,")
    const = "" if c == 0 else (f" + {fmt(c)}" if c > 0 else f" - {fmt(-c)}")
    return f"y = {coef}{log_term(a)}{const}"


def inverse_parts(p):
    alpha, c = p["alpha"], p["c"]
    num = "x" if c == 0 else (f"x - {fmt(c)}" if c > 0 else f"x + {fmt(-c)}")
    return num, alpha


def inverse_latex(p):
    num, alpha = inverse_parts(p)
    expo = num if alpha == 1 else rf"\frac{{{num}}}{{{fmt(alpha)}}}"
    b = "e" if is_e(p["a"]) else f"({fmt(p['a'])})"
    return f"y = {b}^{{{expo}}}"


def inverse_txt(p):
    num, alpha = inverse_parts(p)
    expo = num if alpha == 1 else f"({num})/{fmt(alpha)}"
    return f"y = {base_txt(p['a'])}^({expo})"


def analyze(p):
    alpha, a, c = p["alpha"], p["a"], p["c"]
    rows = {}
    if math.isclose(a, 1):
        rows["สถานะ"] = "ไม่นิยาม: ลอการิทึมฐาน 1 ไม่มีความหมาย (1 ยกกำลังอะไรก็ได้ 1)"
        return rows
    rows["โดเมน"] = "(0, ∞)"
    if math.isclose(alpha, 0, abs_tol=1e-12):
        rows["ลักษณะกราฟ"] = f"เส้นตรงแนวนอน y = {fmt(c)} เฉพาะช่วง x > 0"
        return rows
    k = alpha / math.log(a)
    rows["แนวโน้มของกราฟ"] = "ฟังก์ชันเพิ่ม ↗" if k > 0 else "ฟังก์ชันลด ↘"
    rows["k = α / ln a"] = f"{fmt(k, 4)} (เขียนใหม่ได้เป็น y = {fmt(k, 4)}·ln x + {fmt(c)})"
    rows["เรนจ์"] = "ℝ (จำนวนจริงทั้งหมด)"
    rows["เส้นกำกับแนวตั้ง"] = f"x = 0 (y → {'−∞' if k > 0 else '+∞'} เมื่อ x → 0⁺)"
    rows["จุดอ้างอิง (1, c)"] = f"(1, {fmt(c)})"
    rows["จุดอ้างอิง (a, α + c)"] = f"({fmt(a)}, {fmt(alpha + c)})"
    try:
        rows["จุดตัดแกน X"] = f"({math.exp(-c / k):.4g}, 0)"
    except OverflowError:
        rows["จุดตัดแกน X"] = "มีอยู่ แต่ค่ามากเกินกว่าจะแสดงได้"
    rows["จุดตัดแกน Y"] = "ไม่มี (x = 0 อยู่นอกโดเมน)"
    rows["ความเว้า"] = "เว้าลง ∩" if k > 0 else "เว้าขึ้น ∪"
    rows["เมื่อ x คูณด้วย 2"] = f"y เปลี่ยนไป {fmt(k * math.log(2), 4)} เท่ากันทุกจุด"
    rows["ฟังก์ชันผกผัน"] = inverse_txt(p)
    return rows


def transformation_steps(p):
    alpha, a, c = p["alpha"], p["a"], p["c"]
    if math.isclose(a, 1):
        return ["ลอการิทึมฐาน 1 ไม่นิยาม จึงวาดกราฟไม่ได้"]
    trend = "เป็นฟังก์ชันเพิ่มเพราะ a > 1" if a > 1 else "เป็นฟังก์ชันลดเพราะ 0 < a < 1"
    steps = [f"เริ่มจากกราฟต้นแบบ $y = {log_term(a)}$ ซึ่งผ่านจุด (1, 0) และ ({fmt(a)}, 1) และ{trend}"]
    if alpha != 1:
        if alpha == 0:
            steps.append("คูณด้วย α = 0 : กราฟถูกบีบลงเป็นเส้น y = 0")
        else:
            parts = []
            if abs(alpha) != 1:
                word = "ยืด" if abs(alpha) > 1 else "หด"
                parts.append(f"{word}ในแนวตั้งด้วยตัวคูณ {fmt(abs(alpha))}")
            if alpha < 0:
                parts.append(f"สะท้อนกับแกน X (ได้ผลเหมือนเปลี่ยนฐานเป็น 1/a = {fmt(1 / a)})")
            steps.append(f"คูณด้วย α = {fmt(alpha)} : " + ", ".join(parts)
                         + f" → จุด ({fmt(a)}, 1) ย้ายไปที่ ({fmt(a)}, {fmt(alpha)})")
    if c != 0:
        steps.append(f"เลื่อน{'ขึ้น' if c > 0 else 'ลง'} {fmt(abs(c))} หน่วย → จุด (1, 0) ย้ายไปที่ (1, {fmt(c)})")
    steps.append("เส้นกำกับแนวตั้ง x = 0 ไม่เปลี่ยน เพราะไม่มีการเลื่อนในแนวนอน")
    return steps


def intersection(pf, pg):
    if math.isclose(pf["a"], 1) or math.isclose(pg["a"], 1):
        return None, "หาจุดตัดไม่ได้ เพราะมีกราฟที่ฐานเท่ากับ 1"
    kf = pf["alpha"] / math.log(pf["a"])
    kg = pg["alpha"] / math.log(pg["a"])
    dc = pg["c"] - pf["c"]
    if math.isclose(kf, kg, abs_tol=1e-12):
        if math.isclose(dc, 0, abs_tol=1e-12):
            return None, "สองกราฟทับกันทุกจุด (เป็นกราฟเดียวกัน)"
        return None, "สองกราฟไม่ตัดกัน เพราะ k เท่ากันแต่ c ต่างกัน (ห่างกันในแนวตั้งเท่ากันทุกจุด)"
    try:
        x0 = math.exp(dc / (kf - kg))
    except OverflowError:
        return None, "สองกราฟตัดกันที่ค่า x มากเกินกว่าจะแสดงได้"
    y0 = float(f(np.array([x0]), **pf)[0])
    return (x0, y0), f"สองกราฟตัดกันจุดเดียวที่ ({x0:.4g}, {y0:.4g})"


# ===================== ส่วนกราฟ =====================
def make_fig(curves, x_range, y_range, show_asym=True, show_points=True,
             log_x=False, height=550, extra_points=None):
    fig = go.Figure()
    lo, hi = x_range
    if log_x:
        lo = max(lo, 0.01)
        hi = max(hi, lo * 10)
    start = max(lo, 1e-6)
    xs = (np.unique(np.concatenate([np.geomspace(start, hi, 1500), np.linspace(start, hi, 1000)]))
          if hi > start else np.array([]))
    x_full = np.linspace(lo, hi, 1000)

    for i, cv in enumerate(curves):
        color = cv.get("color", COLORS[i % len(COLORS)])
        line = dict(color=color, width=cv.get("width", 3), dash=cv.get("dash", "solid"))
        if "func" in cv:
            fig.add_trace(go.Scatter(x=x_full, y=cv["func"](x_full), mode="lines",
                                     name=cv["name"], line=line))
            continue
        p = cv["params"]
        fig.add_trace(go.Scatter(x=xs, y=f(xs, **p), mode="lines", name=cv["name"], line=line))
        if cv.get("reference") or not show_points or math.isclose(p["a"], 1):
            continue
        fig.add_trace(go.Scatter(
            x=[1, p["a"]], y=[p["c"], p["alpha"] + p["c"]], mode="markers",
            marker=dict(color=color, size=11, symbol=["circle", "diamond"]),
            showlegend=False, hovertemplate="(%{x:.3g}, %{y:.3g})<extra></extra>",
        ))

    if show_asym and not log_x and lo <= 0 < hi:
        fig.add_vline(x=0, line_dash="dot", line_color="crimson", line_width=2,
                      annotation_text="เส้นกำกับ x = 0", annotation_position="top right")
    for px, py, label in extra_points or []:
        fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers+text", text=[label],
                                 textposition="top center", showlegend=False,
                                 marker=dict(size=13, color="black", symbol="x")))

    xaxis = dict(title="x" + (" (สเกล log)" if log_x else ""), zeroline=True,
                 zerolinewidth=2, zerolinecolor="#444")
    if log_x:
        xaxis.update(type="log", range=[math.log10(lo), math.log10(hi)])
    else:
        xaxis["range"] = [lo, hi]
    fig.update_layout(
        height=height, xaxis=xaxis,
        yaxis=dict(range=list(y_range), zeroline=True, zerolinewidth=2, zerolinecolor="#444", title="y"),
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
    )
    return fig


def param_controls(prefix, d, title=None):
    if title:
        st.markdown(f"#### {title}")
    p = {}
    for key, (sym, lo, hi, step, desc) in PARAMS.items():
        if key == "a" and st.checkbox("ใช้ฐาน e (ln x)", key=f"{prefix}_use_e"):
            p["a"] = math.e
            continue
        p[key] = st.slider(f"{sym} — {desc}", lo, hi, float(d[key]), step, key=f"{prefix}_{key}")
    if math.isclose(p["a"], 1.0):
        st.error("a = 1 ใช้ไม่ได้ เพราะลอการิทึมฐาน 1 ไม่นิยาม")
    if p["alpha"] == 0:
        st.warning("α = 0 ทำให้ได้ y = c (เฉพาะ x > 0)")
    return p


# ===================== Sidebar =====================
st.sidebar.header("⚙️ ตั้งค่าการแสดงผล")
x_range = st.sidebar.slider("ช่วงแกน x", -5.0, 50.0, (-1.0, 10.0), 0.5)
y_range = st.sidebar.slider("ช่วงแกน y", -30.0, 30.0, (-6.0, 6.0), 0.5)
show_asym = st.sidebar.checkbox("แสดงเส้นกำกับ x = 0", True)
show_points = st.sidebar.checkbox("แสดงจุดอ้างอิง (1, c) และ (a, α + c)", True)
log_x = st.sidebar.checkbox("ใช้สเกล log บนแกน x", False,
                            help="กราฟลอการิทึมทุกเส้นจะกลายเป็นเส้นตรง")
st.sidebar.markdown("---")
st.sidebar.latex(r"y = \alpha \log_a x + c")
st.sidebar.caption("● = (1, c)   ◆ = (a, α + c)   ┄ = เส้นกำกับแนวตั้ง")

st.title("📉 เรียนรู้กราฟฟังก์ชันลอการิทึม")
st.latex(r"y = \alpha \log_a x + c")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔍 สำรวจกราฟ", "⚖️ เปรียบเทียบ 2 กราฟ", "🌈 ตระกูลกราฟ", "🎯 แบบฝึกหัด", "📚 สรุปทฤษฎี"]
)

# ===================== Tab 1: สำรวจ =====================
with tab1:
    c1, c2 = st.columns([1, 2.6])
    with c1:
        p1 = param_controls("t1", dict(alpha=1.0, a=2.0, c=0.0), "ปรับพารามิเตอร์")
        show_base = st.checkbox("แสดงกราฟต้นแบบ y = logₐ x", True)
        show_inv = st.checkbox("แสดงฟังก์ชันผกผันและเส้น y = x", False)
    valid1 = not math.isclose(p1["a"], 1)
    with c2:
        if valid1:
            st.latex(latex_eq(p1))
        curves = []
        if show_base and valid1:
            curves.append(dict(params=dict(alpha=1.0, a=p1["a"], c=0.0), name="ต้นแบบ y = logₐ x",
                               color="#aaaaaa", dash="dash", width=2, reference=True))
        curves.append(dict(params=p1, name="กราฟของคุณ", color=COLORS[0]))
        if show_inv and valid1 and p1["alpha"] != 0:
            st.latex(r"f^{-1}:\; " + inverse_latex(p1))
            curves.append(dict(func=lambda x: f_inv(x, **p1), name="ฟังก์ชันผกผัน", color=COLORS[4]))
            curves.append(dict(func=lambda x: x, name="y = x", color="#888888", dash="dot", width=1.5))
        st.plotly_chart(make_fig(curves, x_range, y_range, show_asym, show_points, log_x),
                        use_container_width=True)

    a1, a2 = st.columns(2)
    with a1:
        st.subheader("🧮 สมบัติของกราฟ")
        st.table(pd.DataFrame(analyze(p1).items(), columns=["สมบัติ", "ค่า"]).set_index("สมบัติ"))
    with a2:
        st.subheader("🔄 ขั้นตอนการแปลงจากกราฟต้นแบบ")
        for i, s in enumerate(transformation_steps(p1), 1):
            st.markdown(f"{i}. {s}")
        with st.expander("📋 ตารางค่า: x เป็นลำดับเรขาคณิต → y เป็นลำดับเลขคณิต", expanded=True):
            if valid1:
                n = np.arange(-3, 4)
                xs = p1["a"] ** n.astype(float)
                st.dataframe(pd.DataFrame({"n": n, "x = aⁿ": np.round(xs, 4),
                                           "y": np.round(f(xs, **p1), 4)}),
                             hide_index=True, use_container_width=True)
                st.caption(f"x ถูกคูณด้วย a = {base_txt(p1['a'])} ทุกครั้ง → "
                           f"y เพิ่มขึ้นทีละ α = {fmt(p1['alpha'])} เสมอ")

# ===================== Tab 2: เปรียบเทียบ =====================
with tab2:
    cf, cplot, cg = st.columns([1, 2.4, 1])
    with cf:
        pf = param_controls("f", dict(alpha=1.0, a=2.0, c=0.0), "🔵 กราฟ f")
    with cg:
        pg = param_controls("g", dict(alpha=2.0, a=3.0, c=-1.0), "🔴 กราฟ g")
    pt, msg = intersection(pf, pg)
    with cplot:
        if not math.isclose(pf["a"], 1):
            st.latex(r"f:\; " + latex_eq(pf))
        if not math.isclose(pg["a"], 1):
            st.latex(r"g:\; " + latex_eq(pg))
        extra = [(pt[0], pt[1], "จุดตัด")] if pt else None
        fig = make_fig([dict(params=pf, name="f", color=COLORS[0]),
                        dict(params=pg, name="g", color=COLORS[1])],
                       x_range, y_range, show_asym, show_points, log_x, extra_points=extra)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**จุดตัด:** {msg}")

    st.subheader("📊 ตารางเปรียบเทียบสมบัติ")
    af, ag = analyze(pf), analyze(pg)
    keys = list(dict.fromkeys(list(af) + list(ag)))
    st.table(pd.DataFrame({"สมบัติ": keys,
                           "f": [af.get(k, "—") for k in keys],
                           "g": [ag.get(k, "—") for k in keys]}).set_index("สมบัติ"))

    st.subheader("💡 พารามิเตอร์ที่แตกต่างกันและผลที่เกิดขึ้น")
    diffs = [k for k in PARAMS if not math.isclose(pf[k], pg[k])]
    if not diffs:
        st.info("ทั้งสองกราฟมีพารามิเตอร์เท่ากันทุกตัว ลองเปลี่ยนค่าใดค่าหนึ่งดูครับ")
    for k in diffs:
        st.markdown(f"- **{PARAMS[k][0]}:** f = {fmt(pf[k])}, g = {fmt(pg[k])} → {EFFECTS[k]}")
    if (not math.isclose(pf["a"], 1) and not math.isclose(pg["a"], 1)
            and ("a" in diffs or "alpha" in diffs)):
        kf = pf["alpha"] / math.log(pf["a"])
        kg = pg["alpha"] / math.log(pg["a"])
        if math.isclose(kf, kg, rel_tol=1e-6) and not math.isclose(kf, 0):
            st.success(f"ข้อสังเกต: α / ln a ของทั้งสองกราฟเท่ากัน (= {fmt(kf, 4)}) จึงมีรูปร่างเหมือนกันทุกประการ "
                       "แม้ α และ a จะต่างกัน เช่น 2·log₄ x = log₂ x และ −log₂ x = log₀.₅ x")

# ===================== Tab 3: ตระกูลกราฟ =====================
DEFAULT_FAMILY = {"alpha": "-2, -1, 0.5, 1, 2", "a": "0.25, 0.5, 2, 3, 10", "c": "-2, -1, 0, 1, 2"}

with tab3:
    c1, c2 = st.columns([1, 2.6])
    with c1:
        vary = st.radio("เลือกพารามิเตอร์ที่จะเปลี่ยน", list(PARAMS),
                        format_func=lambda k: f"{PARAMS[k][0]} — {PARAMS[k][4]}", key="fam_vary")
        text = st.text_input("ค่าที่ต้องการ (คั่นด้วย ,)", DEFAULT_FAMILY[vary], key=f"fam_vals_{vary}")
        values = []
        for t in text.split(","):
            try:
                values.append(float(t.strip()))
            except ValueError:
                pass
        if vary == "a":
            bad = [v for v in values if v <= 0 or math.isclose(v, 1)]
            values = [v for v in values if v > 0 and not math.isclose(v, 1)]
            if bad:
                st.warning(f"ตัดค่า a ที่ใช้ไม่ได้ออก (a ≤ 0 หรือ a = 1): {bad}")
        st.markdown("**ค่าคงที่ของพารามิเตอร์อื่น**")
        fixed = {}
        base_default = dict(alpha=1.0, a=2.0, c=0.0)
        for key, (sym, lo, hi, step, desc) in PARAMS.items():
            if key != vary:
                fixed[key] = st.slider(sym, lo, hi, base_default[key], step, key=f"fam_{key}")
        if math.isclose(fixed.get("a", 2.0), 1):
            st.error("a = 1 ใช้ไม่ได้")
    with c2:
        if not values:
            st.error("กรุณาใส่ค่าอย่างน้อย 1 ค่า")
        else:
            n = len(values)
            cols = sample_colorscale("Turbo", [i / (n - 1) if n > 1 else 0.5 for i in range(n)])
            curves = []
            for v, col in zip(values, cols):
                p = dict(fixed)
                p[vary] = v
                curves.append(dict(params=p, name=f"{PARAMS[vary][0]} = {fmt(v)}", color=col))
            st.plotly_chart(make_fig(curves, x_range, y_range, show_asym, show_points, log_x),
                            use_container_width=True)
            st.info(f"**สังเกต:** {EFFECTS[vary]}")


# ===================== Tab 4: แบบฝึกหัด =====================
def new_quiz():
    rng = random.Random()
    p = dict(alpha=rng.choice([-2.0, -1.0, 1.0, 2.0, 3.0]),
             a=rng.choice([2.0, 3.0, 0.5]),
             c=float(rng.randint(-3, 3)))
    xs = np.linspace(0.1, 10, 60)
    options = [p]
    tweaks = [("alpha", lambda v: -v), ("alpha", lambda v: v * 2), ("alpha", lambda v: v / 2),
              ("a", lambda v: 1 / v), ("a", lambda v: 3.0 if v == 2 else 2.0),
              ("c", lambda v: -v if v else 2.0), ("c", lambda v: v + 1), ("c", lambda v: v - 1)]
    rng.shuffle(tweaks)
    for key, fn in tweaks:
        if len(options) == 4:
            break
        q = dict(p)
        q[key] = fn(q[key])
        if all(not np.allclose(f(xs, **q), f(xs, **o), equal_nan=True) for o in options):
            options.append(q)
    rng.shuffle(options)
    st.session_state.quiz = dict(answer=options.index(p), options=options, checked=False)
    st.session_state.quiz_id = st.session_state.get("quiz_id", 0) + 1


if "quiz" not in st.session_state:
    new_quiz()
    st.session_state.score = [0, 0]

with tab4:
    q = st.session_state.quiz
    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown("#### กราฟนี้คือสมการข้อใด?")
        fig = make_fig([dict(params=q["options"][q["answer"]], name="กราฟปริศนา", color=COLORS[2])],
                       (-1, 10), (-6, 6), show_asym=True, show_points=False, height=480)
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        labels = "ABCD"[: len(q["options"])]
        for lab, opt in zip(labels, q["options"]):
            st.markdown(f"**{lab}.**")
            st.latex(latex_eq(opt))
        choice = st.radio("คำตอบของคุณ", list(labels), horizontal=True,
                          key=f"quiz_choice_{st.session_state.quiz_id}")
        b1, b2 = st.columns(2)
        if b1.button("✅ ตรวจคำตอบ", disabled=q["checked"]):
            q["checked"] = True
            st.session_state.score[1] += 1
            q["result"] = labels.index(choice) == q["answer"]
            if q["result"]:
                st.session_state.score[0] += 1
        if b2.button("🔄 ข้อใหม่"):
            new_quiz()
            st.rerun()
        if q["checked"]:
            if q["result"]:
                st.success("ถูกต้อง! 🎉")
            else:
                st.error(f"ยังไม่ถูก คำตอบคือข้อ {labels[q['answer']]}")
            st.caption("คำใบ้: ดูค่า y ที่ x = 1 (ได้ c), ค่า y ที่ x = a (ได้ α + c) "
                       "และดูว่ากราฟเพิ่มหรือลด")
        s = st.session_state.score
        st.metric("คะแนน", f"{s[0]} / {s[1]}")

# ===================== Tab 5: ทฤษฎี =====================
with tab5:
    st.markdown(r"""
### รูปทั่วไป
$y = \alpha \log_a x + c, \qquad a > 0,\; a \neq 1,\; x > 0$

| พารามิเตอร์ | บทบาท | กรณีสำคัญ |
|---|---|---|
| **α** | ยืด/หด แนวตั้ง | α < 0 → สะท้อนกับแกน X |
| **a** | ฐาน กำหนดความชันของกราฟ | a > 1 ฟังก์ชันเพิ่ม, 0 < a < 1 ฟังก์ชันลด (เมื่อ α > 0) |
| **c** | เลื่อนแนวตั้ง | จุด (1, 0) ย้ายไปที่ (1, c) |

### สมบัติสำคัญ
- **โดเมน:** (0, ∞)
- **เรนจ์:** ℝ
- **เส้นกำกับแนวตั้ง:** x = 0 (ไม่มีเส้นกำกับแนวนอน)
- **จุดอ้างอิง:** (1, c) เพราะ logₐ 1 = 0 และ (a, α + c) เพราะ logₐ a = 1
- **จุดตัดแกน X:** x = a^(−c/α) ซึ่งมีเสมอเมื่อ α ≠ 0
- **จุดตัดแกน Y:** ไม่มี
- **ฟังก์ชันเพิ่ม** เมื่อ α / ln a > 0 และ **ฟังก์ชันลด** เมื่อ α / ln a < 0

### เอกลักษณ์ที่ช่วยให้เข้าใจกราฟ
- **เปลี่ยนฐาน:** logₐ x = ln x / ln a ดังนั้น y = (α / ln a)·ln x + c
- **กราฟเดียวกัน:** ถ้า α / ln a เท่ากันและ c เท่ากัน เช่น 2·log₄ x = log₂ x
- **กลับฐาน:** log₁/ₐ x = −logₐ x ดังนั้นการเปลี่ยน a เป็น 1/a ให้ผลเหมือนการสะท้อนกับแกน X
- **ลำดับเรขาคณิต → เลขคณิต:** ถ้าคูณ x ด้วย a ค่า y จะเพิ่มขึ้น α เสมอ
- **สเกล log:** ถ้าแกน x เป็นสเกล log กราฟทุกเส้นจะเป็นเส้นตรง

### ความสัมพันธ์กับฟังก์ชันเอกซ์โพเนนเชียล
ฟังก์ชันผกผันของ y = α logₐ x + c คือ

$y = a^{\frac{x - c}{\alpha}}$

กราฟของฟังก์ชันผกผันคือภาพสะท้อนของกราฟเดิมกับเส้น y = x
ฟังก์ชันนี้คือ y = α' a^(β(x−h)) + c' ที่มี α' = 1, β = 1/α, h = c และ c' = 0
เส้นกำกับแนวตั้ง x = 0 ของลอการิทึมจึงกลายเป็นเส้นกำกับแนวนอน y = 0 ของเอกซ์โพเนนเชียล

### ลำดับการแปลงกราฟจาก y = logₐ x
1. ยืด/หด/สะท้อนแนวตั้งด้วย α
2. เลื่อนแนวตั้งด้วย c
""")
