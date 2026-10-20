import math
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st

st.set_page_config(
    page_title="เรียนรู้กราฟฟังก์ชันเอกซ์โพเนนเชียล",
    page_icon="📈",
    layout="wide",
)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

# ชื่อ, ค่าต่ำสุด, ค่าสูงสุด, step, คำอธิบาย
PARAMS = {
    "alpha": ("α", -5.0, 5.0, 0.1, "ยืด/หด และสะท้อนในแนวตั้ง"),
    "a": ("a", 0.1, 5.0, 0.1, "ฐาน (a > 0, a ≠ 1)"),
    "beta": ("β", -3.0, 3.0, 0.1, "ยืด/หด และสะท้อนในแนวนอน"),
    "h": ("h", -5.0, 5.0, 0.1, "เลื่อนแนวนอน"),
    "c": ("c", -5.0, 5.0, 0.1, "เลื่อนแนวตั้ง / เส้นกำกับ"),
}

EFFECTS = {
    "alpha": "|α| ยืด (|α|>1) หรือหด (|α|<1) กราฟในแนวตั้งโดยวัดจากเส้นกำกับ y = c และถ้า α < 0 กราฟจะสะท้อนกับเส้น y = c เรนจ์จึงเปลี่ยนข้าง",
    "a": "a กำหนดอัตราการเปลี่ยนแปลง: ถ้า a > 1 ค่า a ยิ่งมาก กราฟยิ่งชัน ส่วน 0 < a < 1 จะกลายเป็นการสลายตัว (เมื่อ β > 0)",
    "beta": "|β| ยิ่งมาก กราฟยิ่งหดในแนวนอน (ชันขึ้น) และ β < 0 ทำให้กราฟสะท้อนกับเส้นแนวตั้ง x = h",
    "h": "h เลื่อนกราฟในแนวนอน: h > 0 เลื่อนไปทางขวา และ h < 0 เลื่อนไปทางซ้าย (รูปร่างกราฟเหมือนเดิม)",
    "c": "c เลื่อนกราฟขึ้น/ลง และเป็นตำแหน่งของเส้นกำกับแนวนอน y = c",
}


# ===================== ส่วนคณิตศาสตร์ =====================
def f(x, alpha, a, beta, h, c):
    with np.errstate(over="ignore", invalid="ignore"):
        y = alpha * np.power(a, beta * (x - h)) + c
    return np.where(np.isfinite(y), y, np.nan)


def fmt(v, nd=3):
    v = round(float(v), nd)
    if v == 0:
        v = 0.0
    return f"{v:g}"


def base_str(a):
    return "e" if math.isclose(a, math.e) else f"({fmt(a)})"


def latex_eq(p):
    alpha, a, beta, h, c = p["alpha"], p["a"], p["beta"], p["h"], p["c"]
    inner = "x" if h == 0 else (f"x - {fmt(h)}" if h > 0 else f"x + {fmt(-h)}")
    if beta == 1:
        expo = inner
    elif beta == -1:
        expo = "-x" if h == 0 else f"-({inner})"
    else:
        expo = f"{fmt(beta)}x" if h == 0 else f"{fmt(beta)}({inner})"
    coef = "" if alpha == 1 else ("-" if alpha == -1 else f"{fmt(alpha)}\\cdot ")
    const = "" if c == 0 else (f" + {fmt(c)}" if c > 0 else f" - {fmt(-c)}")
    return f"y = {coef}{base_str(a)}^{{{expo}}}{const}"


def analyze(p):
    alpha, a, beta, h, c = p["alpha"], p["a"], p["beta"], p["h"], p["c"]
    k = beta * math.log(a)  # y - c = α·e^{k(x-h)}
    rows = {}
    if math.isclose(alpha, 0, abs_tol=1e-12) or math.isclose(k, 0, abs_tol=1e-12):
        val = c if math.isclose(alpha, 0, abs_tol=1e-12) else alpha + c
        rows["ลักษณะกราฟ"] = f"ฟังก์ชันคงตัว y = {fmt(val)} (เส้นตรงแนวนอน)"
        rows["เหตุผล"] = "α = 0" if math.isclose(alpha, 0, abs_tol=1e-12) else "a = 1 หรือ β = 0 ทำให้ a^(β(x−h)) = 1 เสมอ"
        return rows

    rows["แนวโน้มของกราฟ"] = "ฟังก์ชันเพิ่ม ↗" if alpha * k > 0 else "ฟังก์ชันลด ↘"
    rows["ชนิด (พิจารณา |y − c|)"] = "การเติบโต (growth)" if k > 0 else "การสลายตัว (decay)"
    rows["ฐานประสิทธิผล a^β"] = f"{fmt(a ** beta, 4)} (x เพิ่ม 1 หน่วย → |y − c| คูณด้วยค่านี้)"
    rows["โดเมน"] = "ℝ (จำนวนจริงทั้งหมด)"
    rows["เรนจ์"] = f"({fmt(c)}, ∞)" if alpha > 0 else f"(−∞, {fmt(c)})"
    side = "x → −∞" if k > 0 else "x → +∞"
    rows["เส้นกำกับแนวนอน"] = f"y = {fmt(c)} (กราฟเข้าใกล้เมื่อ {side})"
    rows["จุดอ้างอิง (h, α + c)"] = f"({fmt(h)}, {fmt(alpha + c)})"
    try:
        rows["จุดตัดแกน Y"] = f"(0, {fmt(alpha * a ** (-beta * h) + c, 4)})"
    except OverflowError:
        rows["จุดตัดแกน Y"] = "ค่ามากเกินกว่าจะคำนวณได้"
    r = -c / alpha
    rows["จุดตัดแกน X"] = (
        f"({fmt(h + math.log(r) / k, 4)}, 0)" if r > 0 else "ไม่มี (กราฟไม่ตัดแกน X)"
    )
    rows["ความเว้า"] = "เว้าขึ้น ∪" if alpha > 0 else "เว้าลง ∩"
    t = math.log(2) / abs(k)
    label = "ระยะที่ |y − c| เพิ่มเป็น 2 เท่า" if k > 0 else "ระยะที่ |y − c| ลดเหลือครึ่ง"
    rows[label] = f"{fmt(t, 4)} หน่วยตามแกน x"
    return rows


def transformation_steps(p):
    alpha, a, beta, h, c = p["alpha"], p["a"], p["beta"], p["h"], p["c"]
    b = base_str(a)
    steps = [f"เริ่มจากกราฟต้นแบบ $y = {b}^x$"]
    if beta != 1:
        parts = []
        if beta == 0:
            parts.append("กราฟกลายเป็นค่าคงตัว")
        else:
            if abs(beta) != 1:
                word = "หด" if abs(beta) > 1 else "ขยาย"
                parts.append(f"{word}ในแนวนอนด้วยตัวคูณ 1/|β| = {fmt(1 / abs(beta))}")
            if beta < 0:
                parts.append("สะท้อนกับแกน Y")
        steps.append(f"แทน x ด้วย {fmt(beta)}x → $y = {b}^{{{fmt(beta)}x}}$ : " + ", ".join(parts))
    if h != 0:
        steps.append(f"เลื่อนไปทาง{'ขวา' if h > 0 else 'ซ้าย'} {fmt(abs(h))} หน่วย (h = {fmt(h)})")
    if alpha != 1:
        parts = []
        if alpha == 0:
            parts.append("กราฟถูกบีบเป็นเส้น y = 0")
        else:
            if abs(alpha) != 1:
                word = "ยืด" if abs(alpha) > 1 else "หด"
                parts.append(f"{word}ในแนวตั้งด้วยตัวคูณ {fmt(abs(alpha))}")
            if alpha < 0:
                parts.append("สะท้อนกับแกน X")
        steps.append(f"คูณด้วย α = {fmt(alpha)} : " + ", ".join(parts))
    if c != 0:
        steps.append(f"เลื่อน{'ขึ้น' if c > 0 else 'ลง'} {fmt(abs(c))} หน่วย → เส้นกำกับย้ายไปที่ y = {fmt(c)}")
    return steps


# ===================== ส่วนกราฟ =====================
def make_fig(curves, x_range, y_range, show_asym=True, show_points=True, height=550):
    fig = go.Figure()
    x = np.linspace(x_range[0], x_range[1], 1000)
    for i, cv in enumerate(curves):
        p = cv["params"]
        color = cv.get("color", COLORS[i % len(COLORS)])
        fig.add_trace(go.Scatter(
            x=x, y=f(x, **p), mode="lines", name=cv["name"],
            line=dict(color=color, width=cv.get("width", 3), dash=cv.get("dash", "solid")),
        ))
        if cv.get("reference"):
            continue
        if show_asym:
            fig.add_hline(y=p["c"], line_dash="dot", line_color=color, opacity=0.7)
        if show_points:
            fig.add_trace(go.Scatter(
                x=[p["h"]], y=[p["alpha"] + p["c"]], mode="markers",
                marker=dict(color=color, size=11, symbol="diamond"),
                showlegend=False, name=f"(h, α+c) {cv['name']}",
            ))
    fig.update_layout(
        height=height,
        xaxis=dict(range=list(x_range), zeroline=True, zerolinewidth=2, zerolinecolor="#444", title="x"),
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
        if key == "a":
            if st.checkbox("ใช้ฐาน e ≈ 2.718", key=f"{prefix}_use_e"):
                p["a"] = math.e
                continue
        p[key] = st.slider(f"{sym} — {desc}", lo, hi, float(d[key]), step, key=f"{prefix}_{key}")
    if math.isclose(p["a"], 1.0):
        st.warning("a = 1 ทำให้กราฟเป็นเส้นตรงแนวนอน (ไม่ใช่ฟังก์ชันเอกซ์โพเนนเชียล)")
    if p["alpha"] == 0:
        st.warning("α = 0 ทำให้ y = c เป็นค่าคงตัว")
    return p


# ===================== Sidebar =====================
st.sidebar.header("⚙️ ตั้งค่าการแสดงผล")
x_range = st.sidebar.slider("ช่วงแกน x", -20.0, 20.0, (-6.0, 6.0), 0.5)
y_range = st.sidebar.slider("ช่วงแกน y", -50.0, 50.0, (-10.0, 10.0), 0.5)
show_asym = st.sidebar.checkbox("แสดงเส้นกำกับ y = c", True)
show_points = st.sidebar.checkbox("แสดงจุดอ้างอิง (h, α + c)", True)
st.sidebar.markdown("---")
st.sidebar.latex(r"y = \alpha \, a^{\beta (x-h)} + c")
st.sidebar.caption("◆ = จุดอ้างอิง (h, α + c)   ┄ = เส้นกำกับแนวนอน")

st.title("📈 เรียนรู้กราฟฟังก์ชันเอกซ์โพเนนเชียล")
st.latex(r"y = \alpha \, a^{\beta (x-h)} + c")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔍 สำรวจกราฟ", "⚖️ เปรียบเทียบ 2 กราฟ", "🌈 ตระกูลกราฟ", "🎯 แบบฝึกหัด", "📚 สรุปทฤษฎี"]
)

# ===================== Tab 1: สำรวจ =====================
with tab1:
    c1, c2 = st.columns([1, 2.6])
    with c1:
        p1 = param_controls("t1", dict(alpha=1.0, a=2.0, beta=1.0, h=0.0, c=0.0), "ปรับพารามิเตอร์")
        show_base = st.checkbox("แสดงกราฟต้นแบบ y = aˣ", True)
    with c2:
        st.latex(latex_eq(p1))
        curves = []
        if show_base:
            base = dict(alpha=1.0, a=p1["a"], beta=1.0, h=0.0, c=0.0)
            curves.append(dict(params=base, name="ต้นแบบ y = aˣ", color="#aaaaaa",
                               dash="dash", width=2, reference=True))
        curves.append(dict(params=p1, name="กราฟของคุณ", color=COLORS[0]))
        st.plotly_chart(make_fig(curves, x_range, y_range, show_asym, show_points),
                        use_container_width=True)

    a1, a2 = st.columns(2)
    with a1:
        st.subheader("🧮 สมบัติของกราฟ")
        st.table(pd.DataFrame(analyze(p1).items(), columns=["สมบัติ", "ค่า"]).set_index("สมบัติ"))
    with a2:
        st.subheader("🔄 ขั้นตอนการแปลงจากกราฟต้นแบบ")
        for i, s in enumerate(transformation_steps(p1), 1):
            st.markdown(f"{i}. {s}")
        with st.expander("📋 ตารางค่า x, y"):
            xs = np.arange(math.ceil(x_range[0]), math.floor(x_range[1]) + 1)
            st.dataframe(pd.DataFrame({"x": xs, "y": np.round(f(xs, **p1), 4)}),
                         hide_index=True, use_container_width=True)

# ===================== Tab 2: เปรียบเทียบ =====================
with tab2:
    cf, cplot, cg = st.columns([1, 2.4, 1])
    with cf:
        pf = param_controls("f", dict(alpha=1.0, a=2.0, beta=1.0, h=0.0, c=0.0), "🔵 กราฟ f")
    with cg:
        pg = param_controls("g", dict(alpha=2.0, a=2.0, beta=1.0, h=1.0, c=-1.0), "🔴 กราฟ g")
    with cplot:
        st.latex(r"f:\; " + latex_eq(pf))
        st.latex(r"g:\; " + latex_eq(pg))
        fig = make_fig(
            [dict(params=pf, name="f", color=COLORS[0]), dict(params=pg, name="g", color=COLORS[1])],
            x_range, y_range, show_asym, show_points,
        )
        st.plotly_chart(fig, use_container_width=True)

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
        sym = PARAMS[k][0]
        st.markdown(f"- **{sym}:** f = {fmt(pf[k])}, g = {fmt(pg[k])} → {EFFECTS[k]}")
    if not math.isclose(pf["a"] ** pf["beta"], 1) and math.isclose(pf["a"] ** pf["beta"], pg["a"] ** pg["beta"]) \
            and ("a" in diffs or "beta" in diffs):
        st.success(f"ข้อสังเกต: a^β ของทั้งสองกราฟเท่ากัน (= {fmt(pf['a'] ** pf['beta'], 4)}) "
                   "จึงมีอัตราการเปลี่ยนแปลงเท่ากัน แม้ a และ β จะต่างกัน")

# ===================== Tab 3: ตระกูลกราฟ =====================
DEFAULT_FAMILY = {"alpha": "-2, -1, 0.5, 1, 2", "a": "0.5, 0.8, 1.5, 2, 3",
                  "beta": "-2, -1, 0.5, 1, 2", "h": "-2, -1, 0, 1, 2", "c": "-2, -1, 0, 1, 2"}

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
            bad = [v for v in values if v <= 0]
            values = [v for v in values if v > 0]
            if bad:
                st.warning(f"ตัดค่า a ≤ 0 ออก: {bad}")
        st.markdown("**ค่าคงที่ของพารามิเตอร์อื่น**")
        fixed = {}
        base_default = dict(alpha=1.0, a=2.0, beta=1.0, h=0.0, c=0.0)
        for key, (sym, lo, hi, step, desc) in PARAMS.items():
            if key != vary:
                fixed[key] = st.slider(sym, lo, hi, base_default[key], step, key=f"fam_{key}")
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
            st.plotly_chart(make_fig(curves, x_range, y_range, show_asym, show_points),
                            use_container_width=True)
            st.info(f"**สังเกต:** {EFFECTS[vary]}")

# ===================== Tab 4: แบบฝึกหัด =====================
def new_quiz():
    rng = random.Random()
    p = dict(alpha=rng.choice([-2.0, -1.0, 0.5, 1.0, 2.0]), a=rng.choice([0.5, 2.0, 3.0]),
             beta=rng.choice([-1.0, 1.0, 2.0]), h=float(rng.randint(-2, 2)), c=float(rng.randint(-3, 3)))
    xs = np.linspace(-5, 5, 41)
    options = [p]
    tweaks = [("alpha", lambda v: -v), ("beta", lambda v: -v), ("h", lambda v: -v if v else 1.0),
              ("c", lambda v: -v if v else 2.0), ("c", lambda v: v + 1), ("h", lambda v: v + 1),
              ("alpha", lambda v: v * 2)]
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
                       (-6, 6), (-10, 10), show_asym=True, show_points=False, height=480)
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
            if labels.index(choice) == q["answer"]:
                st.session_state.score[0] += 1
                q["result"] = True
            else:
                q["result"] = False
        if b2.button("🔄 ข้อใหม่"):
            new_quiz()
            st.rerun()
        if q["checked"]:
            if q["result"]:
                st.success("ถูกต้อง! 🎉")
            else:
                st.error(f"ยังไม่ถูก คำตอบคือข้อ {labels[q['answer']]}")
            st.caption("คำใบ้: ดูเส้นกำกับ (c), ทิศทางเพิ่ม/ลด (α, β, a), จุดตัดแกน Y และจุด (h, α + c)")
        s = st.session_state.score
        st.metric("คะแนน", f"{s[0]} / {s[1]}")

# ===================== Tab 5: ทฤษฎี =====================
with tab5:
    st.markdown(r"""
### รูปทั่วไป
$y = \alpha \, a^{\beta (x-h)} + c, \qquad a > 0,\; a \neq 1$

| พารามิเตอร์ | บทบาท | กรณีสำคัญ |
|---|---|---|
| **α** | ยืด/หด แนวตั้ง | α < 0 → สะท้อนกับเส้น y = c, เรนจ์เปลี่ยนเป็น (−∞, c) |
| **a** | ฐาน กำหนดอัตราการเปลี่ยนแปลง | a > 1 เติบโต, 0 < a < 1 สลายตัว (เมื่อ β > 0) |
| **β** | ยืด/หด แนวนอน | \|β\| > 1 หด (ชันขึ้น), β < 0 สะท้อนกับเส้น x = h |
| **h** | เลื่อนแนวนอน | h > 0 เลื่อนขวา, h < 0 เลื่อนซ้าย |
| **c** | เลื่อนแนวตั้ง | เส้นกำกับแนวนอนคือ y = c |

### สูตรสำคัญ
- **โดเมน:** ℝ
- **เรนจ์:** (c, ∞) ถ้า α > 0 และ (−∞, c) ถ้า α < 0
- **จุดอ้างอิง:** (h, α + c) เพราะ a⁰ = 1
- **จุดตัดแกน Y:** y = α·a^(−βh) + c
- **จุดตัดแกน X:** มีเมื่อ −c/α > 0 ที่ x = h + log_a(−c/α) / β
- **ฟังก์ชันเพิ่ม** เมื่อ α·β·ln a > 0 และ **ฟังก์ชันลด** เมื่อ α·β·ln a < 0
- **ฐานประสิทธิผล:** a^β เช่น 2^(−x) = (1/2)^x ดังนั้น a = 2, β = −1 ให้กราฟเดียวกับ a = 0.5, β = 1

### ลำดับการแปลงกราฟจาก y = aˣ
1. ยืด/หด/สะท้อนแนวนอนด้วย β
2. เลื่อนแนวนอนด้วย h
3. ยืด/หด/สะท้อนแนวตั้งด้วย α
4. เลื่อนแนวตั้งด้วย c
""")
