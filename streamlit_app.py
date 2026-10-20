import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------
# 1. การตั้งค่าหน้าเว็บ (Page Configuration)
# ---------------------------------------------------------
st.set_page_config(
    page_title="การศึกษาพฤติกรรมกราฟ y = α · log_a(β(x-h)) + c",
    page_icon="📉",
    layout="wide"
)

st.title("📉 การศึกษาพฤติกรรมของกราฟลอการิทึม (Logarithmic Graph Explorer)")
st.latex(r"y = \alpha \cdot \log_a(\beta(x - h)) + c")

st.markdown("""
เว็บแอปพลิเคชันนี้ออกแบบมาเพื่อศึกษาพฤติกรรมและการเปลี่ยนแปลงของกราฟลอการิทึมตามสมการ 
**$y = \\alpha \\cdot \\log_a(\\beta(x - h)) + c$** ท่านสามารถปรับเปลี่ยนค่าพารามิเตอร์ $\\alpha, a, \\beta, h, c$ บนแถบด้านข้าง
เพื่อดูการเปลี่ยนแปลงของกราฟ เส้นกำกับแนวตั้ง โดเมน จุดตัดแกน และพฤติกรรมลิมิตได้แบบเรียลไทม์
""")

# ---------------------------------------------------------
# 2. แถบควบคุมพารามิเตอร์ (Sidebar)
# ---------------------------------------------------------
st.sidebar.header("⚙️ กำหนดพารามิเตอร์สมการ")
st.sidebar.latex(r"y = \alpha \cdot \log_a(\beta(x - h)) + c")

st.sidebar.subheader("📌 พารามิเตอร์ Graph 1")

# ตัวเลือกการกำหนดฐาน a
base_option = st.sidebar.radio("เลือกประเภทฐาน a:", ["กำหนดค่า a เอง", "ฐานธรรมชาติ e (ln)"])
if base_option == "ฐานธรรมชาติ e (ln)":
    a = np.e
else:
    a = st.sidebar.number_input("ฐาน a (a > 0, a ≠ 1):", value=2.0, min_value=0.0001, step=0.1)

if a == 1.0:
    st.sidebar.error("❌ ฐาน a ต้องไม่เท่ากับ 1")

alpha = st.sidebar.number_input("สัมประสิทธิ์ α (Alpha - ยืด/สะท้อนแนวตั้ง):", value=1.0, step=0.5)
beta = st.sidebar.number_input("ตัวคูณกำลัง β (Beta - ยืด/สะท้อนแนวนอน):", value=1.0, step=0.5)
if beta == 0:
    st.sidebar.error("❌ ค่า β ต้องไม่เท่ากับ 0")

h = st.sidebar.number_input("การเลื่อนแนวนอน h (เส้นกำกับแนวตั้ง x = h):", value=0.0, step=0.5)
c = st.sidebar.number_input("การเลื่อนแนวตั้ง c (Vertical Shift):", value=0.0, step=0.5)

# โหมดเปรียบเทียบกราฟ 2
compare_mode = st.sidebar.checkbox("🔄 เปิดโหมดเปรียบเทียบกับ Graph 2")
if compare_mode:
    st.sidebar.subheader("📌 พารามิเตอร์ Graph 2")
    base_option2 = st.sidebar.radio("เลือกประเภทฐาน a2:", ["กำหนดค่า a2 เอง", "ฐานธรรมชาติ e (ln)"], key="base2")
    if base_option2 == "ฐานธรรมชาติ e (ln)":
        a2 = np.e
    else:
        a2 = st.sidebar.number_input("ฐาน a2:", value=0.5, min_value=0.0001, step=0.1, key="a2_val")
        
    alpha2 = st.sidebar.number_input("สัมประสิทธิ์ α2:", value=1.0, step=0.5, key="alpha2")
    beta2 = st.sidebar.number_input("ตัวคูณกำลัง β2:", value=-1.0, step=0.5, key="beta2")
    h2 = st.sidebar.number_input("การเลื่อนแนวนอน h2:", value=0.0, step=0.5, key="h2")
    c2 = st.sidebar.number_input("การเลื่อนแนวตั้ง c2:", value=0.0, step=0.5, key="c2")

st.sidebar.subheader("🔍 ขอบเขตแกน X")
x_min = st.sidebar.number_input("X Min", value=-10.0, step=1.0)
x_max = st.sidebar.number_input("X Max", value=10.0, step=1.0)
if x_min >= x_max:
    st.sidebar.error("❌ ค่า X Min ต้องน้อยกว่า X Max")

# ส่วนปรับแต่ง Labels
st.sidebar.subheader("🏷️ ปรับแต่ง Labels บนกราฟ")
custom_title = st.sidebar.text_input("ชื่อหัวเรื่องกราฟ (Graph Title Label):", value="กราฟแสดงฟังก์ชัน y = α · log_a(β(x - h)) + c")
custom_xlabel = st.sidebar.text_input("Label แกน X (X-Axis Label):", value="แกน x")
custom_ylabel = st.sidebar.text_input("Label แกน Y (Y-Axis Label):", value="แกน y")

show_annotations = st.sidebar.checkbox("📌 แสดง Annotations ชี้บอกจุดสำคัญ", value=True)
show_data_labels = st.sidebar.checkbox("🔢 แสดง Data Labels พิกัดจุดบนกราฟ", value=False)

# ---------------------------------------------------------
# 3. ฟังก์ชันคำนวณ และการสร้างกราฟ Plotly
# ---------------------------------------------------------
def eval_logarithm(x, alpha_val, a_val, beta_val, h_val, c_val):
    arg = beta_val * (x - h_val)
    with np.errstate(invalid='ignore', divide='ignore'):
        y = alpha_val * (np.log(arg) / np.log(a_val)) + c_val
        y[arg <= 0] = np.nan
    return y

x_vals = np.linspace(x_min, x_max, 1200)
y_vals = eval_logarithm(x_vals, alpha, a, beta, h, c)

fig = go.Figure()

# หาขอบเขต Y สำหรับวาดเส้นกำกับแนวตั้ง
valid_y = y_vals[~np.isnan(y_vals)]
if len(valid_y) > 0:
    y_plot_min, y_plot_max = np.min(valid_y) - 2, np.max(valid_y) + 2
else:
    y_plot_min, y_plot_max = -10, 10

# 1. เส้นกำกับแนวตั้ง Vertical Asymptote (x = h)
fig.add_trace(go.Scatter(
    x=[h, h],
    y=[y_plot_min, y_plot_max],
    mode='lines',
    name=f'เส้นกำกับ Vertical Asymptote (x = {h})',
    line=dict(color='crimson', dash='dash', width=2)
))

# 2. เส้นกราฟหลัก (Graph 1)
a_disp = "ln" if np.isclose(a, np.e) else f"log_{{{round(a, 2)}}}"
if np.isclose(a, np.e):
    eq_label = f"y = {alpha} · ln({beta}(x - {h})) + {c}"
else:
    eq_label = f"y = {alpha} · log_{{{round(a, 2)}}}({beta}(x - {h})) + {c}"

fig.add_trace(go.Scatter(
    x=x_vals,
    y=y_vals,
    mode='lines',
    name=f'Graph 1: {eq_label}',
    line=dict(color='#1f77b4', width=3),
    connectgaps=False,
    hovertemplate='<b>x</b>: %{x:.3f}<br><b>y</b>: %{y:.3f}<extra></extra>'
))

# 3. Data Labels (แสดงพิกัดจุดตัวอย่างช่วงที่อยู่ในโดเมน)
if show_data_labels:
    sample_x = np.linspace(x_min, x_max, 25)
    sample_y = eval_logarithm(sample_x, alpha, a, beta, h, c)
    valid_idx = ~np.isnan(sample_y)
    
    fig.add_trace(go.Scatter(
        x=sample_x[valid_idx],
        y=sample_y[valid_idx],
        mode='markers+text',
        name='Data Labels',
        marker=dict(color='#1f77b4', size=7),
        text=[f"({x:.1f}, {y:.1f})" for x, y in zip(sample_x[valid_idx], sample_y[valid_idx])],
        textposition="top center",
        textfont=dict(size=9, color="darkblue")
    ))

# 4. คำนวณจุดตัดแกน X (y = 0)
# α log_a(β(x - h)) + c = 0 ➔ x = h + (a^(-c/α)) / β
x_intercept = None
if alpha != 0 and beta != 0 and a > 0 and a != 1:
    x_int_val = h + (a ** (-c / alpha)) / beta
    if x_min <= x_int_val <= x_max:
        x_intercept = x_int_val
        fig.add_trace(go.Scatter(
            x=[x_intercept],
            y=[0],
            mode='markers',
            name=f'จุดตัดแกน X ({x_intercept:.2f}, 0)',
            marker=dict(color='green', size=11, symbol='diamond')
        ))

# 5. คำนวณจุดตัดแกน Y (x = 0)
y_intercept = None
if x_min <= 0 <= x_max:
    y_int_arr = eval_logarithm(np.array([0.0]), alpha, a, beta, h, c)
    if not np.isnan(y_int_arr[0]):
        y_intercept = y_int_arr[0]
        fig.add_trace(go.Scatter(
            x=[0],
            y=[y_intercept],
            mode='markers',
            name=f'จุดตัดแกน Y (0, {y_intercept:.2f})',
            marker=dict(color='purple', size=11, symbol='circle')
        ))

# 6. กราฟเปรียบเทียบ (Graph 2)
if compare_mode:
    y_vals2 = eval_logarithm(x_vals, alpha2, a2, beta2, h2, c2)
    eq_label2 = f"y = {alpha2} · log_{round(a2,2) if not np.isclose(a2, np.e) else 'e'}({beta2}(x - {h2})) + {c2}"
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals2,
        mode='lines',
        name=f'Graph 2: {eq_label2}',
        line=dict(color='#ff7f0e', width=3, dash='dot'),
        connectgaps=False,
        hovertemplate='<b>x</b>: %{x:.3f}<br><b>y</b>: %{y:.3f}<extra></extra>'
    ))
    if h2 != h:
        fig.add_trace(go.Scatter(
            x=[h2, h2],
            y=[y_plot_min, y_plot_max],
            mode='lines',
            name=f'เส้นกำกับ Graph 2 (x = {h2})',
            line=dict(color='orange', dash='dash', width=1.5)
        ))

# ---------------------------------------------------------
# Annotations ชี้ตำแหน่งสำคัญบนกราฟ
# ---------------------------------------------------------
# คำนวณความชันเพื่อระบุประเภทฟังก์ชัน dy/dx = α / ((x-h) * ln(a))
# สัญญาณความชันขึ้นกับ sign(α * β / ln(a))
growth_indicator = (alpha * beta) / np.log(a) if a > 0 and a != 1 else 0

if show_annotations:
    # Annotation เส้นกำกับแนวตั้ง x = h
    fig.add_annotation(
        x=h,
        y=y_plot_max if len(valid_y) > 0 else 5,
        text=f" Vertical Asymptote: x = {h}",
        showarrow=False,
        xanchor="left" if beta > 0 else "right",
        font=dict(color="crimson", size=12),
        bgcolor="rgba(255, 230, 230, 0.85)"
    )

    # Annotation จุดตัดแกน X
    if x_intercept is not None:
        fig.add_annotation(
            x=x_intercept,
            y=0,
            text=f"📍 จุดตัดแกน X: ({x_intercept:.2f}, 0)",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            ax=40,
            ay=-35,
            bgcolor="rgba(235, 255, 235, 0.9)",
            bordercolor="green",
            borderwidth=1
        )

    # Annotation จุดตัดแกน Y
    if y_intercept is not None:
        fig.add_annotation(
            x=0,
            y=y_intercept,
            text=f"📍 จุดตัดแกน Y: (0, {y_intercept:.2f})",
            showarrow=True,
            arrowhead=2,
            arrowcolor="purple",
            ax=-50,
            ay=35,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="purple",
            borderwidth=1
        )

    # Annotation บอกชนิดกราฟ
    if np.isclose(growth_indicator, 0):
        behavior_text = "🔵 Constant Line"
    elif growth_indicator > 0:
        behavior_text = "📈 Logarithmic Increasing (ฟังก์ชันเพิ่ม)"
    else:
        behavior_text = "📉 Logarithmic Decreasing (ฟังก์ชันลด)"

    if len(valid_y) > 0:
        mid_valid_x = np.nanmean(x_vals[~np.isnan(y_vals)])
        mid_valid_y = eval_logarithm(np.array([mid_valid_x]), alpha, a, beta, h, c)[0]
        
        fig.add_annotation(
            x=mid_valid_x,
            y=mid_valid_y,
            text=f"<b>{behavior_text}</b>",
            showarrow=True,
            arrowhead=1,
            arrowcolor="#1f77b4",
            ax=0,
            ay=-45,
            bgcolor="rgba(240, 248, 255, 0.9)",
            bordercolor="#1f77b4",
            font=dict(size=12, color="#1f77b4")
        )

# ปรับ Layout & Graph Titles
fig.update_layout(
    title=dict(text=f"<b>{custom_title}</b>", font=dict(size=18)),
    xaxis_title=custom_xlabel,
    yaxis_title=custom_ylabel,
    hovermode="x unified",
    template="plotly_white",
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.75)"),
    dragmode="pan"
)

# ---------------------------------------------------------
# 4. การแสดงผลบนหน้าจอ (Main Layout)
# ---------------------------------------------------------
col_graph, col_info = st.columns([3, 2])

with col_graph:
    st.plotly_chart(fig, use_container_width=True)
    st.info("""
    💡 **วิธี Zoom & Control มุมมอง:**
    - **Zoom In / Zoom Out:** หมุน Scroll Wheel ของเมาส์
    - **Pan (ลากขยับมุมมอง):** คลิกเมาส์ค้างแล้วลากไปมาได้อย่างอิสระ
    - **Reset View:** ดับเบิ้ลคลิกบนพื้นที่กราฟเพื่อกลับสู่มุมมองเริ่มต้น
    """)

with col_info:
    st.subheader("🔍 วิเคราะห์พฤติกรรมทางคณิตศาสตร์")
    
    # วิเคราะห์ฟังก์ชันเพิ่ม / ฟังก์ชันลด
    if np.isclose(growth_indicator, 0):
        st.info("🔵 **ชนิดฟังก์ชัน:** ฟังก์ชันคงตัว")
    elif growth_indicator > 0:
        st.success("🟢 **ชนิดฟังก์ชัน:** ฟังก์ชันเพิ่ม (Strictly Increasing)")
    else:
        st.warning("🟠 **ชนิดฟังก์ชัน:** ฟังก์ชันลด (Strictly Decreasing)")

    st.markdown("---")
    st.markdown("### 📌 สรุปคุณสมบัติของ $y = \\alpha \\cdot \\log_a(\\beta(x-h)) + c$")
    
    # Domain Calculation
    domain_str = f"({h}, +\\infty)" if beta > 0 else f"(-\\infty, {h})"
    
    st.write(f"- **โดเมน (Domain):** ${domain_str}$")
    st.write(f"- **เรนจ์ (Range):** $(-\\infty, +\\infty)$ หรือ จำนวนจริงทั้งหมด")
    st.write(f"- **เส้นกำกับแนวตั้ง (Vertical Asymptote):** $x = {h}$")
    
    if x_intercept is not None:
        st.write(f"- **จุดตัดแกน X (X-intercept):** $({x_intercept:.4f}, 0)$")
    else:
        st.write("- **จุดตัดแกน X:** อยู่นอกขอบเขตมุมมอง หรือไม่มีจุดตัด")

    if y_intercept is not None:
        st.write(f"- **จุดตัดแกน Y (Y-intercept):** $(0, {y_intercept:.4f})$")
    else:
        st.write("- **จุดตัดแกน Y:** ไม่อยู่ในโดเมนของฟังก์ชัน (ไม่ตัดแกน Y)")

    st.markdown("---")
    st.markdown("### ♾️ พฤติกรรมลิมิตใกล้เส้นกำกับและอินฟินิตี้")
    
    # Limits Analysis
    if beta > 0:
        near_asymptote = "x \\to h^+"
        far_infinity = "x \\to +\\infty"
    else:
        near_asymptote = "x \\to h^-"
        far_infinity = "x \\to -\\infty"

    if growth_indicator > 0:
        lim_asymptote = "-\\infty" if alpha > 0 else "+\\infty"
        lim_far = "+\\infty" if alpha > 0 else "-\\infty"
    else:
        lim_asymptote = "+\\infty" if alpha < 0 else "-\\infty"
        lim_far = "-\\infty" if alpha < 0 else "+\\infty"

    st.latex(f"\\lim_{{{near_asymptote}}} y = {lim_asymptote}")
    st.latex(f"\\lim_{{{far_infinity}}} y = {lim_far}")

# ---------------------------------------------------------
# 5. แท็บรายละเอียดเพิ่มเติม
# ---------------------------------------------------------
st.divider()
tab1, tab2, tab3 = st.tabs(["🧮 คำนวณค่าพิกัด (X ➔ Y)", "📋 ตารางค่า (X, Y)", "📚 อธิบายหน้าที่ของแต่ละพารามิเตอร์"])

with tab1:
    st.subheader("คำนวณค่า Y จากค่า X ที่กำหนด")
    col_in, col_out = st.columns(2)
    with col_in:
        test_x = st.number_input("ป้อนค่า X:", value=h + (1.0 / abs(beta)), step=0.5)
    with col_out:
        res_arr = eval_logarithm(np.array([test_x]), alpha, a, beta, h, c)
        if np.isnan(res_arr[0]):
            st.error(f"❌ ค่า X = {test_x} ไม่อยู่ในโดเมน ${domain_str}$")
        else:
            st.metric(label=f"ค่า Y ของ Graph 1 ที่ X = {test_x}", value=f"{res_arr[0]:.6f}")
        
        if compare_mode:
            res_arr2 = eval_logarithm(np.array([test_x]), alpha2, a2, beta2, h2, c2)
            if not np.isnan(res_arr2[0]):
                st.metric(label=f"ค่า Y ของ Graph 2 ที่ X = {test_x}", value=f"{res_arr2[0]:.6f}")

with tab2:
    st.subheader("ตารางพิกัด (X, Y) เฉพาะช่วงที่อยู่ในโดเมน")
    sample_x = np.linspace(x_min, x_max, 21)
    sample_y = eval_logarithm(sample_x, alpha, a, beta, h, c)
    df_data = {"X": sample_x, "Y (Graph 1)": sample_y}
    if compare_mode:
        df_data["Y (Graph 2)"] = eval_logarithm(sample_x, alpha2, a2, beta2, h2, c2)
    df = pd.DataFrame(df_data)
    st.dataframe(df.style.format("{:.4f}", na_rep="Out of Domain"), use_container_width=True)

with tab3:
    st.markdown(r"""
    ### 📚 รายละเอียดพารามิเตอร์ในสมการ $y = \alpha \cdot \log_a(\beta(x - h)) + c$

    1. **การเลื่อนแนวนอนและเส้นกำกับแนวตั้ง $h$ (Vertical Asymptote):**
       - เส้นตรง **$x = h$** คือเส้นกำกับแนวตั้งที่กราฟจะพุ่งเข้าหาอินฟินิตี้ ($\pm\infty$) แต่ไม่เคยตัดหรือข้ามเส้นนี้
       - โดเมนขึ้นอยู่กับ $h$: หาก $\beta > 0$ โดเมนคือ $(h, +\infty)$ หาก $\beta < 0$ โดเมนคือ $(-\infty, h)$

    2. **สัมประสิทธิ์ $\alpha$ (Vertical Stretch / Reflection):**
       - **$\alpha > 0$:** กราฟมีทิศทางปกติ
       - **$\alpha < 0$:** เกิดการสะท้อนกลับหัวในแนวตั้ง (Vertical Reflection)
       - ยิ่ง $|\alpha|$ มีค่ามาก กราฟยิ่งขยายความชันขึ้นในแนวตั้ง

    3. **ตัวคูณภายใน $\beta$ (Horizontal Stretch / Domain Direction):**
       - **$\beta > 0$:** กราฟอยู่ทาง **ขวา** ของเส้นกำกับ $x = h$
       - **$\beta < 0$:** เกิดการสะท้อนแนวนอน (Horizontal Reflection) กราฟจะย้ายไปอยู่ทาง **ซ้าย** ของเส้นกำกับ $x = h$

    4. **ฐาน $a$ (Base Value):**
       - ต้องเป็นจำนวนจริงบวก และ $a \neq 1$
       - $a > 1$: ฐานมาตรฐาน (เช่น $\log_{10}$ หรือ $\ln$)
       - $0 < a < 1$: สลับทิศทางความชันของกราฟ

    5. **การเลื่อนแนวตั้ง $c$ (Vertical Shift):**
       - ย้ายกราฟขึ้น ($c > 0$) หรือลง ($c < 0$) ในแนวตั้ง
    """)
