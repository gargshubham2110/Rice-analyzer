import cv2
import numpy as np
import streamlit as st

from analyzer import Params, analyze, load_image

st.set_page_config(page_title="Rice Grain Analyzer", page_icon="🌾", layout="wide")
st.title("🌾 Rice Grain Analyzer")

with st.sidebar:
    st.header("Setup")
    ref_d = st.number_input("Reference coin diameter (mm)", 5.0, 60.0, 25.0, 0.1,
                            help="Measure your coin with a vernier caliper.")
    manual = st.number_input("Manual scale mm/pixel (0 = use coin)", 0.0, 1.0, 0.0, 0.001, format="%.4f")
    st.header("Tuning")
    sens = st.slider("Segmentation sensitivity", 0.5, 1.5, 1.0, 0.05)
    broken = st.slider("Broken if shorter than (x whole-grain length)", 0.4, 0.95, 0.75, 0.05)
    yellow = st.slider("Yellow threshold (b*)", 5.0, 40.0, 18.0, 0.5)
    dark = st.slider("Dark-spot area fraction", 0.01, 0.30, 0.04, 0.01)
    chalk = st.slider("Chalky-patch area fraction", 0.05, 0.60, 0.15, 0.01)

with st.expander("How to take the photo", expanded=False):
    st.markdown(
        "- Plain **black matte paper** as background (any uniform colour works).\n"
        "- Spread grains so they **do not touch** (a few touching is OK).\n"
        "- Put a **round coin** of known diameter in the frame. Optional: a **white paper card** for colour/whiteness.\n"
        "- Shoot **straight from above**, ~30 cm, soft even light, **no flash**, no shadows.\n"
        "- Use the full-resolution photo (Upload) rather than the in-app camera."
    )

up = st.file_uploader("Take or upload a photo", type=["jpg", "jpeg", "png"])
cam = st.camera_input("...or use the camera (lower resolution)") if up is None else None
src = up or cam

if src:
    try:
        img = load_image(src.getvalue())
        res = analyze(img, Params(ref_d, manual, sens, broken, yellow, dark, chalk))
    except ValueError as e:
        st.error(str(e))
        st.stop()

    s = res.summary
    st.image(cv2.cvtColor(res.annotated, cv2.COLOR_BGR2RGB),
             caption="Green = OK, red = broken, orange = yellow, magenta = dark spots, cyan = chalky",
             use_container_width=True)
    for n in res.notes:
        st.caption("• " + n)

    c = st.columns(4)
    c[0].metric("Grains", s["grains"])
    c[1].metric("Avg length (mm)", f"{s['avg_length_mm']:.2f}", f"± {s['std_length_mm']:.2f}", delta_color="off")
    c[2].metric("Avg width (mm)", f"{s['avg_width_mm']:.2f}")
    c[3].metric("L/W ratio", f"{s['avg_l_w_ratio']:.2f}")
    c = st.columns(4)
    c[0].metric("Whiteness index", f"{s['avg_whiteness']:.1f}")
    c[1].metric("Broken %", f"{s['broken_pct']:.1f}")
    c[2].metric("Yellow / dark %", f"{s['yellow_pct']:.1f} / {s['dark_pct']:.1f}")
    c[3].metric("Chalky %", f"{s['chalky_pct']:.1f}")

    hist, edges = np.histogram(res.table.length_mm, bins=12)
    st.subheader("Length distribution")
    st.bar_chart({f"{edges[i]:.1f}": int(hist[i]) for i in range(len(hist))})

    st.subheader("Per-grain results")
    show = ["id", "length_mm", "width_mm", "l_w_ratio", "whiteness", "defects"]
    st.dataframe(res.table[show], use_container_width=True, hide_index=True)
    st.download_button("Download CSV", res.table.to_csv(index=False), "rice_results.csv", "text/csv")
