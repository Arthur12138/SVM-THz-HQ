import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心逻辑函数
# ==========================================

def read_txt_from_buffer(uploaded_file):
    """从 Streamlit 上传的文件流中读取数据"""
    single_file_data = []
    # 将上传的字节流转为字符串列表
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    lines = content.splitlines()

    start_index = -1
    for i, line in enumerate(lines):
        if "Calculated value" in line:
            start_index = i + 1
            break

    if start_index != -1:
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    single_file_data.append([x, y])
                except ValueError: continue
    return np.array(single_file_data)

def perform_fft_complex(data):
    """执行FFT"""
    t = data[:, 0]
    signal = data[:, 1]
    n = len(t)
    dt = t[1] - t[0]
    fft_complex = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, fft_complex

# ==========================================
# 2. 页面配置与模型加载
# ==========================================

st.set_page_config(page_title="太赫兹中医药AI鉴定", layout="wide")

@st.cache_resource # 缓存模型，避免每次点击都重新加载
def load_model():
    return joblib.load('thz_svm_model.pkl')

try:
    model = load_model()
    labels = {0: "甘肃黄芪", 1: "内蒙黄芪", 2: "云南黄芪"}
except Exception as e:
    st.error(f"模型加载失败，请确保目录下有 thz_svm_model.pkl 文件。错误: {e}")
    st.stop()

# ==========================================
# 3. 侧边栏及文件上传
# ==========================================

st.sidebar.title("📁 数据上传")
air_file = st.sidebar.file_uploader("上传 AIR 背景文件 (.txt)", type=["txt"])
sample_file = st.sidebar.file_uploader("上传 样品检测文件 (.txt)", type=["txt"])

# ==========================================
# 4. 主界面设计
# ==========================================

st.title("🌿 太赫兹中医药产地快速识别系统")
st.markdown("---")

if air_file and sample_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 信号预览")
        # 处理数据
        air_raw = read_txt_from_buffer(air_file)
        sample_raw = read_txt_from_buffer(sample_file)
        
        # 绘图预览
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(air_raw[:, 0], air_raw[:, 1], label="AIR", alpha=0.7)
        ax.plot(sample_raw[:, 0], sample_raw[:, 1], label="Sample", alpha=0.7)
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("🧪 分析与识别")
        if st.button("开始 AI 智能识别", type="primary"):
            with st.spinner('正在执行 FFT 变换及特征提取...'):
                try:
                    # 1. FFT
                    _, air_fft = perform_fft_complex(air_raw)
                    _, sample_fft = perform_fft_complex(sample_raw)
                    
                    # 2. 计算比值
                    ratio = sample_fft[:91] / air_fft[:91]
                    
                    # 3. 特征构造 (实部+虚部)
#                    features = np.hstack([ratio.real, ratio.imag]).reshape(1, -1)
                    features = np.abs(ratio).reshape(1,-1)                  
                    # 4. 预测
                    pred_idx = model.predict(features)[0]
                    confidence = model.predict_proba(features).max() if hasattr(model, "predict_proba") else None
                    
                    # 5. 展示结果
                    st.success(f"### 识别结果：{labels[pred_idx]}")
                    if confidence:
                        st.info(f"**可靠度预测：{confidence:.2%}**")
                    
                    # 绘制比值的幅值图
                    st.write("**太赫兹传递函数 (幅值):**")
                    fig2, ax2 = plt.subplots(figsize=(8, 3))
                    ax2.plot(np.abs(ratio))
                    st.pyplot(fig2)
                    
                except Exception as e:
                    st.error(f"分析出错: {e}")
else:
    st.info("请在左侧边栏上传相应的 AIR 和 Sample 数据文件以开始鉴定。")
    st.warning("注意：TXT 文件必须包含 'Calculated value' 标记。")
