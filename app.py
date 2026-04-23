import os
import oss2
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# ==========================================
# 1. 阿里云 OSS 配置与逻辑
# ==========================================
# 建议在生产环境使用 st.secrets 管理密钥
ACCESS_KEY_ID = 'LTAI5t91mUZm7g3dnSscrAti'
ACCESS_KEY_SECRET = 'LaApzAJHqIz9gC5N0y77yUhp6DvSTl'
ENDPOINT = 'https://oss-cn-shanghai.aliyuncs.com' 
BUCKET_NAME = 'duo-chen'

# 初始化 OSS
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

def upload_data_to_cloud(air_file, sample_file):
    """
    将本次上传的两个文件打包上传到云端文件夹
    文件夹命名格式：YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = f"uploads/{timestamp}/"
    
    try:
        # 上传 Air 文件
        air_name = f"{folder_path}AIR_{air_file.name}"
        bucket.put_object(air_name, air_file.getvalue())
        
        # 上传 Sample 文件
        sample_name = f"{folder_path}SAMPLE_{sample_file.name}"
        bucket.put_object(sample_name, sample_file.getvalue())
        
        return True, folder_path
    except Exception as e:
        return False, str(e)

# ==========================================
# 2. 核心算法逻辑函数
# ==========================================

def read_txt_from_buffer(uploaded_file):
    """从 Streamlit 上传的文件流中读取数据"""
    single_file_data = []
    # 使用 getvalue() 确保重复读取不会出问题
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
                except ValueError:
                    continue
    return np.array(single_file_data)

def perform_fft_complex(data):
    """执行FFT"""
    if data is None or len(data) < 2:
        raise ValueError("数据点不足，无法进行FFT")
    t = data[:, 0]
    signal = data[:, 1]
    n = len(t)
    dt = t[1] - t[0]
    fft_complex = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, fft_complex

def build_feature_from_two_files(air_file, sample_file, num_points=91, feature_method="abs"):
    """从二进制流构建特征"""
    air_raw = read_txt_from_buffer(air_file)
    sample_raw = read_txt_from_buffer(sample_file)

    if air_raw.size == 0 or sample_raw.size == 0:
        raise ValueError("文件解析失败，请检查格式。")

    _, air_fft = perform_fft_complex(air_raw)
    _, sample_fft = perform_fft_complex(sample_raw)

    ratio = sample_fft[:num_points] / (air_fft[:num_points] + 1e-12)

    if feature_method == "abs":
        features = np.abs(ratio).reshape(1, -1)
    else:
        # 其他模式简化处理
        features = np.abs(ratio).reshape(1, -1)

    return air_raw, sample_raw, ratio, features

# ==========================================
# 3. 模型配置
# ==========================================

MODEL_CONFIG = {
    "当归": {
        "产地": {"model_path": "models/thz_svm_model_dg.pkl", "label_map": {0: "甘肃", 1: "四川", 2: "云南"}, "available": True},
        "年份": {"model_path": None, "label_map": {}, "available": False}
    },
    "黄芪": {
        "产地": {"model_path": "models/thz_svm_model_hq.pkl", "label_map": {0: "甘肃", 1: "内蒙", 2: "云南"}, "available": True},
        "年份": {"model_path": None, "label_map": {}, "available": False}
    },
    "人参": {"产地": {"available": False}, "真假": {"available": False}},
    "酸枣仁": {"产地": {"available": False}, "真假": {"available": False}}
}

# ==========================================
# 4. 页面 UI
# ==========================================

st.set_page_config(page_title="太赫兹中药材AI鉴定", layout="wide")

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

st.sidebar.title("🌿 系统控制面板")
herb_name = st.sidebar.selectbox("请选择药材", list(MODEL_CONFIG.keys()))
task_options = list(MODEL_CONFIG[herb_name].keys())
task_name = st.sidebar.selectbox("请选择分类标准", task_options)

st.sidebar.markdown("---")
st.sidebar.title("📁 数据上传")
air_file = st.sidebar.file_uploader("上传 AIR 背景文件 (.txt)", type=["txt"])
sample_file = st.sidebar.file_uploader("上传 Sample 检测文件 (.txt)", type=["txt"])

st.title("🌿 太赫兹中药材智能识别系统")
st.markdown("---")

task_cfg = MODEL_CONFIG[herb_name].get(task_name, {"available": False})

if not task_cfg.get("available"):
    st.warning(f"⚠️ 暂无 {herb_name}-{task_name} 的预测模型。")
    st.stop()

# 主界面逻辑
if air_file and sample_file:
    # 准备特征
    try:
        air_raw, sample_raw, ratio, features = build_feature_from_two_files(air_file, sample_file)
    except Exception as e:
        st.error(f"数据解析失败: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 信号预览")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(air_raw[:, 0], air_raw[:, 1], label="AIR")
        ax.plot(sample_raw[:, 0], sample_raw[:, 1], label="Sample")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("🧪 分析与云端同步")
        if st.button("开始 AI 智能识别", type="primary"):
            # --- 第一步：云端同步 ---
            with st.spinner("正在同步数据至云端备份..."):
                success, info = upload_data_to_cloud(air_file, sample_file)
                if success:
                    st.toast(f"✅ 数据已同步至云端文件夹: {info}", icon='☁️')
                else:
                    st.error(f"❌ 云端同步失败: {info}")

            # --- 第二步：AI 识别 ---
            with st.spinner("正在进行智能识别..."):
                try:
                    model = load_model(task_cfg["model_path"])
                    pred_idx = model.predict(features)[0]
                    pred_label = task_cfg["label_map"].get(int(pred_idx), str(pred_idx))
                    
                    st.success(f"### 识别结果：{pred_label}")
                    
                    # 结果展示
                    res_df = pd.DataFrame({
                        "检测时间": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                        "药材": [herb_name],
                        "识别结论": [pred_label],
                        "云端存储路径": [info if success else "未同步"]
                    })
                    st.table(res_df)
                except Exception as e:
                    st.error(f"预测过程出错: {e}")
else:
    st.info("💡 请在左侧上传 AIR 和 Sample 文件。")
