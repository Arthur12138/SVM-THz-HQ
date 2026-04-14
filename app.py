import os
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd


# ==========================================
# 1. 核心逻辑函数
# ==========================================

def read_txt_from_buffer(uploaded_file):
    """从 Streamlit 上传的文件流中读取数据"""
    single_file_data = []
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

    if dt == 0:
        raise ValueError("时间间隔 dt 为 0，无法进行FFT")

    fft_complex = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, fft_complex


def build_feature_from_two_files(air_file, sample_file, num_points=91, feature_method="abs"):
    """
    从 AIR 文件和 Sample 文件构建特征
    """
    air_raw = read_txt_from_buffer(air_file)
    sample_raw = read_txt_from_buffer(sample_file)

    if air_raw.size == 0:
        raise ValueError("AIR 文件中未读取到有效数据，请检查是否包含 'Calculated value' 后的数据")
    if sample_raw.size == 0:
        raise ValueError("Sample 文件中未读取到有效数据，请检查是否包含 'Calculated value' 后的数据")

    _, air_fft = perform_fft_complex(air_raw)
    _, sample_fft = perform_fft_complex(sample_raw)

    if len(air_fft) < num_points or len(sample_fft) < num_points:
        raise ValueError(f"FFT点数不足，无法截取前 {num_points} 个频点")

    ratio = sample_fft[:num_points] / (air_fft[:num_points] + 1e-12)

    if feature_method == "abs":
        features = np.abs(ratio).reshape(1, -1)
    elif feature_method == "real_imag":
        features = np.hstack([ratio.real, ratio.imag]).reshape(1, -1)
    elif feature_method == "abs_phase":
        features = np.hstack([np.abs(ratio), np.angle(ratio)]).reshape(1, -1)
    else:
        raise ValueError(f"不支持的特征方式: {feature_method}")

    return air_raw, sample_raw, ratio, features


# ==========================================
# 2. 药材与模型配置
# ==========================================

MODEL_CONFIG = {
    "当归": {
        "产地": {
            "model_path": "models/thz_svm_model_dg.pkl",
            "label_map": {
                0: "甘肃",
                1: "四川",
                2: "云南"
            },
            "available": True
        },
        "年份": {
            "model_path": None,
            "label_map": {},
            "available": False
        }
    },
    "黄芪": {
        "产地": {
            "model_path": "models/thz_svm_model_hq.pkl",
            "label_map": {
                0: "甘肃",
                1: "内蒙",
                2: "云南"
            },
            "available": True
        },
        "年份": {
            "model_path": None,
            "label_map": {},
            "available": False
        }
    },
    "人参": {
        "产地": {
            "model_path": None,
            "label_map": {},
            "available": False
        },
        "真假": {
            "model_path": None,
            "label_map": {},
            "available": False
        }
    },
    "酸枣仁": {
        "产地": {
            "model_path": None,
            "label_map": {},
            "available": False
        },
        "真假": {
            "model_path": None,
            "label_map": {},
            "available": False
        },
        "年份": {
            "model_path": None,
            "label_map": {},
            "available": False
        }
    }
}


# ==========================================
# 3. 页面配置与模型加载
# ==========================================

st.set_page_config(page_title="太赫兹中药材AI鉴定", layout="wide")


@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    return joblib.load(model_path)


def predict_result(model, features, label_map):
    pred_idx = model.predict(features)[0]
    pred_label = label_map.get(int(pred_idx), str(pred_idx))

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(features)[0]
            confidence = np.max(prob)
        except Exception:
            confidence = None

    return pred_idx, pred_label, confidence


# ==========================================
# 4. 页面侧边栏
# ==========================================

st.sidebar.title("🌿 分类任务选择")

herb_name = st.sidebar.selectbox(
    "请选择药材",
    ["当归", "黄芪", "人参", "酸枣仁"]
)

task_options = list(MODEL_CONFIG[herb_name].keys())
task_name = st.sidebar.selectbox(
    "请选择分类标准",
    task_options
)

st.sidebar.markdown("---")
st.sidebar.title("📁 数据上传")
air_file = st.sidebar.file_uploader("上传 AIR 背景文件 (.txt)", type=["txt"])
sample_file = st.sidebar.file_uploader("上传 Sample 检测文件 (.txt)", type=["txt"])


# ==========================================
# 5. 主界面
# ==========================================

st.title("🌿 太赫兹中药材智能识别系统")
st.markdown("---")

task_cfg = MODEL_CONFIG[herb_name][task_name]

# 显示当前任务信息
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.subheader("当前任务")
    st.write(f"**药材：** {herb_name}")
    st.write(f"**分类标准：** {task_name}")
    st.write(f"**模型状态：** {'已建立' if task_cfg['available'] else '暂无模型'}")

with col_info2:
    st.subheader("分类标准说明")
    if task_cfg["available"] and task_cfg["label_map"]:
        desc_df = pd.DataFrame({
            "类别编码": list(task_cfg["label_map"].keys()),
            "类别名称": list(task_cfg["label_map"].values())
        })
        st.dataframe(desc_df, use_container_width=True)
    else:
        st.info("该任务当前暂无可用模型。")

st.markdown("---")

# 未建立模型时直接提示
if not task_cfg["available"]:
    st.warning(f"{herb_name} 的 {task_name} 分类模型暂未建立。")
    st.stop()

# 加载模型
try:
    model = load_model(task_cfg["model_path"])
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()

if air_file and sample_file:
    try:
        air_raw, sample_raw, ratio, features = build_feature_from_two_files(
            air_file,
            sample_file,
            num_points=91,
            feature_method="abs"
        )
    except Exception as e:
        st.error(f"数据处理失败：{e}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 时域信号预览")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(air_raw[:, 0], air_raw[:, 1], label="AIR", alpha=0.8)
        ax.plot(sample_raw[:, 0], sample_raw[:, 1], label="Sample", alpha=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.subheader("📈 传递函数幅值图")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(np.abs(ratio), color="tab:blue")
        ax2.set_xlabel("Frequency Point Index")
        ax2.set_ylabel("|Sample/AIR|")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

    with col2:
        st.subheader("🧪 分析与识别")

        st.write("**当前特征方式：** 幅值特征 `abs(ratio)`")
        st.write(f"**特征维度：** {features.shape[1]}")

        if st.button("开始 AI 智能识别", type="primary"):
            with st.spinner("正在执行 FFT 变换、特征提取与分类识别..."):
                try:
                    pred_idx, pred_label, confidence = predict_result(
                        model,
                        features,
                        task_cfg["label_map"]
                    )

                    st.success(f"### 识别结果：{pred_label}")

                    if confidence is not None:
                        st.info(f"**可靠度预测：{confidence:.2%}**")

                    result_df = pd.DataFrame({
                        "药材": [herb_name],
                        "分类标准": [task_name],
                        "预测编码": [pred_idx],
                        "预测结果": [pred_label],
                        "AIR文件名": [air_file.name],
                        "Sample文件名": [sample_file.name]
                    })

                    if confidence is not None:
                        result_df["置信度"] = [confidence]

                    st.subheader("结果汇总")
                    st.dataframe(result_df, use_container_width=True)

                    csv_data = result_df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label="下载结果 CSV",
                        data=csv_data,
                        file_name=f"{herb_name}_{task_name}_prediction.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"分析出错: {e}")

else:
    st.info("请在左侧边栏上传 AIR 文件和 Sample 文件以开始鉴定。")
    st.warning("注意：TXT 文件必须包含 'Calculated value' 标记。")
