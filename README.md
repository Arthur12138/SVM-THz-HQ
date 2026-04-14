# THz Web App

基于 Streamlit 的太赫兹中药材智能识别系统。

## 功能
- 当归：产地分类、年份（待开发）
- 黄芪：产地分类、年份（待开发）
- 人参：产地（待开发）、真假（待开发）
- 酸枣仁：产地（待开发）、真假（待开发）、年份（待开发）

## 当前模型
- 当归产地模型：`models/thz_svm_model_dg.pkl`
- 黄芪产地模型：`models/thz_svm_model_hq.pkl`

## 运行方法
```bash
pip install -r requirements.txt
streamlit run app.py
