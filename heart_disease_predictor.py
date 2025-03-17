# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 加载训练好的XGBoost模型
model = joblib.load('XGBoost.pkl')

# 设置页面标题和布局
st.set_page_config(page_title="设备状态预测系统", layout="wide")
st.title("🏭 设备运行状态预测系统")

# 侧边栏输入参数（调整step和显示格式）
with st.sidebar:
    st.header("⚙️ 设备运行参数输入")
    
    yaw_position = st.number_input(
        "偏航角度 (yaw_position)",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.001,
        format="%.3f",
        help="设备偏航角度，范围[-3.0, 3.0]"
    )
    
    environment_tmp = st.number_input(
        "环境温度 (environment_tmp)",
        min_value=-5.0,
        max_value=50.0,
        value=25.0,
        step=0.001,
        format="%.3f",
        help="环境温度，单位℃"
    )
    
    power = st.number_input(
        "输出功率 (power)",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.001,
        format="%.3f",
        help="设备输出功率，标准化后的值"
    )
    
    wind_speed = st.number_input(
        "风速 (wind_speed)",
        min_value=-2.0,
        max_value=20.0,
        value=5.0,
        step=0.001,
        format="%.3f",
        help="实时风速，标准化后的值"
    )
    
    pitch3_angle = st.number_input(
        "桨叶3角度 (pitch3_angle)",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.001,
        format="%.3f",
        help="第三个桨叶的倾斜角度"
    )
    
    int_tmp = st.number_input(
        "内部温度 (int_tmp)",
        min_value=-5.0,
        max_value=80.0,
        value=45.0,
        step=0.001,
        format="%.3f",
        help="设备内部温度，单位℃"
    )
    
    pitch2_angle = st.number_input(
        "桨叶2角度 (pitch2_angle)",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.001,
        format="%.3f",
        help="第二个桨叶的倾斜角度"
    )

# 预测按钮
if st.button("开始预测"):
    # 构建特征数组
    features = np.array([[yaw_position, environment_tmp, power, 
                        wind_speed, pitch3_angle, int_tmp, pitch2_angle]])
    
    try:
        # 预测
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        # 显示结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 预测结果")
            status = "正常" if prediction == 0 else "异常"
            color = "green" if prediction == 0 else "red"
            st.markdown(f"<h2 style='color:{color};'>{status}</h2>", 
                       unsafe_allow_html=True)
            
            st.write(f"正常状态概率: {proba[0]:.3%}")  # 改为3位小数
            st.write(f"异常状态概率: {proba[1]:.3%}")  # 改为3位小数
        
        with col2:
            st.subheader("📈 概率分布")
            fig, ax = plt.subplots(figsize=(8, 3))
            bars = ax.barh(['Normal', 'malfunction'], 
                          [proba[0], proba[1]], 
                          color=['#4CAF50', '#FF5252'])
            
            # 美化图表
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # 添加数值标签（改为3位小数）
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.02, 
                       bar.get_y() + bar.get_height()/2,
                       f'{width:.3%}',  # 改为3位小数
                       va='center',
                       fontsize=12)
            
            st.pyplot(fig)
        
        # 显示建议
        st.subheader("🔧 维护建议")
        if prediction == 0:
            st.success("""
            ✅ 设备运行状态正常！建议：
            1. 继续保持当前运行参数
            2. 定期检查润滑系统
            3. 监控温度变化趋势
            """)
        else:
            st.error("""
            ⚠️ 检测到异常状态！建议：
            1. 立即执行全面诊断检查
            2. 检查桨叶角度控制系统
            3. 监控功率输出稳定性
            4. 联系维护工程师进行现场检查
            """)
            
    except Exception as e:
        st.error(f"预测出错：{str(e)}")

# 添加使用说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    **操作指南：**
    1. 在左侧边栏输入设备实时参数
    2. 点击【开始预测】按钮
    3. 查看预测结果和维护建议
    
    **参数说明：**
    - 偏航角度：设备朝向与风向的夹角（支持3位小数）
    - 环境温度：设备所在环境的温度（支持3位小数）
    - 输出功率：标准化后的功率输出值（支持3位小数）
    - 风速：标准化后的实时风速（支持3位小数）
    - 桨叶角度：各桨叶的实时倾斜角度（支持3位小数）
    - 内部温度：设备核心部件温度（支持3位小数）
    
    **预测结果解释：**
    - 正常（绿色）：设备运行在安全参数范围内
    - 异常（红色）：检测到潜在故障风险
    """)

# 添加页脚
st.markdown("---")
st.markdown("🔍 预测模型基于Transformer-LSTM法训练，版本：1.0.0 | © 2025 风电机组覆冰监测系统")
