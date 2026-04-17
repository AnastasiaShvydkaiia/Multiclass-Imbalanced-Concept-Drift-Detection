import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from metrics import MetricsTracker
from drift_detectors import DriftDetector
from stream_generator import create_stream
from classifier import Classifier

# =====================INIT STATE======================
def reset_experiment(drift_type, drift_speed, drift_intensity, imbalance, detector_name):
    """Reset all experiment state to start fresh"""
    st.session_state.step = 0
    st.session_state.steps = []
    st.session_state.errors = []
    st.session_state.f0 = []
    st.session_state.f1 = []
    st.session_state.classes = []
    st.session_state.drift_points = []
    st.session_state.manual_drifts = []
    st.session_state.err_window.clear()
    st.session_state.drift_triggered = False
    st.session_state.drift_start = None
    st.session_state.clf = Classifier()
    st.session_state.detector = DriftDetector(detector_name,n_features=20, n_classes=3)
    st.session_state.stream = create_stream(drift_type, drift_speed, drift_intensity,imbalance, st.session_state.get("manual_drifts", []))
    st.session_state.metrics = MetricsTracker()
    st.session_state.en_acc_history = []
    st.session_state.gmean_history = []
    st.session_state.hdd_history = []
    st.session_state.delay_history = []

if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.running = False
    st.session_state.err_window = deque(maxlen=100)
    reset_experiment("real", "sudden", "global","1:1:1", "ADWIN")  # Initial default state

# =======================UI===========================
st.title("Drift Detection Demo")

col1, col2, col3 = st.columns(3)

drift_type = col1.selectbox("Drift Type", ["real", "virtual"])
drift_speed = col2.selectbox("Drift Speed", ["sudden", "gradual"])
drift_intensity = col3.selectbox("Drift Intensity", ["global", "local"])
imbalance = st.selectbox("Imbalance Ratio", ["1:1:1", "3:2:1", "5:2:1","10:1:1"])
detector_name = st.selectbox("Detector", ["ADWIN", "KSWIN", "DHAE"])

c1, c2, c3 = st.columns(3)

if c1.button("▶ Start"):
    reset_experiment(drift_type, drift_speed, drift_intensity, imbalance, detector_name)
    st.session_state.running = True

if c2.button("⏸ Stop"):
    st.session_state.running = False

if c3.button("💉 Inject Drift"):
    st.session_state.drift_triggered = True
    if "drift_count" not in st.session_state:
        st.session_state.drift_count = 0
    st.session_state.drift_count += 1
    
    st.session_state.drift_start = st.session_state.step
    st.session_state.manual_drifts.append(st.session_state.step)
    print("Drift injected at atep=", st.session_state.step)

# ===================STREAM LOOP==================
window = 50 # Smoothing window
if st.session_state.running and st.session_state.stream:
    try:
        for _ in range(window): 
            x, y, step = next(st.session_state.stream)
            y_pred = st.session_state.clf.predict(x)
            probas= st.session_state.clf.predict_proba(x)
            error = int(y_pred != y)
            st.session_state.err_window.append(error)
            smooth_error = np.mean(st.session_state.err_window)

            # Drift detection
            st.session_state.detector.update(smooth_error, probas, x)
            drift_detected = st.session_state.detector.detected()
        
            # Update metrics
            st.session_state.metrics.update(
                y_true=y,
                y_pred=y_pred,
                drift_detected=drift_detected,
                manual_drifts=st.session_state.manual_drifts 
            )

            metrics_dict = st.session_state.metrics.get_metrics()
            st.session_state.en_acc_history.append(metrics_dict["en_accuracy"])
            st.session_state.gmean_history.append(metrics_dict["g_mean"])

            if drift_detected:
                st.session_state.drift_points.append(step)
                print ("DRIFT DETECTED AT STEP=", step)
                st.session_state.clf= Classifier()

            # Update classifies
            st.session_state.clf.learn(x, y)

            # Save data
            st.session_state.steps.append(step)
            st.session_state.errors.append(smooth_error)
            st.session_state.f0.append(x[0])
            st.session_state.f1.append(x[1])
            st.session_state.classes.append(y)
            st.session_state.step += 1
    except StopIteration:
        st.session_state.running = False 
        st.success("Datastream is over")

# ====================VISUALIZATION========================
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

steps = st.session_state.steps
classes = getattr(st.session_state, "classes", [])

# Smoothed features
if len(st.session_state.f0) > 0:
    f0_smooth = pd.Series(st.session_state.f0).rolling(window, min_periods=1).mean()
    f1_smooth = pd.Series(st.session_state.f1).rolling(window, min_periods=1).mean()
    axes[0].plot(steps, f0_smooth, label="f0 (smooth)", color="purple", alpha=0.8)
    axes[0].plot(steps, f1_smooth, label="f1 (smooth)", color="brown", alpha=0.8)
    
axes[0].set_title("Feature Evolution")
axes[0].legend()

# Error
axes[1].plot(steps, st.session_state.errors, color="black")
axes[1].set_title("Classification Error")

# Class distributioin
class_names = [0, 1, 2]
colors = ["blue", "orange", "green"]

for i, cname in enumerate(class_names):
    presence = [1 if c == cname else 0 for c in classes]
    if len(presence) > 0:
        rolling = pd.Series(presence).rolling(window, min_periods=1).mean()
        axes[2].plot(steps, rolling, label=cname, color=colors[i])

axes[2].set_title("Class Distribution")
axes[2].set_ylabel("Proportion")
axes[2].set_xlabel("Time")
axes[2].set_ylim(0, 1)
axes[2].legend()

# Metrics
axes[3].plot(steps, st.session_state.en_acc_history, label="EN-Acc")
axes[3].plot(steps, st.session_state.gmean_history, label="G-Mean")
axes[3].set_title("Performance Metrics")
axes[3].set_ylabel("Score")
axes[3].set_ylim(0, 1)
axes[3].legend()

# Drift lines
for ax in axes:
    for dp in st.session_state.manual_drifts:
        ax.axvline(dp, color="red", linestyle="--", alpha=0.6, label="Injected" if ax == axes[0] else "")
    
    for dp in st.session_state.drift_points:
        ax.axvline(dp, color="green", linestyle="-", alpha=0.8, label="Detected" if ax == axes[0] else "")

plt.tight_layout()
st.pyplot(fig)

# ==================== STATISTICS SECTION ====================
st.write("---") 
st.subheader("Statistics")

stats = st.session_state.metrics.get_metrics()

# False alarm rate
total_alarms = stats['drifts_detected'] + stats.get('false_alarms', 0)
far = (stats.get('false_alarms', 0) / total_alarms) if total_alarms > 0 else 0.0

col_det1, col_det2, col_det3 = st.columns(3)
st.write("") 
col_mod1, col_mod2 = st.columns(2)

with col_det1:
    st.metric(
        label="Harmonic Drift Detection", 
        value=f"{stats['h_dd']:.3f}",
        help="Measures the overall quality of the drift detector in distinguishing between windows with and without drift"
    )

with col_det2:
    st.metric(
        label="Avg Detection Delay", 
        value=f"{stats['avg_detection_delay']:.1f} steps",
        help="Average number of steps between detection and drift"
    )

with col_det3:
    st.metric(
        label="False Alarm Rate (FAR)", 
        value=f"{far:.1%}",
        help="A proportion of false alarms from the total number of detector signals"
    )

with col_mod1:
    st.metric(
        label="G-Mean", 
        value=f"{stats['g_mean']:.3f}",
        help="Measures how effectively the system maintains classification quality, particularly for minority classes"
    )

with col_mod2:
    st.metric(
        label="Accuracy", 
        value=f"{stats['en_accuracy']:.2%}",
        help="A proportion of correctly identified drifted windows"
    )

# ================LOOP REFRESH======================
if st.session_state.running:
    time.sleep(1)
    st.rerun()


