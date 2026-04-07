import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Linear Regression Explorer", layout="wide")
st.title("Linear Regression — Interactive Visualizer")

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Dataset")
    n_points     = st.slider("Data Points",   20, 100, 50)
    noise_level  = st.slider("Noise Level",   0.0, 3.0, 1.0, step=0.1)
    add_outliers = st.checkbox("Add Outliers")
    if add_outliers:
        n_outliers = st.slider("Number of Outliers", 1, 10, 3)
    else:
        n_outliers = 0

    st.header("Line Parameters")
    m = st.slider("Slope (m)",      -5.0, 5.0,   1.0, step=0.1)
    b = st.slider("Intercept (b)", -10.0, 10.0,  0.0, step=0.1)

    st.header("Gradient Descent")
    lr     = st.slider("Learning Rate (α)", 0.001, 0.3,  0.01, step=0.001, format="%.3f")
    n_iter = st.slider("Iterations",          10,  300,  100)

# ── Helpers ───────────────────────────────────────────────────────────────────
def generate_data(n, noise, n_out, seed=42):
    np.random.seed(seed)
    X = np.linspace(-5, 5, n)
    y = 2.0 * X + 1.0 + np.random.normal(0, noise, n)
    if n_out > 0:
        idx = np.random.choice(n, n_out, replace=False)
        y[idx] += np.random.choice([-1, 1], n_out) * np.random.uniform(8, 15, n_out)
    return X, y

def compute_mse(X, y, slope, intercept):
    return float(np.mean((y - (slope * X + intercept)) ** 2))

def run_gd(X, y, m0, b0, learning_rate, iterations):
    m_cur, b_cur = m0, b0
    n = len(X)
    ms, bs, losses = [m_cur], [b_cur], [compute_mse(X, y, m_cur, b_cur)]
    for _ in range(iterations):
        y_pred = m_cur * X + b_cur
        dm = -2 / n * np.sum(X * (y - y_pred))
        db = -2 / n * np.sum(y - y_pred)
        m_cur -= learning_rate * dm
        b_cur -= learning_rate * db
        ms.append(m_cur)
        bs.append(b_cur)
        losses.append(compute_mse(X, y, m_cur, b_cur))
    return np.array(ms), np.array(bs), np.array(losses)

# Generate data once for this render
X, y = generate_data(n_points, noise_level, n_outliers)
current_mse = compute_mse(X, y, m, b)

# Pre-compute loss landscape grid (shared by Tab 3 & 4)
m_grid = np.linspace(-5, 5, 40)
b_grid = np.linspace(-10, 10, 40)
M, B = np.meshgrid(m_grid, b_grid)
Z = np.array([[compute_mse(X, y, mi, bi) for mi in m_grid] for bi in b_grid])

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data & Line Fit",
    "Error Visualization",
    "Loss Landscape",
    "Gradient Descent",
    "Learning Rate Experiments",
    "Noise & Robustness",
])

# ── Tab 1: Data & Line Fit ────────────────────────────────────────────────────
with tab1:
    st.info("Move the **Slope (m)** and **Intercept (b)** sliders in the sidebar to fit the line to the data. Watch the MSE drop as you get closer to the true relationship (y = 2x + 1)!")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color="steelblue", alpha=0.7, label="Data points")
    ax.plot(X, m * X + b, color="red", linewidth=2, label=f"y = {m:.1f}x + {b:.1f}")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Data & Fitted Line")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.metric("Mean Squared Error (MSE)", f"{current_mse:.3f}")

# ── Tab 2: Error Visualization ────────────────────────────────────────────────
with tab2:
    st.info("The **red dashed lines** are residuals — the vertical distance between each data point and the model's prediction. MSE averages the squares of all these distances.")

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pred = m * X + b
    ax.scatter(X, y, color="steelblue", alpha=0.7, zorder=3, label="Data points")
    ax.plot(X, y_pred, color="red", linewidth=2, label="Fitted line")
    for xi, yi, yp in zip(X, y, y_pred):
        ax.plot([xi, xi], [yi, yp], color="tomato", linewidth=1, linestyle="--")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Residuals (Errors)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.metric("MSE  =  mean( (y − ŷ)² )", f"{current_mse:.3f}")

# ── Tab 3: Loss Landscape ─────────────────────────────────────────────────────
with tab3:
    st.info("This contour map shows MSE for every combination of slope and intercept. The **red dot** marks your current (m, b). Move the sliders — the dot follows! The darker the region, the lower the loss.")

    fig, ax = plt.subplots(figsize=(7, 5))
    cp = ax.contourf(M, B, Z, levels=30, cmap="viridis")
    plt.colorbar(cp, ax=ax, label="MSE")
    ax.scatter([m], [b], color="red", s=120, zorder=5, label=f"Current (m={m:.1f}, b={b:.1f})")
    ax.set_xlabel("Slope (m)")
    ax.set_ylabel("Intercept (b)")
    ax.set_title("Loss Landscape  J(m, b)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 4: Gradient Descent ───────────────────────────────────────────────────
with tab4:
    st.info("Gradient Descent starts from your chosen **(m, b)** and iteratively steps downhill on the loss surface. The white path shows how the model learns, and the chart shows loss decreasing over iterations.")

    ms_hist, bs_hist, losses_hist = run_gd(X, y, m, b, lr, n_iter)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.contourf(M, B, Z, levels=30, cmap="viridis", alpha=0.8)
        ax.plot(ms_hist, bs_hist, color="white", linewidth=1.5, alpha=0.9, label="GD path")
        ax.scatter([ms_hist[0]], [bs_hist[0]], color="yellow", s=80, zorder=5, label="Start")
        ax.scatter([ms_hist[-1]], [bs_hist[-1]], color="red", s=80, zorder=5, label="End")
        ax.set_xlabel("Slope (m)")
        ax.set_ylabel("Intercept (b)")
        ax.set_title("Gradient Descent Trajectory")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(losses_hist, color="steelblue", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE")
        ax.set_title("Loss over Iterations")
        st.pyplot(fig)
        plt.close(fig)

    st.metric("Final MSE after GD", f"{losses_hist[-1]:.3f}",
              delta=f"{losses_hist[-1] - losses_hist[0]:.3f}")
    st.caption(f"Converged to  m = {ms_hist[-1]:.3f},  b = {bs_hist[-1]:.3f}")

# ── Tab 5: Learning Rate Experiments ─────────────────────────────────────────
with tab5:
    st.info("Compare how different learning rates affect convergence. **Too small** = very slow. **Too large** = unstable or diverges. The right rate converges smoothly and quickly.")

    rates  = [0.001, 0.01, 0.05, 0.2]
    colors = ["steelblue", "green", "orange", "red"]
    labels = ["0.001 — too slow", "0.01 — good", "0.05 — fast", "0.2 — risky"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for rate, color, label in zip(rates, colors, labels):
        _, _, losses_r = run_gd(X, y, 0.0, 0.0, rate, n_iter)
        # Clip extreme spikes so diverging runs don't collapse the y-axis
        clipped = np.clip(losses_r, 0, losses_r[0] * 2)
        ax.plot(clipped, color=color, linewidth=2, label=f"α = {label}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.set_title("Loss vs Iterations — Different Learning Rates")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 6: Noise & Robustness ─────────────────────────────────────────────────
with tab6:
    st.info("Linear Regression is sensitive to **outliers** because squaring errors amplifies their effect. Compare how the fitted line shifts when outliers are introduced.")

    X_clean, y_clean = generate_data(n_points, noise=0.5, n_out=0)
    X_noisy, y_noisy = generate_data(n_points, noise_level, n_out=max(n_outliers, 3))

    m_clean, b_clean = np.polyfit(X_clean, y_clean, 1)
    m_noisy, b_noisy = np.polyfit(X_noisy, y_noisy, 1)

    mse_clean = compute_mse(X_clean, y_clean, m_clean, b_clean)
    mse_noisy = compute_mse(X_noisy, y_noisy, m_noisy, b_noisy)

    # Identify which points are outliers (same seed/logic as generate_data)
    np.random.seed(42)
    outlier_idx = set(np.random.choice(n_points, max(n_outliers, 3), replace=False))
    point_colors = ["red" if i in outlier_idx else "steelblue" for i in range(n_points)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Clean Data")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(X_clean, y_clean, color="steelblue", alpha=0.7)
        ax.plot(X_clean, m_clean * X_clean + b_clean, color="green", linewidth=2,
                label=f"y = {m_clean:.2f}x + {b_clean:.2f}")
        ax.set_title("Without Outliers")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
        st.metric("MSE (clean)", f"{mse_clean:.3f}")

    with col2:
        st.subheader("Noisy Data with Outliers")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(X_noisy, y_noisy, color=point_colors, alpha=0.7)
        ax.plot(X_noisy, m_noisy * X_noisy + b_noisy, color="red", linewidth=2,
                label=f"y = {m_noisy:.2f}x + {b_noisy:.2f}")
        ax.set_title("With Outliers (red points)")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
        st.metric("MSE (with outliers)", f"{mse_noisy:.3f}")
