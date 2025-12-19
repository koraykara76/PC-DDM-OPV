
"""
PC-DDM: Physically Constrained Double-Diode Modeling for OPV / Perovskite Devices
-------------------------------------------------------------------------------

This script performs current–voltage (J–V) analysis of organic and perovskite
solar cells using a physically constrained double-diode model (DDM).

Main features:
- Reads simple two-column .dat files containing voltage (V) and current density (J).
- Extracts key photovoltaic parameters (Isc, Voc, MPP).
- Computes an initial parameter set using a classical analytical double-diode
  initialization.
- Applies a physically constrained nonlinear least-squares (PC-DDM) refinement
  to simultaneously optimize all model parameters.
- Compares measured J–V data, classical DDM initialization, and refined PC-DDM fit.
- Outputs fitted parameters to the console and visualizes results graphically.

Key characteristics:
- No Excel or file-based result export (console output and plots only).
- No device area input required.
- No dependence on Keithley-specific file formats.
- Designed for reproducible, transparent, and publication-ready analysis.

Model parameters:
- IL   : Photogenerated current density (mA/cm^2)
- I01  : Saturation current density of diode 1 (mA/cm^2)
- I02  : Saturation current density of diode 2 (mA/cm^2)
- Rs   : Series resistance (Ohm·cm^2)
- Rsh  : Shunt resistance (Ohm·cm^2)
- n1   : Ideality factor of diode 1
- n2   : Ideality factor of diode 2

Intended use:
This code is provided for research and educational purposes, particularly for
device physics analysis of emerging photovoltaic technologies.

Author: Koray Kara
"""


import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.optimize import least_squares
import os

# ---------------------------------------------------
# Physical constants
# ---------------------------------------------------
k_B = 1.380649e-23     # J/K
q   = 1.602176634e-19  # C

T_DEFAULT  = 298.0     # K (around 25 °C)
NS_DEFAULT = 1         # Single cell / OPV



# ---------------------------------------------------
# Simple two-column IV data loader (.dat / .txt / .csv)
# ---------------------------------------------------
def load_iv_two_column(filename):
    print(f"Selected file: {filename}")

    V_list = []
    I_list = []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("%"):
                continue

            # Replace decimal comma with dot
            line = line.replace(",", ".")

            # First split by TAB, otherwise by whitespace
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue

            try:
                v = float(parts[0].strip())
                i = float(parts[1].strip())
            except ValueError:
                continue

            V_list.append(v)
            I_list.append(i)

    if len(V_list) < 5:
        raise ValueError("Insufficient data points. Check file format.")

    V = np.array(V_list, dtype=float)
    I = np.array(I_list, dtype=float)

    # Sort by voltage
    idx = np.argsort(V)
    V = V[idx]
    I = I[idx]

    print(f"Total points read: {len(V)}")
    print("First 5 points (V, I):")
    for k in range(min(5, len(V))):
        print(f"{V[k]: .5e}\t{I[k]: .5e}")

    # Flip sign if short-circuit current is negative
    idx0 = np.argmin(np.abs(V))
    if I[idx0] < 0:
        I = -I
        print("Note: Jsc was negative; current sign flipped for fitting.")

    return V, I




# ---------------------------------------------------
# General loader
# ---------------------------------------------------
def load_iv_plaintext():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("Please select IV data file (.dat)...")
    filename = filedialog.askopenfilename(
        title="Select IV file",
        filetypes=[("DAT files", "*.dat"), ("All files", "*.*")]
    )

    if not filename:
        raise FileNotFoundError("No file selected.")

    # ONLY simple two-column .dat files
    V, I = load_iv_two_column(filename)

    return V, I, filename



# ---------------------------------------------------
# Keypoint extraction (Isc, Voc, Imp, Vmp)
# ---------------------------------------------------
def extract_keypoints(V, I):
    # Isc: point closest to V = 0
    idx0 = np.argmin(np.abs(V))
    Isc = I[idx0]

    # Voc: minimum |I| in positive voltage region
    mask_voc = V >= 0
    if np.any(mask_voc):
        idx_voc_local = np.argmin(np.abs(I[mask_voc]))
        Voc = V[mask_voc][idx_voc_local]
    else:
        idx_voc = np.argmin(np.abs(I))
        Voc = V[idx_voc]

    # MPP: maximum power in V > 0 and I > 0 region
    mask_mpp = (V > 0) & (I > 0)
    if not np.any(mask_mpp):
        mask_mpp = V > 0

    P = V[mask_mpp] * I[mask_mpp]
    idx_mpp = np.argmax(P)
    Vmp = V[mask_mpp][idx_mpp]
    Imp = I[mask_mpp][idx_mpp]

    return Isc, Voc, Imp, Vmp




# ---------------------------------------------------
# Double-diode current calculation (Newton–Raphson)
# ---------------------------------------------------
def ddm_current_at_voltage(V, params, T, Ns, I_init=None,
                           max_iter=60, tol=1e-9):
    """
    params = (IL, I01, I02, Rs, Rsh, n1, n2)
    """
    IL, I01, I02, Rs, Rsh, n1, n2 = params
    VT = k_B * T * Ns / q

    I_guess = IL if I_init is None else I_init

    for _ in range(max_iter):
        Vd = V + I_guess * Rs
        arg1 = Vd / (n1 * VT)
        arg2 = Vd / (n2 * VT)

        exp1 = np.exp(np.clip(arg1, -100, 100))
        exp2 = np.exp(np.clip(arg2, -100, 100))

        f = (IL
             - I01 * (exp1 - 1.0)
             - I02 * (exp2 - 1.0)
             - Vd / Rsh
             - I_guess)

        df_dI = (-I01 * exp1 * (Rs / (n1 * VT))
                 - I02 * exp2 * (Rs / (n2 * VT))
                 - Rs / Rsh
                 - 1.0)

        if np.abs(df_dI) < 1e-20:
            break

        step = f / df_dI
        I_new = I_guess - step

        if np.abs(step) < tol:
            I_guess = I_new
            break

        I_guess = I_new

    return I_guess


def simulate_ddm_curve(V, params, T, Ns, I_meas=None):
    I_sim = np.zeros_like(V)
    for i, v in enumerate(V):
        init = I_meas[i] if I_meas is not None else None
        I_sim[i] = ddm_current_at_voltage(v, params, T, Ns, I_init=init)
    return I_sim
# ---------------------------------------------------
# Senturk–style analytical / recursive DDM parameter extraction
# (n1, n2 not fixed; X, Y tuning)
# ---------------------------------------------------
def compute_ddm_ali(Isc, Voc, Imp, Vmp, T, Ns,
                    X=1.0, Y=1.0,
                    n1=1.0, n2=2.0,
                    n_runs=2):
    VT = k_B * T * Ns / q

    # Initial Rs0, Rsh0 (tuning via X, Y)
    Rs0 = (Voc - Vmp) / (X * Imp)
    Rsh0 = (Y * Vmp) / (Isc - Imp)

    Rs = Rs0
    Rsh = Rsh0

    for _ in range(n_runs):
        # IL
        IL = Isc * (1.0 + Rs / Rsh)

        # Abbreviations
        B = np.exp(Voc / (n1 * VT)) - 1.0
        J = np.exp(Voc / (n2 * VT)) - 1.0
        K = IL - Voc / Rsh

        Vd_mp = Vmp + Imp * Rs
        M = np.exp(Vd_mp / (n1 * VT)) - 1.0
        N = np.exp(Vd_mp / (n2 * VT)) - 1.0

        G = IL - Imp - Vd_mp / Rsh

        Det = B * N - J * M
        if np.abs(Det) < 1e-30:
            raise ValueError("Determinant too small; cannot compute I01 / I02.")

        I01 = (K * N - J * G) / Det
        I02 = (B * G - K * M) / Det

        # Update Rs
        denom = (I01 * np.exp(Voc / (n1 * VT)) / (n1 * VT)
                 + I02 * np.exp(Voc / (n2 * VT)) / (n2 * VT))
        if np.abs(denom) < 1e-30:
            raise ValueError("Denominator too small for Rs update.")
        Rs = Rs0 - 1.0 / denom

        # Update Rsh
        Rsh = Voc / (IL
                     - I01 * (np.exp(Voc / (n1 * VT)) - 1.0)
                     - I02 * (np.exp(Voc / (n2 * VT)) - 1.0))

        Rs0 = Rs
        Rsh0 = Rsh

    return (IL, I01, I02, Rs, Rsh, n1, n2)


# ---------------------------------------------------
# DDM fitting (Classical DDM initialization + n1 / n2 / X / Y scan)
# ---------------------------------------------------
def cost_full_with_knee_weight(V, I_sim, I_meas, Voc,
                               w_knee=3.0, w_tail=1.0):
    mask1 = (V >= 0) & (V <= Voc + 1e-3)
    mask2 = V > Voc

    E1 = I_sim[mask1] - I_meas[mask1]
    E2 = I_sim[mask2] - I_meas[mask2]

    rmse1 = np.sqrt(np.mean(E1 ** 2)) if len(E1) > 0 else 0.0
    rmse2 = np.sqrt(np.mean(E2 ** 2)) if len(E2) > 0 else 0.0

    rng = np.max(I_meas) - np.min(I_meas) + 1e-30
    return (w_knee * rmse1 + w_tail * rmse2) / rng


def fit_ddm_ali_with_tuning(V, I, T=T_DEFAULT, Ns=NS_DEFAULT):
    Isc, Voc, Imp, Vmp = extract_keypoints(V, I)
    print(f"\nMeasured keypoints: "
          f"Isc={Isc: .4e}, Voc={Voc: .4e}, "
          f"Imp={Imp: .4e}, Vmp={Vmp: .4e}")

    n1_values = [1.0, 1.2, 1.4, 1.6]
    n2_values = [2.0, 2.4, 2.8, 3.2]
    X_values  = np.arange(0.8, 1.21, 0.05)
    Y_values  = [0.5, 1.0, 1.5, 2.0]

    best_cost   = np.inf
    best_params = None
    best_hyper  = None

    for n1 in n1_values:
        for n2 in n2_values:
            for X in X_values:
                for Y in Y_values:
                    try:
                        params = compute_ddm_ali(
                            Isc, Voc, Imp, Vmp,
                            T, Ns,
                            X=X, Y=Y,
                            n1=n1, n2=n2,
                            n_runs=2
                        )

                        IL, I01, I02, Rs, Rsh, _, _ = params

                        if (IL <= 0) or (I01 <= 0) or (I02 <= 0) or (Rs <= 0) or (Rsh <= 0):
                            continue

                        I_sim = simulate_ddm_curve(V, params, T, Ns, I_meas=I)
                        cost = cost_full_with_knee_weight(V, I_sim, I, Voc,
                                                          w_knee=3.0,
                                                          w_tail=1.0)

                        if cost < best_cost:
                            best_cost   = cost
                            best_params = params
                            best_hyper  = (n1, n2, X, Y)

                    except Exception:
                        continue

    # Fallback: simple DDM initialization if Classical DDM tuning fails
    if best_params is None:
        print("\nWarning: No valid solution found via Classical DDM tuning.")
        print("         Using simple DDM initialization (fallback).")

        VT = k_B * T * Ns / q

        n1_f = 1.2
        n2_f = 2.0

        Rs_f = max((Voc / (Imp + 1e-9)) * 0.1, 1e-4)

        denom_rsh = (Isc - Imp)
        if abs(denom_rsh) < 1e-9:
            Rsh_f = 1e3
        else:
            Rsh_f = max(Voc / denom_rsh, 1e-3)

        IL_f = max(Isc * (1.0 + Rs_f / Rsh_f), 1e-6)

        try:
            I01_f = (IL_f - Voc / Rsh_f) / (np.exp(Voc / (n1_f * VT)) - 1.0)
        except OverflowError:
            I01_f = 1e-12

        if (not np.isfinite(I01_f)) or (I01_f <= 0):
            I01_f = 1e-12

        I02_f = 1e-14

        best_params = (IL_f, I01_f, I02_f, Rs_f, Rsh_f, n1_f, n2_f)
        best_cost   = np.nan

        return best_params, best_cost, (Isc, Voc, Imp, Vmp)

    n1_best, n2_best, X_best, Y_best = best_hyper
    print(f"\nBest tuning parameters (2-diode DDM, DDM initialization):")
    print(f"  n1 = {n1_best:.2f}, n2 = {n2_best:.2f}")
    print(f"  X  = {X_best:.3f}, Y  = {Y_best:.3f}")
    print(f"NRMSE (DDM initial, 0..Vmax, knee-weighted): {best_cost:.4e}")

    IL, I01, I02, Rs, Rsh, _, _ = best_params
    best_params = (IL, I01, I02, Rs, Rsh, n1_best, n2_best)

    return best_params, best_cost, (Isc, Voc, Imp, Vmp)


# ---------------------------------------------------
# Parameter vector <-> DDM parameters
# ---------------------------------------------------
def params_to_vec(params):
    IL, I01, I02, Rs, Rsh, n1, n2 = params
    return np.array([
        np.log(IL),
        np.log(I01),
        np.log(I02),
        np.log(Rs),
        np.log(Rsh),
        n1,
        n2
    ], dtype=float)


def vec_to_params(p_vec):
    IL  = np.exp(p_vec[0])
    I01 = np.exp(p_vec[1])
    I02 = np.exp(p_vec[2])
    Rs  = np.exp(p_vec[3])
    Rsh = np.exp(p_vec[4])
    n1  = p_vec[5]
    n2  = p_vec[6]
    return (IL, I01, I02, Rs, Rsh, n1, n2)


# ---------------------------------------------------
# Full DDM refinement (least_squares, 7 parameters)
# ---------------------------------------------------

def refine_ddm_with_least_squares(V, I_meas, params_ini, keypoints,
                                  T=T_DEFAULT, Ns=NS_DEFAULT):
    Isc, Voc, Imp, Vmp = keypoints

    p0 = params_to_vec(params_ini)

    lower = np.array([
        np.log(1e-12 * abs(Isc) + 1e-18),
        np.log(1e-20),
        np.log(1e-20),
        np.log(1e-4),
        np.log(1e-3),
        0.7,
        1.2
    ])
    upper = np.array([
        np.log(1e3 * abs(Isc) + 1e-18),
        np.log(1e-1),
        np.log(1e-1),
        np.log(1e3),
        np.log(1e6),
        2.4,
        5.2
    ])

    def residuals(p_vec):
        p_vec = np.copy(p_vec)
        p_vec[5] = np.clip(p_vec[5], 0.7, 3.0)
        p_vec[6] = np.clip(p_vec[6], 1.2, 6.0)

        params = vec_to_params(p_vec)
        I_sim = simulate_ddm_curve(V, params, T, Ns, I_meas=I_meas)

        Vmax = np.max(V)
        z = (V - 0.7 * Voc) / (Vmax - 0.7 * Voc + 1e-9)
        w = 1.0 + 4.0 * np.clip(z, 0.0, 1.0)

        return (I_sim - I_meas) * w

    res = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        max_nfev=200,
    )

    p_best = res.x
    params_best = vec_to_params(p_best)

    I_sim_best = simulate_ddm_curve(V, params_best, T, Ns, I_meas=I_meas)
    rng = np.max(I_meas) - np.min(I_meas) + 1e-30
    nrmse = np.sqrt(np.mean((I_sim_best - I_meas) ** 2)) / rng

    return params_best, I_sim_best, nrmse


# ---------------------------------------------------
# Main script
# ---------------------------------------------------
if __name__ == "__main__":
   
    # 1) Load IV data
    V, I, fname = load_iv_plaintext()

    # 2) Initial parameters via Classical DDM tuning
    params_ini, cost_ini, keypoints = fit_ddm_ali_with_tuning(
        V, I, T_DEFAULT, NS_DEFAULT
    )
    IL0, I010, I020, Rs0, Rsh0, n10, n20 = params_ini
    I_sim_ini = simulate_ddm_curve(V, params_ini, T_DEFAULT, NS_DEFAULT, I_meas=I)

    Rs0_ohm_cm2  = Rs0  * 1e3
    Rsh0_ohm_cm2 = Rsh0 * 1e3

    print("\n--- Classical Double Diode method (initial) 2-diode DDM parameters ---")
    print(f"IL  = {IL0: .6e} mA/cm²")
    print(f"I01 = {I010: .6e} mA/cm²")
    print(f"I02 = {I020: .6e} mA/cm²")
    print(f"Rs  = {Rs0_ohm_cm2:.3f} Ω·cm²   (DDM initial)")
    print(f"Rsh = {Rsh0_ohm_cm2:.3f} Ω·cm²  (DDM initial)")
    print(f"n1  = {n10: .2f}")
    print(f"n2  = {n20: .2f}")

    # 3) Nonlinear refinement of all parameters
    params_ref, I_sim_ref, cost_ref = refine_ddm_with_least_squares(
        V, I, params_ini, keypoints, T_DEFAULT, NS_DEFAULT
    )
    ILr, I01r, I02r, Rsr, Rshr, n1r, n2r = params_ref

    Rs_ohm_cm2  = Rsr * 1e3
    Rsh_ohm_cm2 = Rshr * 1e3

    print("\n--- Refined 2-diode DDM parameters (DDM + LSQ) ---")
    print(f"IL   = {ILr:.4f} mA/cm²")
    print(f"I01  = {I01r:.4e} mA/cm²")
    print(f"I02  = {I02r:.4e} mA/cm²")
    print(f"Rs   = {Rs_ohm_cm2:.3f} Ω·cm²")
    print(f"Rsh  = {Rsh_ohm_cm2:.3f} Ω·cm²")
    print(f"n1   = {n1r:.2f}")
    print(f"n2   = {n2r:.2f}")
    print(f"NRMSE (refined, 0..Vmax) = {cost_ref:.4e}")

    Isc, Voc, Imp, Vmp = keypoints

    
    base_dir = os.path.dirname(fname)
    if base_dir == "":
        base_dir = os.getcwd()

   
    # 4) Plot
    plt.figure()
    plt.plot(V, I, "o", color="red", ms=8, mfc="none", label="Measured IV")
    plt.plot(V, I_sim_ini, "-", color="green", lw=2.5, label="Classical Method (DDM)")
    plt.plot(V, I_sim_ref, "-", color="black", lw=3.5, label="Our Method (PC-DDM)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current density (mA/cm²)")
    plt.title("Refined double-diode model fit for OPV / perovskite devices (DDM + nonlinear LSQ)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
