# -*- coding: utf-8 -*-
"""
PC-DDM: Physically Constrained Double-Diode Modeling for OPV Devices
----------------------------------------------------------------------
This script performs J-V analysis of organic photovoltaic (OPV) devices
using a Physically Constrained Double-Diode Model (PC-DDM).

Main features:
- Reads two-column .dat files (Voltage [V], Current density [mA/cm²]).
- Extracts key photovoltaic parameters: Jsc, Voc, Jmp, Vmp.
- Computes initial parameters via RAI-DDM (Sentuerk recursive analytical scheme).
- Applies PC-DDM constrained nonlinear least-squares refinement.
- Multi-start optimization for robust convergence.
- Console output only — no Excel or file export.

Key methodological choices:
- n1 fixed to 1.0 (physically motivated regularization).
- n2 upper bound = 10.0 (confirmed by identifiability analysis).
- Post-Voc high-voltage weighting for improved Rs/n2/J02 sensitivity.

Model parameters:
  JL   : Photogenerated current density (mA/cm²)
  J01  : Saturation current density — diode 1, n1=1 (mA/cm²)
  J02  : Saturation current density — diode 2, recombination (mA/cm²)
  Rs   : Series resistance (Ω·cm²)
  Rsh  : Shunt resistance (Ω·cm²)
  n1   : Diffusion-diode ideality factor (fixed = 1.0)
  n2   : Recombination-diode ideality factor (optimized, ≤ 10.0)

Author: Koray Kara
"""
DDM refined fitting with MULTI-START, n1 FIXED (removed from optimization vector)

- Reads OPV J-V data (.dat/.csv two columns or Keithley multi-channel .txt)
- Builds an initial parameter guess (RAI-DDM initialization if possible, otherwise fallback)
- Refines parameters via bounded nonlinear least squares with post-Voc weighting
- Multi-start explores the bounded parameter space robustly (uniform + local perturbations)
- Saves: (i) IV + initial/refined fits to Excel for Origin, (ii) fit results to Excel

Author: Koray Kara
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
from tkinter import filedialog
from scipy.optimize import least_squares
import os

# ---------------------------------------------------
# Physical constants
# ---------------------------------------------------
k_B = 1.380649e-23     # J/K
q   = 1.602176634e-19  # C

T_DEFAULT  = 298.0     # K
NS_DEFAULT = 1         # OPV: single cell

# n1 is fixed globally
N1_FIXED = 1.0

# Global active area (cm^2) – asked at program start
ACTIVE_AREA_CM2 = None


# ---------------------------------------------------
# Basic 2-column IV loader (.dat/.txt/.csv)
# ---------------------------------------------------
def load_iv_two_column(filename: str):
    print(f"Selected file: {filename}")
    V_list, I_list = [], []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("%"):
                continue
            line = line.replace(",", ".")
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
        raise ValueError("Insufficient data. Check file format.")

    V = np.array(V_list, dtype=float)
    I = np.array(I_list, dtype=float)

    idx = np.argsort(V)
    V, I = V[idx], I[idx]

    print(f"Total {len(V)} data points loaded.")
    print("First 5 points (V, I):")
    for k in range(min(5, len(V))):
        print(f"{V[k]: .5e}\t{I[k]: .5e}")

    idx0 = np.argmin(np.abs(V))
    if I[idx0] < 0:
        I = -I
        print("Note: Jsc was negative; current sign converted to positive for fitting.")

    return V, I


# ---------------------------------------------------
# Keithley multi-channel .txt loader
# ---------------------------------------------------
def load_iv_plaintext():
    """Open a file dialog and load J-V data from a two-column .dat file."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("Please select a J-V data file (.dat)...")
    filename = filedialog.askopenfilename(
        title="Select J-V data file",
        filetypes=[("Data files", "*.dat"), ("All files", "*.*")]
    )
    if not filename:
        raise FileNotFoundError("No file selected.")

    V, I = load_iv_two_column(filename)
    return V, I, filename


def extract_keypoints(V, I):
    idx0 = np.argmin(np.abs(V))
    Isc = I[idx0]

    mask_voc = V >= 0
    if np.any(mask_voc):
        idx_voc_local = np.argmin(np.abs(I[mask_voc]))
        Voc = V[mask_voc][idx_voc_local]
    else:
        idx_voc = np.argmin(np.abs(I))
        Voc = V[idx_voc]

    mask_mpp = (V > 0) & (I > 0)
    if not np.any(mask_mpp):
        mask_mpp = V > 0

    P = V[mask_mpp] * I[mask_mpp]
    idx_mpp = np.argmax(P)
    Vmp = V[mask_mpp][idx_mpp]
    Imp = I[mask_mpp][idx_mpp]

    return Isc, Voc, Imp, Vmp


# ---------------------------------------------------
# Double-diode current solver (Newton-Raphson)
# params = (IL, I01, I02, Rs, Rsh, n1_fixed, n2)
# ---------------------------------------------------
def ddm_current_at_voltage(V, params, T, Ns, I_init=None, max_iter=80, tol=1e-9):
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
        # prevent runaway steps
        step = np.clip(step, -1.0, 1.0)   # simple and effective
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
# RAI-DDM: Recursive Analytical Initialization (Şentürk scheme)
# NOTE: This part still uses n1,n2 for initialization only.
# ---------------------------------------------------
def compute_rai_ddm(Isc, Voc, Imp, Vmp, T, Ns, X=1.0, Y=1.0, n1=1.0, n2=2.0, n_runs=2):
    VT = k_B * T * Ns / q

    Rs0 = (Voc - Vmp) / (X * Imp)
    Rsh0 = (Y * Vmp) / (Isc - Imp)

    Rs, Rsh = Rs0, Rsh0

    for _ in range(n_runs):
        IL = Isc * (1.0 + Rs / Rsh)

        B = np.exp(Voc / (n1 * VT)) - 1.0
        J = np.exp(Voc / (n2 * VT)) - 1.0
        K = IL - Voc / Rsh

        Vd_mp = Vmp + Imp * Rs
        M = np.exp(Vd_mp / (n1 * VT)) - 1.0
        N = np.exp(Vd_mp / (n2 * VT)) - 1.0

        G = IL - Imp - Vd_mp / Rsh

        Det = B * N - J * M
        if np.abs(Det) < 1e-30:
            raise ValueError("Determinant too small; I01/I02 cannot be computed.")

        I01 = (K * N - J * G) / Det
        I02 = (B * G - K * M) / Det

        denom = (I01 * np.exp(Voc / (n1 * VT)) / (n1 * VT)
                 + I02 * np.exp(Voc / (n2 * VT)) / (n2 * VT))
        if np.abs(denom) < 1e-30:
            raise ValueError("Denominator too small for Rs update.")
        Rs = Rs0 - 1.0 / denom

        Rsh = Voc / (IL
                     - I01 * (np.exp(Voc / (n1 * VT)) - 1.0)
                     - I02 * (np.exp(Voc / (n2 * VT)) - 1.0))

        Rs0, Rsh0 = Rs, Rsh

    return (IL, I01, I02, Rs, Rsh, n1, n2)


def cost_full_with_knee_weight(V, I_sim, I_meas, Voc, w_knee=3.0, w_tail=1.0):
    mask1 = (V >= 0) & (V <= Voc + 1e-3)
    mask2 = V > Voc

    E1 = I_sim[mask1] - I_meas[mask1]
    E2 = I_sim[mask2] - I_meas[mask2]

    rmse1 = np.sqrt(np.mean(E1 ** 2)) if len(E1) else 0.0
    rmse2 = np.sqrt(np.mean(E2 ** 2)) if len(E2) else 0.0

    rng = np.max(I_meas) - np.min(I_meas) + 1e-30
    return (w_knee * rmse1 + w_tail * rmse2) / rng


def fit_rai_ddm_initialization(V, I, T=T_DEFAULT, Ns=NS_DEFAULT):
    Isc, Voc, Imp, Vmp = extract_keypoints(V, I)
    print(f"\nKeypoints (measured): Jsc={Isc: .4e}, Voc={Voc: .4e}, Jmp={Imp: .4e}, Vmp={Vmp: .4e}")

    n1_values = [1.0, 1.2, 1.5]
    n2_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    X_values  = [0.8, 0.9, 1.0, 1.1, 1.2]
    Y_values  = [0.5, 1.0, 1.5, 2.0]

    best_cost = np.inf
    best_params = None
    best_hyper = None

    for n1 in n1_values:
        for n2 in n2_values:
            for X in X_values:
                for Y in Y_values:
                    try:
                        params = compute_rai_ddm(Isc, Voc, Imp, Vmp, T, Ns, X=X, Y=Y, n1=n1, n2=n2, n_runs=2)
                        IL, I01, I02, Rs, Rsh, _, _ = params
                        if (IL <= 0) or (I01 <= 0) or (I02 <= 0) or (Rs <= 0) or (Rsh <= 0):
                            continue
                        I_sim = simulate_ddm_curve(V, params, T, Ns, I_meas=I)
                        cost = cost_full_with_knee_weight(V, I_sim, I, Voc, w_knee=3.0, w_tail=1.0)
                        if cost < best_cost:
                            best_cost = cost
                            best_params = params
                            best_hyper = (n1, n2, X, Y)
                    except Exception:
                        continue

    if best_params is None:
        print("\nWarning: RAI-DDM initialization did not yield valid parameters. Using fallback.")
        print("      Using simple fallback DDM initialization.")

        VT = k_B * T * Ns / q
        n1_f, n2_f = 1.2, 2.0

        Rs_f = max((Voc / (Imp + 1e-9)) * 0.1, 1e-4)

        denom_rsh = (Isc - Imp)
        Rsh_f = max(Voc / denom_rsh, 1e-3) if abs(denom_rsh) >= 1e-9 else 1e3

        IL_f = max(Isc * (1.0 + Rs_f / Rsh_f), 1e-6)

        try:
            I01_f = (IL_f - Voc / Rsh_f) / (np.exp(Voc / (n1_f * VT)) - 1.0)
        except OverflowError:
            I01_f = 1e-12
        if (not np.isfinite(I01_f)) or (I01_f <= 0):
            I01_f = 1e-12

        I02_f = 1e-14
        best_params = (IL_f, I01_f, I02_f, Rs_f, Rsh_f, n1_f, n2_f)
        best_cost = np.nan
        return best_params, best_cost, (Isc, Voc, Imp, Vmp)

    n1_best, n2_best, X_best, Y_best = best_hyper
    print(f"\nBest RAI-DDM initialization parameters: n1={n1_best:.2f}, n2={n2_best:.2f}, X={X_best:.3f}, Y={Y_best:.3f}")
    print(f"NRMSE (RAI-DDM initial, post-Voc weighted) = {best_cost:.4e}")

    IL, I01, I02, Rs, Rsh, _, _ = best_params
    best_params = (IL, I01, I02, Rs, Rsh, n1_best, n2_best)
    return best_params, best_cost, (Isc, Voc, Imp, Vmp)


# ---------------------------------------------------
# Parameter vector mapping (n1 removed from optimization)
# p_vec = [log(IL), log(I01), log(I02), log(Rs), log(Rsh), n2]
# params = (IL, I01, I02, Rs, Rsh, n1_fixed, n2)
# ---------------------------------------------------
def params_to_vec_n1_fixed(params):
    IL, I01, I02, Rs, Rsh, _, n2 = params
    return np.array([np.log(IL), np.log(I01), np.log(I02), np.log(Rs), np.log(Rsh), float(n2)], dtype=float)

def vec_to_params_n1_fixed(p_vec, n1_fixed=N1_FIXED):
    IL  = np.exp(p_vec[0])
    I01 = np.exp(p_vec[1])
    I02 = np.exp(p_vec[2])
    Rs  = np.exp(p_vec[3])
    Rsh = np.exp(p_vec[4])
    n2  = float(p_vec[5])
    return (IL, I01, I02, Rs, Rsh, float(n1_fixed), n2)


# ---------------------------------------------------
# Bounded least squares refinement (n1 fixed)
# ---------------------------------------------------
def refine_ddm_with_least_squares(
    V, I_meas, params_ini, keypoints,
    T=298.0, Ns=1,
    p0_override=None,
    return_res=False,
    rs_upper=None, rsh_upper=None, n2_upper=None,
    max_nfev=6000
):

    Isc, Voc, Imp, Vmp = keypoints

    # 6D initial vector
    p0 = params_to_vec_n1_fixed(params_ini)
    if p0_override is not None:
        p0 = np.array(p0_override, dtype=float)

    # 6D bounds
    lower = np.array([
        np.log(1e-12 * abs(Isc) + 1e-18),  # IL
        np.log(1e-20),                     # I01
        np.log(1e-20),                     # I02
        np.log(1e-4),                      # Rs
        np.log(1e-3),                      # Rsh
        1.2                                # n2
    ], dtype=float)

    upper = np.array([
        np.log(1e3 * abs(Isc) + 1e-18),     # IL
        np.log(1e-1),                       # I01
        np.log(1e-1),                       # I02
        np.log(rs_upper),                   # Rs
        np.log(rsh_upper),                  # Rsh
        float(n2_upper)                     # n2
    ], dtype=float)

    # ensure p0 inside bounds
    margin = 1e-12
    p0 = np.maximum(p0, lower + margin)
    p0 = np.minimum(p0, upper - margin)

    def residuals(p_vec):
        p_vec = np.copy(p_vec)
        # n2 clip (index 5)
        p_vec[5] = np.clip(p_vec[5], 1.2, float(n2_upper))

        params = vec_to_params_n1_fixed(p_vec, n1_fixed=N1_FIXED)
        I_sim = simulate_ddm_curve(V, params, T, Ns, I_meas=I_meas)

        # Post-Voc emphasis (as in original workflow)
        Vmax = np.max(V)
        z = (V - 0.7 * Voc) / (Vmax - 0.7 * Voc + 1e-12)
        w = 1.0 + 4.0 * np.clip(z, 0.0, 1.0)
        return (I_sim - I_meas) * w

    res = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        method="trf",
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
        max_nfev=max_nfev
    )

    p_best = res.x
    params_best = vec_to_params_n1_fixed(p_best, n1_fixed=N1_FIXED)

    I_sim_best = simulate_ddm_curve(V, params_best, T, Ns, I_meas=I_meas)
    rng = np.max(I_meas) - np.min(I_meas) + 1e-30
    nrmse = np.sqrt(np.mean((I_sim_best - I_meas) ** 2)) / rng

    if return_res:
        return params_best, I_sim_best, nrmse, res
    return params_best, I_sim_best, nrmse


# ---------------------------------------------------
# Multi-start (n1 fixed, 6D exploration)
# ---------------------------------------------------
def _make_p0_uniform_in_bounds(lower, upper, rng):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    margin = 1e-12
    span = np.maximum(upper - lower, 1e-15)
    p0 = lower + rng.random(lower.size) * span
    p0 = np.minimum(p0, upper - margin)
    p0 = np.maximum(p0, lower + margin)
    return p0


def multistart_refine_ddm(
    V, I_meas, params_ini, keypoints,
    n_starts=50, seed=1,
    T=T_DEFAULT, Ns=NS_DEFAULT,
    local_fraction=0.3,
    local_log_sigma=0.35,
    local_n2_sigma=0.8,
    rs_upper=1e5, rsh_upper=1e6, n2_upper=7.2,
    verbose=True
):
    Isc, Voc, Imp, Vmp = keypoints

    # Build the anchor start vector from params_ini (6D)
    p0_anchor = params_to_vec_n1_fixed(params_ini)

    # bounds must match refine
    lower = np.array([
        np.log(1e-12 * abs(Isc) + 1e-18),
        np.log(1e-20),
        np.log(1e-20),
        np.log(1e-4),
        np.log(1e-3),
        1.2
    ], dtype=float)
    upper = np.array([
        np.log(1e3 * abs(Isc) + 1e-18),
        np.log(1e-1),
        np.log(1e-1),
        np.log(rs_upper),
        np.log(rsh_upper),
        float(n2_upper)
    ], dtype=float)

    margin = 1e-12
    p0_anchor = np.maximum(p0_anchor, lower + margin)
    p0_anchor = np.minimum(p0_anchor, upper - margin)

    rng_master = np.random.default_rng(seed)
    runs = []
    best = None

    for k in range(n_starts):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))

        if k == 0:
            p0_rand = p0_anchor.copy()
            start_type = "anchor"
        else:
            if rng.random() < local_fraction:
                p0_rand = p0_anchor.copy()
                p0_rand[:5] = p0_rand[:5] + rng.normal(0.0, local_log_sigma, size=5)
                p0_rand[5]  = p0_rand[5]  + rng.normal(0.0, local_n2_sigma)
                p0_rand = np.maximum(p0_rand, lower + margin)
                p0_rand = np.minimum(p0_rand, upper - margin)
                start_type = "local"
            else:
                # safer "global": wide perturbation around anchor (still diverse, but not crazy)
                p0_rand = p0_anchor.copy()
                p0_rand[:5] = p0_rand[:5] + rng.normal(0.0, 0.8, size=5)  # log-params
                p0_rand[5]  = p0_rand[5]  + rng.normal(0.0, 1.0)          # n2
                p0_rand = np.maximum(p0_rand, lower + margin)
                p0_rand = np.minimum(p0_rand, upper - margin)
                start_type = "global"


        if verbose:
            print(f"[multi-start] run {k+1}/{n_starts} ({start_type}) ...")

        params_k, I_sim_k, nrmse_k, res_k = refine_ddm_with_least_squares(
            V, I_meas, params_ini, keypoints,
            T=T, Ns=Ns,
            return_res=True,
            p0_override=p0_rand,
            rs_upper=rs_upper, rsh_upper=rsh_upper, n2_upper=n2_upper
        )

        run_info = {
            "run": k,
            "start_type": start_type,
            "p0": p0_rand,
            "params": params_k,
            "I_sim": I_sim_k,
            "nrmse": float(nrmse_k),
            "success": bool(getattr(res_k, "success", True)),
            "nfev": int(getattr(res_k, "nfev", -1)),
            "cost": float(getattr(res_k, "cost", np.nan)),
            "res": res_k,
        }
        runs.append(run_info)

        if verbose:
            print(f"    done: success={run_info['success']}  nrmse={run_info['nrmse']:.6g}  nfev={run_info['nfev']}")

        if (best is None) or (run_info["nrmse"] < best["nrmse"]):
            best = run_info

    return best, runs


def summarize_multistart(best, runs, rel_tol=0.01):
    best_nrmse = float(best["nrmse"])
    eq = [r for r in runs if r["success"] and float(r["nrmse"]) <= best_nrmse * (1.0 + float(rel_tol))]
    P = np.array([r["params"] for r in eq], dtype=float) if len(eq) else np.empty((0, 7))

    out = {
        "n_total": int(len(runs)),
        "n_success": int(sum(1 for r in runs if r["success"])),
        "n_equivalent": int(len(eq)),
        "best_nrmse": best_nrmse,
        "median": (np.median(P, axis=0) if len(eq) else None),
        "iqr": ((np.percentile(P, 75, axis=0) - np.percentile(P, 25, axis=0)) if len(eq) else None),
    }
    return out, eq, P


def jacobian_correlation_matrix(res):
    """
    Correlation matrix from Gauss–Newton approximation.
    Note: uses the weighted Jacobian in `res.jac`.
    """
    J = res.jac
    m, n = J.shape
    JTJ = J.T @ J
    JTJ_inv = np.linalg.pinv(JTJ)
    dof = max(m - n, 1)
    sigma2 = (2.0 * float(res.cost)) / float(dof)
    cov = sigma2 * JTJ_inv
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, np.inf))
    corr = cov / np.outer(d, d)
    return cov, corr


# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    # 0) active area
    try:
        area_str = input("Enter active area (cm²) [e.g., 0.05]. Leave blank for default 0.05: ")
        ACTIVE_AREA_CM2 = 0.05 if area_str.strip() == "" else float(area_str.replace(",", "."))
    except Exception:
        ACTIVE_AREA_CM2 = 0.05

    print(f"Active area: {ACTIVE_AREA_CM2:.4f} cm²\n")

    # 1) load IV
    V, I, fname = load_iv_plaintext()

    # 2) RAI-DDM initialization
    params_ini, cost_ini, keypoints = fit_rai_ddm_initialization(V, I, T_DEFAULT, NS_DEFAULT)
    IL0, I010, I020, Rs0, Rsh0, n10, n20 = params_ini
    I_sim_ini = simulate_ddm_curve(V, params_ini, T_DEFAULT, NS_DEFAULT, I_meas=I)

    Rs0_ohm_cm2  = Rs0  * 1e3
    Rsh0_ohm_cm2 = Rsh0 * 1e3

    print("\n--- RAI-DDM Initialization Parameters (Şentürk scheme) ---")
    print(f"IL  = {IL0: .6e} mA/cm²")
    print(f"I01 = {I010: .6e} mA/cm²")
    print(f"I02 = {I020: .6e} mA/cm²")
    print(f"Rs  = {Rs0_ohm_cm2:.3f} Ω·cm²   (RAI-DDM initial)")
    print(f"Rsh = {Rsh0_ohm_cm2:.3f} Ω·cm²  (RAI-DDM initial)")
    print(f"n1_init = {n10: .2f}  (RAI-DDM init only; final n1 FIXED to {N1_FIXED:.1f})")
    print(f"n2  = {n20: .2f}")

    # 3) Multi-start refine
    n_starts = 100
    seed = 1
    rel_tol = 0.01

    best_run, runs = multistart_refine_ddm(
        V, I, params_ini, keypoints,
        n_starts=n_starts, seed=seed,
        T=T_DEFAULT, Ns=NS_DEFAULT,
        verbose=True
    )

    ms_summary, eq_runs, _P = summarize_multistart(best_run, runs, rel_tol=rel_tol)

    print()
    print("--- Multi-start summary ---")
    print(f"Total starts      : {ms_summary['n_total']}")
    print(f"Successful        : {ms_summary['n_success']}")
    print(f"Equivalent (±{rel_tol*100:.1f}%) : {ms_summary['n_equivalent']}")
    print(f"Best NRMSE        : {ms_summary['best_nrmse']:.6g}")

    params_ref = best_run["params"]
    I_sim_ref = simulate_ddm_curve(V, params_ref, T_DEFAULT, NS_DEFAULT, I_meas=I)
    cost_ref = best_run["nrmse"]

    # 4) Correlation matrix (best run) — for 6D optimization parameters
    cov, corr = jacobian_correlation_matrix(best_run["res"])
    param_names = ["logIL", "logI01", "logI02", "logRs", "logRsh", "n2"]

    print()
    print("--- Parameter correlation matrix (best run, 6D) ---")
    header = "       " + " ".join([f"{n:>8s}" for n in param_names])
    print(header)
    for i, name_i in enumerate(param_names):
        row = " ".join([f"{corr[i, j]:+8.3f}" for j in range(len(param_names))])
        print(f"{name_i:>6s} {row}")

    ILr, I01r, I02r, Rsr, Rshr, n1r, n2r = params_ref

    Rs_ohm_cm2  = Rsr * 1e3
    Rsh_ohm_cm2 = Rshr * 1e3

    print("\n--- PC-DDM Refined Parameters (Multi-start, n1 fixed) ---")
    print(f"IL   = {ILr:.4f} mA/cm²")
    print(f"I01  = {I01r:.4e} mA/cm²")
    print(f"I02  = {I02r:.4e} mA/cm²")
    print(f"Rs   = {Rs_ohm_cm2:.3f} Ω·cm²")
    print(f"Rsh  = {Rsh_ohm_cm2:.3f} Ω·cm²")
    print(f"n1   = {N1_FIXED:.2f} (fixed)")
    print(f"n2   = {n2r:.2f}")
    print(f"NRMSE (PC-DDM refined, full range) = {cost_ref:.4e}")

    # 5) Plot
    plt.figure()
    plt.plot(V, I, "o", label="Measured J-V")
    plt.plot(V, I_sim_ini, "-", label="RAI-DDM fit (Şentürk initialization)")
    plt.plot(V, I_sim_ref, "-", label="PC-DDM fit (multi-start, n1 fixed)")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current Density (mA/cm²)")
    plt.title("PC-DDM Fit — OPV J-V Analysis (Multi-start, n1 fixed)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
