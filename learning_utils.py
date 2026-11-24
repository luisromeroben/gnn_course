import os
import numpy as np
import pandas as pd

def save_task1_results(filepath, df_res, snapshots_dict):
    """Saves both the benchmark summary table and spatial predictions to a .npz archive."""
    payload = {
        "table_sensor_pct": df_res["Sensor %"].to_numpy(dtype=str),
        "table_rho": df_res["rho"].to_numpy(dtype=np.float64),
        "table_model": df_res["Model"].to_numpy(dtype=str),
        "table_family": df_res["Family"].to_numpy(dtype=str),
        "table_mae": df_res["MAE (m)"].to_numpy(dtype=np.float64),
        "table_rmse": df_res["RMSE (m)"].to_numpy(dtype=np.float64),
        "table_r2": df_res["R2"].to_numpy(dtype=np.float64),
        "table_params": df_res["Params"].to_numpy(dtype=np.int64),
        "ratios": np.array(sorted(snapshots_dict.keys()), dtype=np.float64),
    }
    for r_idx, rho in enumerate(payload["ratios"]):
        snap = snapshots_dict[float(rho)]
        payload[f"r{r_idx}_mask_np"] = np.asarray(snap["mask_np"], dtype=np.float32)
        model_names = [k for k in snap.keys() if k != "mask_np"]
        payload[f"r{r_idx}_models"] = np.array(model_names, dtype=str)
        for m_idx, m_name in enumerate(model_names):
            m = snap[m_name]
            payload[f"r{r_idx}_m{m_idx}_stats"] = np.array(
                [m["MAE (m)"], m["RMSE (m)"], m["R2"], m["Params"]], dtype=np.float64
            )
            payload[f"r{r_idx}_m{m_idx}_pred"] = np.asarray(m["sample_pred"], dtype=np.float32)
            payload[f"r{r_idx}_m{m_idx}_true"] = np.asarray(m["sample_true"], dtype=np.float32)
    np.savez_compressed(filepath, **payload)

def load_task1_results(filepath):
    """Restores df_task1 and saved_snapshots from disk without retraining."""
    with np.load(filepath, allow_pickle=False) as d:
        df_loaded = pd.DataFrame({
            "Sensor %": d["table_sensor_pct"].tolist(),
            "rho": d["table_rho"],
            "Model": d["table_model"].tolist(),
            "Family": d["table_family"].tolist(),
            "MAE (m)": d["table_mae"],
            "RMSE (m)": d["table_rmse"],
            "R2": d["table_r2"],
            "Params": d["table_params"],
        })
        snapshots_loaded = {}
        for r_idx, rho in enumerate(d["ratios"]):
            rho_f = round(float(rho), 6)
            snap = {"mask_np": d[f"r{r_idx}_mask_np"]}
            for m_idx, m_name in enumerate(d[f"r{r_idx}_models"].tolist()):
                st = d[f"r{r_idx}_m{m_idx}_stats"]
                snap[m_name] = {
                    "MAE (m)": float(st[0]),
                    "RMSE (m)": float(st[1]),
                    "R2": float(st[2]),
                    "Params": int(round(st[3])),
                    "sample_pred": d[f"r{r_idx}_m{m_idx}_pred"],
                    "sample_true": d[f"r{r_idx}_m{m_idx}_true"],
                }
            snapshots_loaded[rho_f] = snap
    return df_loaded, snapshots_loaded
