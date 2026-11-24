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

def save_task2_results(filepath, df_res, loc_snaps, mask_arr):
    payload = {
        "model": df_res["Model"].to_numpy(dtype=str),
        "family": df_res["Family"].to_numpy(dtype=str),
        "top1": df_res["Top-1 Acc (%)"].to_numpy(dtype=np.float64),
        "top5": df_res["Top-5 Acc (%)"].to_numpy(dtype=np.float64),
        "hops": df_res["Hop Error"].to_numpy(dtype=np.float64),
        "dist": df_res["Dist Error (m)"].to_numpy(dtype=np.float64),
        "params": df_res["Params"].to_numpy(dtype=np.int64),
        "mask_np": np.asarray(mask_arr, dtype=np.float32),
        "sample_true": np.array(loc_snaps["sample_true"], dtype=np.int64),
    }
    for idx, m_name in enumerate(df_res["Model"].tolist()):
        payload[f"m{idx}_probs"] = np.asarray(loc_snaps[m_name]["sample_probs"], dtype=np.float32)
    np.savez_compressed(filepath, **payload)

def load_task2_results(filepath):
    with np.load(filepath, allow_pickle=False) as d:
        df_loaded = pd.DataFrame({
            "Model": d["model"].tolist(),
            "Family": d["family"].tolist(),
            "Top-1 Acc (%)": d["top1"],
            "Top-5 Acc (%)": d["top5"],
            "Hop Error": d["hops"],
            "Dist Error (m)": d["dist"],
            "Params": d["params"],
        })
        loc_snaps = {
            "mask_np": d["mask_np"],
            "sample_true": int(d["sample_true"]),
        }
        for idx, m_name in enumerate(df_loaded["Model"].tolist()):
            loc_snaps[m_name] = {"sample_probs": d[f"m{idx}_probs"]}
    return df_loaded, loc_snaps

def save_task3_results(filepath, df_res, det_snaps):
    payload = {
        "model": df_res["Model"].to_numpy(dtype=str),
        "family": df_res["Family"].to_numpy(dtype=str),
        "acc": df_res["Accuracy (%)"].to_numpy(dtype=np.float64),
        "f1": df_res["F1-Score"].to_numpy(dtype=np.float64),
        "auc": df_res["ROC-AUC"].to_numpy(dtype=np.float64),
        "params": df_res["Params"].to_numpy(dtype=np.int64),
        "gae_scores": np.asarray(det_snaps["GraphAE (Unsup. GAE)"]["raw_scores"], dtype=np.float32),
        "gae_y": np.asarray(det_snaps["GraphAE (Unsup. GAE)"]["y_true"], dtype=np.int64),
        "gae_th": np.array(det_snaps["GraphAE (Unsup. GAE)"]["raw_threshold"], dtype=np.float64),
    }
    for idx, m_name in enumerate(df_res["Model"].tolist()):
        payload[f"m{idx}_fpr"] = np.asarray(det_snaps[m_name]["fpr"], dtype=np.float32)
        payload[f"m{idx}_tpr"] = np.asarray(det_snaps[m_name]["tpr"], dtype=np.float32)
    np.savez_compressed(filepath, **payload)

def load_task3_results(filepath):
    with np.load(filepath, allow_pickle=False) as d:
        df_loaded = pd.DataFrame({
            "Model": d["model"].tolist(),
            "Family": d["family"].tolist(),
            "Accuracy (%)": d["acc"],
            "F1-Score": d["f1"],
            "ROC-AUC": d["auc"],
            "Params": d["params"],
        })
        det_snaps = {
            "GraphAE (Unsup. GAE)": {
                "raw_scores": d["gae_scores"],
                "y_true": d["gae_y"],
                "raw_threshold": float(d["gae_th"]),
            }
        }
        for idx, m_name in enumerate(df_loaded["Model"].tolist()):
            det_snaps.setdefault(m_name, {})["fpr"] = d[f"m{idx}_fpr"]
            det_snaps[m_name]["tpr"] = d[f"m{idx}_tpr"]
    return df_loaded, det_snaps


def save_task4_results(filepath, df_res, fc_snaps):
    payload = {
        "model": df_res["Model"].to_numpy(dtype=str),
        "family": df_res["Family"].to_numpy(dtype=str),
        "mae": df_res["MAE (L/s)"].to_numpy(dtype=np.float64),
        "rmse": df_res["RMSE (L/s)"].to_numpy(dtype=np.float64),
        "r2": df_res["R2"].to_numpy(dtype=np.float64),
        "params": df_res["Params"].to_numpy(dtype=np.int64),
        "traj_true": np.asarray(fc_snaps["Persistence (d[t])"]["traj_true"], dtype=np.float32),
    }
    for idx, m_name in enumerate(df_res["Model"].tolist()):
        payload[f"m{idx}_traj_pred"] = np.asarray(fc_snaps[m_name]["traj_pred"], dtype=np.float32)
    np.savez_compressed(filepath, **payload)

def load_task4_results(filepath):
    with np.load(filepath, allow_pickle=False) as d:
        df_loaded = pd.DataFrame({
            "Model": d["model"].tolist(),
            "Family": d["family"].tolist(),
            "MAE (L/s)": d["mae"],
            "RMSE (L/s)": d["rmse"],
            "R2": d["r2"],
            "Params": d["params"],
        })
        fc_snaps = {"traj_true": d["traj_true"]}
        for idx, m_name in enumerate(df_loaded["Model"].tolist()):
            fc_snaps[m_name] = {
                "traj_true": d["traj_true"],
                "traj_pred": d[f"m{idx}_traj_pred"],
            }
    return df_loaded, fc_snaps
