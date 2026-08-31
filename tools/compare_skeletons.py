import argparse
import os
import glob
import yaml
import numpy as np
import json
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_gt_skeleton(txt_path):
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) == 150:
                coords = np.array([float(x) for x in parts]).reshape(2, 25, 3)
                data.append(coords)
    return np.array(data)

def load_extracted_skeleton(csv_path):
    frame_ids = []
    data = []
    with open(csv_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if not parts: continue
            if len(parts) == 151:
                frame_ids.append(int(parts[0]))
                coords = np.array([float(x) for x in parts[1:]]).reshape(2, 25, 3)
                data.append(coords)
    return np.array(frame_ids), np.array(data)

def align_root_and_scale_gt(gt_skel, ext_skel, root_idx=0, spine_idx=1):
    aligned_gt = gt_skel.copy()
    aligned_ext = ext_skel.copy()
    
    for t in range(len(gt_skel)):
        for p in range(2):
            root_gt = gt_skel[t, p, root_idx].copy()
            if not np.all(root_gt == 0):
                aligned_gt[t, p] -= root_gt
            root_ext = ext_skel[t, p, root_idx].copy()
            if not np.all(root_ext == 0):
                aligned_ext[t, p] -= root_ext

    aligned_gt[:, :, :, 0] *= -1
    aligned_gt[:, :, :, 1] *= -1
    
    for p in range(2):
        spine_vecs = aligned_gt[:, p, spine_idx] - aligned_gt[:, p, root_idx]
        spine_lengths = np.linalg.norm(spine_vecs, axis=-1)
        valid_lengths = spine_lengths[spine_lengths > 0.01]
        if len(valid_lengths) > 0:
            scale = np.mean(valid_lengths)
            aligned_gt[:, p] /= scale
            
    return aligned_gt, aligned_ext

def procrustes_align(gt_pts, ext_pts):
    # Procrustes Analysis (Translation, Scaling, Rotation) to align ext to gt
    valid = (np.linalg.norm(gt_pts, axis=1) > 0) & (np.linalg.norm(ext_pts, axis=1) > 0)
    if np.sum(valid) < 3:
        return ext_pts
    
    A = gt_pts[valid]
    B = ext_pts[valid]
    
    c_A = np.mean(A, axis=0)
    c_B = np.mean(B, axis=0)
    A_c = A - c_A
    B_c = B - c_B
    
    scale_A = np.linalg.norm(A_c)
    scale_B = np.linalg.norm(B_c)
    if scale_B == 0:
        return ext_pts
        
    s = scale_A / scale_B
    B_c = B_c * s
    
    H = B_c.T @ A_c
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U @ Vt
        
    ext_aligned = (ext_pts - c_B) * s @ R + c_A
    return ext_aligned

def calculate_errors(gt_frame, ext_frame, apply_pa=True):
    # Calculate both MPJPE and PA-MPJPE for a matched pair (pA, pB)
    mask = (np.any(gt_frame != 0, axis=-1)) & (np.any(ext_frame != 0, axis=-1))
    if not np.any(mask):
        return np.full(25, np.nan), np.full(25, np.nan)
        
    # Standard MPJPE
    dist = np.linalg.norm(gt_frame - ext_frame, axis=-1)
    dist[~mask] = np.nan
    
    # PA-MPJPE
    if apply_pa:
        pa_ext_frame = procrustes_align(gt_frame, ext_frame)
        pa_dist = np.linalg.norm(gt_frame - pa_ext_frame, axis=-1)
        pa_dist[~mask] = np.nan
    else:
        pa_dist = dist.copy()
        
    return dist, pa_dist

def match_persons_and_calculate_error(gt_f, ext_f):
    # Try permutation (0-0, 1-1)
    dist_00, pa_dist_00 = calculate_errors(gt_f[0], ext_f[0])
    dist_11, pa_dist_11 = calculate_errors(gt_f[1], ext_f[1])
    score_1 = np.nanmean([np.nanmean(dist_00), np.nanmean(dist_11)])
    
    # Try permutation (0-1, 1-0)
    dist_01, pa_dist_01 = calculate_errors(gt_f[0], ext_f[1])
    dist_10, pa_dist_10 = calculate_errors(gt_f[1], ext_f[0])
    score_2 = np.nanmean([np.nanmean(dist_01), np.nanmean(dist_10)])
    
    if np.isnan(score_1) and np.isnan(score_2):
        return (dist_00, dist_11), (pa_dist_00, pa_dist_11)
        
    if score_1 <= score_2 or np.isnan(score_2):
        return (dist_00, dist_11), (pa_dist_00, pa_dist_11)
    else:
        return (dist_01, dist_10), (pa_dist_01, pa_dist_10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tools/config_compare.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gt_dir = cfg['paths']['gt_dir']
    ext_dir = cfg['paths']['ext_dir']
    out_dir = cfg['paths']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    # Find all CSV files in extracted dir
    csv_files = glob.glob(os.path.join(ext_dir, "*.csv"))
    if args.limit:
        csv_files = csv_files[:args.limit]
        
    print(f"Found {len(csv_files)} extracted videos to compare.")
    
    all_mpjpe = []
    all_pa_mpjpe = []
    video_summaries = []
    
    for csv_path in tqdm(csv_files, desc="Comparing Skeletons"):
        video_name = os.path.basename(csv_path).replace('.csv', '')
        gt_path = os.path.join(gt_dir, f"{video_name}.txt")
        
        if not os.path.exists(gt_path):
            continue
            
        gt_data = load_gt_skeleton(gt_path)
        ext_fids, ext_data = load_extracted_skeleton(csv_path)
        
        matched_gt = []
        matched_ext = []
        for i, fid in enumerate(ext_fids):
            gt_idx = fid - 1
            if 0 <= gt_idx < len(gt_data):
                matched_gt.append(gt_data[gt_idx])
                matched_ext.append(ext_data[i])
                
        if len(matched_gt) == 0:
            continue
            
        matched_gt = np.array(matched_gt)
        matched_ext = np.array(matched_ext)
        
        if cfg['alignment']['align_root']:
            aligned_gt, aligned_ext = align_root_and_scale_gt(matched_gt, matched_ext, cfg['alignment']['root_joint_idx'])
        else:
            aligned_gt, aligned_ext = matched_gt, matched_ext
            
        video_mpjpe = []
        video_pa_mpjpe = []
        
        for i in range(len(matched_gt)):
            (dists, pa_dists) = match_persons_and_calculate_error(aligned_gt[i], aligned_ext[i])
            
            for p in range(2):
                if not np.all(np.isnan(dists[p])):
                    video_mpjpe.append(dists[p])
                    video_pa_mpjpe.append(pa_dists[p])
                    
        if video_mpjpe:
            all_mpjpe.extend(video_mpjpe)
            all_pa_mpjpe.extend(video_pa_mpjpe)
            
            video_summaries.append({
                "video": video_name,
                "frames": len(video_mpjpe),
                "mpjpe": float(np.nanmean(video_mpjpe)),
                "pa_mpjpe": float(np.nanmean(video_pa_mpjpe))
            })

    if not all_mpjpe:
        print("No matching data found.")
        return
        
    all_mpjpe = np.array(all_mpjpe)
    all_pa_mpjpe = np.array(all_pa_mpjpe)
    
    # Calculate PCK @ 0.2 (20% of normalized spine length)
    threshold = 0.2
    pck_mask = all_mpjpe < threshold
    pck = np.nansum(pck_mask, axis=0) / np.sum(~np.isnan(all_mpjpe), axis=0)
    
    pa_pck_mask = all_pa_mpjpe < threshold
    pa_pck = np.nansum(pa_pck_mask, axis=0) / np.sum(~np.isnan(all_pa_mpjpe), axis=0)
    
    results = {
        "dataset_summary": {
            "total_videos": len(video_summaries),
            "total_samples_compared": len(all_mpjpe),
            "overall_mpjpe": float(np.nanmean(all_mpjpe)),
            "overall_pa_mpjpe": float(np.nanmean(all_pa_mpjpe)),
            "overall_pck_02": float(np.nanmean(pck)),
            "overall_pa_pck_02": float(np.nanmean(pa_pck)),
            "per_joint_mpjpe": np.nanmean(all_mpjpe, axis=0).tolist(),
            "per_joint_pa_mpjpe": np.nanmean(all_pa_mpjpe, axis=0).tolist(),
            "per_joint_pck": pck.tolist(),
            "per_joint_pa_pck": pa_pck.tolist()
        },
        "video_summaries": video_summaries
    }
    
    out_file = os.path.join(out_dir, "dataset_comparison.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Done! Dataset-level results saved to {out_file}")
    print(f"Overall MPJPE: {results['dataset_summary']['overall_mpjpe']:.4f}")
    print(f"Overall PA-MPJPE: {results['dataset_summary']['overall_pa_mpjpe']:.4f}")
    print(f"Overall PCK@0.2: {results['dataset_summary']['overall_pck_02']:.4%}")
    print(f"Overall PA-PCK@0.2: {results['dataset_summary']['overall_pa_pck_02']:.4%}")

if __name__ == "__main__":
    main()
