import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

NTU_JOINT_NAMES = [
    "SpineBase", "SpineMid", "Neck", "Head", "ShoulderLeft", 
    "ElbowLeft", "WristLeft", "HandLeft", "ShoulderRight", "ElbowRight", 
    "WristRight", "HandRight", "HipLeft", "KneeLeft", "AnkleLeft", 
    "FootLeft", "HipRight", "KneeRight", "AnkleRight", "FootRight", 
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dataset_comparison.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return
        
    with open(args.input, 'r') as f:
        data = json.load(f)
        
    out_dir = os.path.dirname(args.input)
    summary = data.get("dataset_summary", {})
    
    mpjpe = summary.get("per_joint_mpjpe", [])
    pa_mpjpe = summary.get("per_joint_pa_mpjpe", [])
    pck = summary.get("per_joint_pck", [])
    pa_pck = summary.get("per_joint_pa_pck", [])
    
    if not mpjpe:
        print("No valid data to plot.")
        return
        
    x = np.arange(len(NTU_JOINT_NAMES))
    width = 0.35
    
    # 1. Bar Chart for MPJPE vs PA-MPJPE
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, mpjpe, width, label='MPJPE', color='salmon')
    rects2 = ax.bar(x + width/2, pa_mpjpe, width, label='PA-MPJPE', color='skyblue')

    ax.set_ylabel('Error (Relative Units)')
    ax.set_title(f'Mean Per Joint Position Error - Overall: {summary["overall_mpjpe"]:.2f} | PA: {summary["overall_pa_mpjpe"]:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels(NTU_JOINT_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    mpjpe_out = os.path.join(out_dir, "dataset_mpjpe_comparison.png")
    plt.savefig(mpjpe_out)
    print(f"Saved MPJPE chart to {mpjpe_out}")
    plt.close()
    
    # 2. Bar Chart for PCK vs PA-PCK
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, [p * 100 for p in pck], width, label='PCK@0.2', color='lightgreen')
    rects2 = ax.bar(x + width/2, [p * 100 for p in pa_pck], width, label='PA-PCK@0.2', color='mediumseagreen')

    ax.set_ylabel('Percentage of Correct Keypoints (%)')
    ax.set_title(f'PCK@0.2 - Overall: {summary["overall_pck_02"]*100:.1f}% | PA: {summary["overall_pa_pck_02"]*100:.1f}%')
    ax.set_xticks(x)
    ax.set_xticklabels(NTU_JOINT_NAMES, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    pck_out = os.path.join(out_dir, "dataset_pck_comparison.png")
    plt.savefig(pck_out)
    print(f"Saved PCK chart to {pck_out}")
    plt.close()

if __name__ == "__main__":
    main()
