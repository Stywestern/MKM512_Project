import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_tracking_plots(csv_filepath):
    # Load the telemetry data
    if not os.path.exists(csv_filepath):
        print(f"Error: The file {csv_filepath} does not exist.")
        return

    df = pd.read_csv(csv_filepath)
    
    # Deriving camera tracking center parameters from raw logic formulas:
    # dx = center_x - cx  ->  center_x = pan_ref + pan_pixel_error
    # dy = cy - center_y  ->  center_y = tilt_ref - tilt_pixel_error

    # Configure global clean plotting styles
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#eeeeee'
    plt.rcParams['grid.linestyle'] = '--'

    # 1. Plot 1: Pan Axis (Pan Reference, Pan Pixel Error, Camera X)
    fig_pan, ax_pan = plt.subplots(figsize=(10, 5))
    ax_pan.plot(df['id'], df['pan_ref'], color='#1f77b4', linewidth=1.5, label='Pan Reference (cx)')
    ax_pan.plot(df['id'], df['pan_pixel_error'], color='#d62728', linewidth=1.2, label='Pan Pixel Error (dx)')
    ax_pan.plot(df['id'], df['camera_x'], color='#2ca02c', linewidth=1.2, linestyle='--', label='Camera Center (X)')
    
    ax_pan.set_ylabel('Value (pixels)')
    ax_pan.set_xlabel('Frame ID / Sequence Index')
    ax_pan.set_title('Pan Axis Diagnostics: Reference, Error, and Camera Axis', fontsize=13, pad=12)
    ax_pan.grid(True)
    ax_pan.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
    
    plt.tight_layout()
    fig_pan.savefig('pan_combined_diagnostic.png', dpi=300)
    plt.close()

    # 2. Plot 2: Tilt Axis (Tilt Reference, Tilt Pixel Error, Camera Y)
    fig_tilt, ax_tilt = plt.subplots(figsize=(10, 5))
    ax_tilt.plot(df['id'], df['tilt_ref'], color='#2ca02c', linewidth=1.5, label='Tilt Reference (cy)')
    ax_tilt.plot(df['id'], df['tilt_pixel_error'], color='#ff7f0e', linewidth=1.2, label='Tilt Pixel Error (dy)')
    ax_tilt.plot(df['id'], df['camera_y'], color='#9467bd', linewidth=1.2, linestyle='--', label='Camera Center (Y)')
    
    ax_tilt.set_ylabel('Value (pixels)')
    ax_tilt.set_xlabel('Frame ID / Sequence Index')
    ax_tilt.set_title('Tilt Axis Diagnostics: Reference, Error, and Camera Axis', fontsize=13, pad=12)
    ax_tilt.grid(True)
    ax_tilt.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')
    
    plt.tight_layout()
    fig_tilt.savefig('tilt_combined_diagnostic.png', dpi=300)
    plt.close()

    # 3. Plot 3: Transformation Analysis (Side-by-Side Subplots)
    fig_trans, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 3A: Twin-X Time Series Comparison
    ax_t1.plot(df['id'], df['pan_pixel_error'], color='#d62728', linewidth=1.2, label='Pixel Error (px)')
    ax_t1_deg = ax_t1.twinx()
    ax_t1_deg.plot(df['id'], df['pan_deg_error'], color='#9467bd', linewidth=1.2, linestyle=':', label='Degree Error (°)')
    
    ax_t1.set_xlabel('Frame ID / Sequence Index')
    ax_t1.set_ylabel('Pixel Error (pixels)', color='#d62728')
    ax_t1_deg.set_ylabel('Degree Error (degrees)', color='#9467bd')
    ax_t1.set_title('Error Profiles Synchronization Over Time', fontsize=11, pad=8)
    ax_t1.grid(True)
    
    # Synchronize dual legends into one frame box
    lines_1, labels_1 = ax_t1.get_legend_handles_labels()
    lines_2, labels_2 = ax_t1_deg.get_legend_handles_labels()
    ax_t1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    # Subplot 3B: Direct Linearity Transformation Mapping
    ax_t2.scatter(df['pan_pixel_error'], df['pan_deg_error'], color='#1f77b4', alpha=0.6, s=12, label='Pixel-to-Degree Samples')
    ax_t2.set_xlabel('Pan Pixel Error (pixels)')
    ax_t2.set_ylabel('Pan Degree Error (degrees)')
    ax_t2.set_title('Pixel-to-Degree Scale Linearity Map', fontsize=11, pad=8)
    ax_t2.grid(True)
    ax_t2.legend(loc='upper left', frameon=True, facecolor='#ffffff', edgecolor='#dddddd')

    plt.suptitle('Transformation Performance: Pixel Error to Degree Translation', fontsize=14, y=0.98)
    plt.tight_layout()
    fig_trans.savefig('pan_transformation_analysis.png', dpi=300)
    plt.close()

    print("Processing completed successfully. 3 tracking visual outputs saved.")

if __name__ == '__main__':
    generate_tracking_plots('logs/target_telemetry.csv')