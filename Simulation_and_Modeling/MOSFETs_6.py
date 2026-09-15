import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

def parse_segmented_blocks(file_path):
    """
    Parses the parameter analyzer .DAT file dynamically line by line,
    isolating independent measurement data arrays from text headers.
    """
    blocks = []
    current_data_rows = []
    is_reading_data = False
    
    with open(file_path, 'r') as f:
        for line in f:
            line_str = line.strip()
            
            # Identify when a data table starts
            if line_str.startswith("VD,ID,VG"):
                is_reading_data = True
                current_data_rows = []
                continue
            
            # Identify when a data block ends (blank line or next structural block)
            if is_reading_data and (not line_str or "TEST RESULTS" in line_str or "SETUP:" in line_str):
                if current_data_rows:
                    # Convert the gathered rows into a standalone DataFrame
                    df_block = pd.DataFrame([r.split(',') for r in current_data_rows])
                    blocks.append(df_block)
                is_reading_data = False
                continue
                
            if is_reading_data:
                current_data_rows.append(line_str)
                
        # Catch a remaining block at EOF if any
        if is_reading_data and current_data_rows:
            df_block = pd.DataFrame([r.split(',') for r in current_data_rows])
            blocks.append(df_block)

    # Reassemble and enforce numerical values column by column
    cleaned_dfs = []
    columns_mapping = {0: 'VD', 1: 'ID', 2: 'VG'}
    
    for b in blocks:
        # Keep only the columns we care about
        b = b[[0, 1, 2]].rename(columns=columns_mapping)
        
        for col in ['VD', 'ID', 'VG']:
            # Strip trailing asterisks (*) or formatting symbols
            b[col] = b[col].astype(str).str.replace(r'[^\d.E+-]', '', regex=True)
            b[col] = pd.to_numeric(b[col], errors='coerce')
            
        b = b.dropna(subset=['VD', 'ID', 'VG'])
        cleaned_dfs.append(b)
        
    return pd.concat(cleaned_dfs, ignore_index=True)

def filter_outliers_iqr(df):
    """
    Applies statistical filtering grouped cleanly by uniform bias conditions.
    """
    filtered_chunks = []
    
    # Precise rounding prevents float precision differences from splitting groups
    df['VD_round'] = df['VD'].round(3)
    df['VG_round'] = df['VG'].round(3)
    
    for _, group in df.groupby(['VD_round', 'VG_round']):
        if len(group) > 2:  # Statistical boundaries require a meaningful distribution size
            q1 = group['ID'].quantile(0.25)
            q3 = group['ID'].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            clean_group = group[(group['ID'] >= lower_bound) & (group['ID'] <= upper_bound)]
            filtered_chunks.append(clean_group)
        else:
            filtered_chunks.append(group)
            
    return pd.concat(filtered_chunks, ignore_index=True)

def export_plots_to_png(df):
    # Establish average curve mapping from the clean data points
    mean_df = df.groupby(['VG_round', 'VD_round'])['ID'].mean().reset_index()
    
    # Set global font styles for professional appearance
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    
    # Marker cycle for consistency
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*']
    
    # =========================================================================
    # EXPORT 1: ID vs VDS (Output Characteristics) - Single Overlaid PNG
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    output_data = mean_df.groupby('VG_round').filter(lambda x: len(x['VD_round'].unique()) > 5)
    
    for i, (vg, group) in enumerate(output_data.groupby('VG_round')):
        group = group.sort_values('VD_round')
        marker = markers[i % len(markers)]
        
        # FIXED: Removed literal LaTeX inside f-strings to completely evade variable interpolation errors
        line_label = "$V_{GS}$ = " + f"{vg:.1f} V"
        
        ax1.plot(group['VD_round'], group['ID'], 
                 marker=marker, markersize=5, linestyle='-', linewidth=1.5,
                 label=line_label)
        
    ax1.set_title('Output Characteristics ($I_D$ vs $V_{DS}$)', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('$V_{DS}$ [V]', fontsize=11)
    ax1.set_ylabel('$I_D$ [A]', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', fontsize=9)
    fig1.tight_layout()
    
    # Save the output characteristics plot as a PNG file
    output_png_name = 'output_characteristics_Id_vs_Vds.png'
    fig1.savefig(output_png_name, dpi=300)
    plt.close(fig1)
    print(f"Saved: {output_png_name}")
    
    # =========================================================================
    # EXPORT 2: ID vs VGS (Transfer Characteristics) - Dedicated PNG per curve
    # =========================================================================
    transfer_data = mean_df.groupby('VD_round').filter(lambda x: len(x['VG_round'].unique()) > 5)
    
    for i, (vd, group) in enumerate(transfer_data.groupby('VD_round')):
        group = group.sort_values('VG_round')
        marker = markers[i % len(markers)]
        
        # Open a completely fresh figure context for each individual transfer sweep
        fig_trans, ax_trans = plt.subplots(figsize=(6.5, 5.5))
        
        ax_trans.plot(group['VG_round'], group['ID'], 
                marker=marker, markersize=5, linestyle='-', linewidth=1.5, color='darkblue')
        
        # FIXED: Isolated string construction safely away from string formatting rules
        title_text = "Transfer Characteristics ($I_D$ vs $V_{GS}$) at $V_{DS}$ = " + f"{vd:.1f} V"
        
        ax_trans.set_title(title_text, fontsize=11, fontweight='bold', pad=10)
        ax_trans.set_xlabel('$V_{GS}$ [V]', fontsize=11)
        ax_trans.set_ylabel('$I_D$ [A]', fontsize=11)
        ax_trans.grid(True, linestyle=':', alpha=0.6, color='gray')
        fig_trans.tight_layout()
        
        # Save this specific linear transfer sweep curve to its own PNG file
        transfer_png_name = f'transfer_curve_Vds_{vd:.1f}_V.png'
        fig_trans.savefig(transfer_png_name, dpi=300)
        plt.close(fig_trans)
        print(f"Saved: {transfer_png_name}")

# --- Execution ---
file_path = 'MOSFET 10 x100.DAT'

try:
    master_df = parse_segmented_blocks(file_path)
    cleaned_df = filter_outliers_iqr(master_df)
    export_plots_to_png(cleaned_df)
    print("\nAll tasks successfully completed.")
except Exception as e:
    print(f"Error executing script: {e}")