import os
import glob
import h5py

def stitch_svi_files(target_dir, satellite="j01", t_code="t2053"):
    print(f"Scanning {target_dir} for individual SVI files (sat={satellite}, t={t_code})...")

    # Find all SVI files in the target directory
    svi_files = {}
    for i in range(1, 6):
        pattern = os.path.join(target_dir, f"SVI0{i}_{satellite}_*_{t_code}*.h5")
        matches = glob.glob(pattern)
        if matches:
            svi_files[f"SVI0{i}"] = matches[0]
        else:
            print(f"Warning: Could not find SVI0{i} file in {target_dir}")
            return None
            
    if len(svi_files) != 5:
        print("Error: Missing one or more SVI files to stitch!")
        return None
        
    print("Found all 5 SVI bands. Preparing to stitch...")
    
    # The GIMGO filename traditionally mimics the timestamps of the original files.
    # We will pull the timestamp metadata from the SVI01 filename to generate it.
    svi01_name = os.path.basename(svi_files["SVI01"])
    # Example: SVI01_j02_d20240314_t0009433_e0011080_...
    parts = svi01_name.split('_')
    
    # Reconstruct the official NOAA-style GIMGO filename
    gimgo_name = f"GIMGO-SVI01-SVI02-SVI03-SVI04-SVI05_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}_stitched.h5"
    gimgo_path = os.path.join(target_dir, gimgo_name)
    
    print(f"\nCreating stitched file: {gimgo_name}")
    
    with h5py.File(gimgo_path, 'w') as f_out:
        # The viirs_training_loader expects everything inside the 'All_Data' group
        all_data_group = f_out.create_group('All_Data')
        
        for band_id in sorted(svi_files.keys()):
            src_path = svi_files[band_id]
            band_group_name = f"VIIRS-I{band_id[-1]}-SDR_All"
            
            print(f"  -> Copying {band_group_name} from {os.path.basename(src_path)}...")
            with h5py.File(src_path, 'r') as f_in:
                if 'All_Data' in f_in and band_group_name in f_in['All_Data']:
                    src_group = f_in['All_Data'][band_group_name]
                    # Copy the entire group (arrays, factors, attributes) into our new file
                    f_in.copy(src_group, all_data_group)
                else:
                    print(f"Error: Could not find {band_group_name} inside {src_path}")
                    
    print(f"\nSuccess! All bands stitched into: {gimgo_path}")
    print("This file can now be loaded directly by your team's viirs_training_loader.py!")
    return gimgo_path

if __name__ == "__main__":
    stitch_svi_files("viirs_data", satellite="j01", t_code="t2144")
