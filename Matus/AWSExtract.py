import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from datetime import datetime

# The VIIRS product prefixes in the NOAA SNPP bucket
PREFIXES = {
    'SVI01': 'VIIRS-I1-SDR',
    'SVI02': 'VIIRS-I2-SDR',
    'SVI03': 'VIIRS-I3-SDR',
    'SVI04': 'VIIRS-I4-SDR',
    'SVI05': 'VIIRS-I5-SDR',
    'GITCO': 'VIIRS-IMG-GEO-TC'
}

BUCKET_NAME = 'noaa-nesdis-n20-pds'  # J01

def download_viirs_data(target_date_str, target_utc_hour, download_dir='./viirs_data', max_files_per_band=1):
    """
    Downloads VIIRS I-Bands and GITCO files from AWS S3.
    
    :param target_date_str: 'YYYY-MM-DD'
    :param target_utc_hour: e.g. '_t17' string to filter
    :param download_dir: Local directory to save files
    :param max_files_per_band: How many files to grab (VIIRS takes pictures in ~1 minute sequential strips)
    """
    os.makedirs(download_dir, exist_ok=True)
    print(f"Connecting to S3 anonymously...")
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Parse date to build the S3 folder path (YYYY/MM/DD)
    date_obj = datetime.strptime(target_date_str, '%Y-%m-%d')
    date_path = f"{date_obj.strftime('%Y')}/{date_obj.strftime('%m')}/{date_obj.strftime('%d')}"
    
    for band_name, prefix in PREFIXES.items():
        s3_search_prefix = f"{prefix}/{date_path}/"
        print(f"\nSearching for {band_name} in: s3://{BUCKET_NAME}/{s3_search_prefix}")
        
        try:
            paginator = s3.get_paginator('list_objects_v2')
            matched_files = []
            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_search_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if target_utc_hour in obj['Key']:
                            matched_files.append(obj)
            
            if not matched_files:
                print(f" -> No files matched '{target_utc_hour}' for this band.")
                continue
                
            count = 0
            for file_info in matched_files:
                if count >= max_files_per_band:
                    break
                    
                s3_key = file_info['Key']
                filename = os.path.basename(s3_key)
                local_path = os.path.join(download_dir, filename)
                
                # Download Logic
                if os.path.exists(local_path):
                    print(f" -> {filename} already exists. Skipping.")
                else:
                    print(f" -> Downloading: {filename} ({file_info['Size']/1024/1024:.1f} MB)")
                    s3.download_file(BUCKET_NAME, s3_key, local_path)
                count += 1
        except Exception as e:
            print(f"Error while processing {band_name}: {e}")

if __name__ == "__main__":
    # Execute the downloader
    TARGET_DATE = '2025-10-11'
    TARGET_HOUR = '_t2144'

    download_viirs_data(TARGET_DATE, TARGET_HOUR, download_dir='./viirs_data', max_files_per_band=1)
