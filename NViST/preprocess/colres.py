import subprocess
import os
import glob
import argparse

def main():
    """
    Processes scene directories in the MVImgNet dataset by reading COLMAP results where applicable.

    This script iterates over category directories within the base MVImgNet directory. For each scene
    directory that contains a 'sparse/0' directory but no JSON files in 'camera_new/', it triggers
    the `preprocess.read_colmap_results` module to process the COLMAP results for that scene.
    """

    arg_parser = argparse.ArgumentParser(description="Wrapper for MVImgNet")
    arg_parser.add_argument("--base_dir", default="./data/", help="category")
    args = arg_parser.parse_args()
    base_dir = args.base_dir

    catdirs = os.listdir(base_dir)
    catdirs = [catdir for catdir in catdirs if os.path.isdir(os.path.join(base_dir, catdir))]

    for catdir in catdirs:
        scenedirs = os.listdir(os.path.join(base_dir, catdir))
        scenedirs = [scenedir for scenedir in scenedirs if os.path.isdir(os.path.join(base_dir, catdir, scenedir))]
        
        for scenedir in scenedirs:
            sparse_dir = os.path.join(base_dir, catdir, scenedir, 'sparse/0')
            json_files = glob.glob(os.path.join(base_dir, catdir, scenedir, 'camera_new/*.json'))
            
            if os.path.exists(sparse_dir) and not json_files:
                print(f'Processing {catdir} {scenedir}...')
                cmd = [
                    'python', 
                    '-m', 
                    'preprocess.read_colmap_results', 
                    '--base_dir', os.path.join(base_dir, catdir), 
                    '--capture_name', scenedir
                ]
                try:
                    result = subprocess.run(cmd, check=True)
                    if result.returncode != 0:
                        print(f"Error processing {catdir} {scenedir}. Return code: {result.returncode}")
                except subprocess.CalledProcessError as e:
                    print(f"Subprocess failed for {catdir} {scenedir}: {e}")
            else: 
                print(f'Already processed or no data for {catdir} {scenedir}')

if __name__ == "__main__":
    main()
