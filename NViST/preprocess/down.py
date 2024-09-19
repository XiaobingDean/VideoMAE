import os
from PIL import Image
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='Downsample images')
    parser.add_argument("-d", "--data_dir", type=str, default="../data", help="where the dataset is")
    parser.add_argument("-r", "--multi_resolution", type=bool, default=False, help="if yes, we also downsample images by 3 and 6")
    args = parser.parse_args()

    dirnames = os.listdir(args.data_dir)[::-1]
    dirnames = [dirname for dirname in dirnames if os.path.isdir(os.path.join(args.data_dir, dirname))]

    for dirname in dirnames:
        scenenames = sorted(os.listdir(os.path.join(args.data_dir, dirname)))
        for scenename in scenenames:
            imgfiles = sorted(glob.glob(os.path.join(args.data_dir, dirname, scenename, 'images', '*.jpg')))
            savedir_12 = os.path.join(args.data_dir, dirname, scenename, 'images_12')
            os.makedirs(savedir_12, exist_ok=True)
            
            if args.multi_resolution:
                savedir_3 = os.path.join(args.data_dir, dirname, scenename, 'images_3')
                savedir_4 = os.path.join(args.data_dir, dirname, scenename, 'images_4')
                savedir_6 = os.path.join(args.data_dir, dirname, scenename, 'images_6')
                savedir_128 = os.path.join(args.data_dir, dirname, scenename, 'images_128')
                os.makedirs(savedir_3, exist_ok=True)
                os.makedirs(savedir_4, exist_ok=True)
                os.makedirs(savedir_6, exist_ok=True)
                os.makedirs(savedir_128, exist_ok=True)

            if len(imgfiles) > 0:
                print(dirname, scenename, savedir_12)
                for imgfile in imgfiles:
                    if imgfile.endswith(".jpg") or imgfile.endswith(".png"):
                        fname = os.path.basename(imgfile)  # Only get the filename
                        savepath_12 = os.path.join(savedir_12, fname)  # Correctly join directory and filename
                        
                        if not os.path.isfile(savepath_12):
                            with Image.open(imgfile) as img:
                                width, height = img.size
                                scale = 160 / max(width, height)
                                new_width, new_height = int(width * scale), int(height * scale)
                                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                                resized_img.save(savepath_12)

                        if args.multi_resolution:
                            save_image_in_resolution(imgfile, savedir_3, 640)
                            save_image_in_resolution(imgfile, savedir_4, 480)
                            save_image_in_resolution(imgfile, savedir_6, 320)
                            save_image_in_resolution(imgfile, savedir_128, 128, min_dimension=True)
            else:
                print(f"No images found in {dirname}/{scenename}")

def save_image_in_resolution(imgfile, savedir, resolution, min_dimension=False):
    """
    Helper function to save image in specified resolution.
    """
    savepath = os.path.join(savedir, os.path.basename(imgfile))  # Only get the filename
    if not os.path.isfile(savepath):
        with Image.open(imgfile) as img:
            width, height = img.size
            scale = resolution / (min(width, height) if min_dimension else max(width, height))
            new_width, new_height = int(width * scale), int(height * scale)
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            resized_img.save(savepath)

if __name__ == "__main__":
    main()
