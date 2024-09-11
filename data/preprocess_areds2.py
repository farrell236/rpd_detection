import argparse
import cv2
import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from utils import _pad_to_square, _get_retina_bb, rgb_clahe

# import matplotlib.pyplot as plt
# import multiprocessing as mp


def process_row(row):

    # Load Image
    image_file = os.path.join(data_root, row['filepath_cfp'])
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image); plt.show()

    # Localise and center retina image
    x, y, w, h, mask = _get_retina_bb(image)
    image = image[y:y + h, x:x + w, :]
    image = _pad_to_square(image, border=0)
    image = cv2.resize(image, (image_res, image_res))

    # Center retina mask
    mask = np.uint8(mask[..., None] > 0)
    mask = mask[y:y + h, x:x + w, :]
    mask = _pad_to_square(mask, border=0)
    mask = cv2.resize(mask, (image_res, image_res), 0, 0, cv2.INTER_NEAREST)

    # Apply CLAHE pre-processing
    image = rgb_clahe(image)

    # Display or save image
    # plt.imshow(image); plt.show()
    # plt.imshow(mask); plt.show()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Write image and mask to disk
    rDIR_PATH = row['filepath_cfp'].rsplit('/', 1)[0]
    os.makedirs(os.path.join(pp_root, 'images', rDIR_PATH), exist_ok=True)
    os.makedirs(os.path.join(pp_root, 'masks', rDIR_PATH), exist_ok=True)
    cv2.imwrite(os.path.join(pp_root, 'images', row['filepath_cfp']), image)
    cv2.imwrite(os.path.join(pp_root, 'masks', row['filepath_cfp']), mask)


def main(df):

    # Convert DataFrame rows to a list of Series, each representing a row
    rows = [row[1] for row in df.iterrows()]

    # Disabled as multiprocessing does not seem to play nice with SimpleITK
    # TODO: leaving here as a future fix
    # # Create a pool of workers, the number of workers is typically set to the number of cores
    # pool = mp.Pool(mp.cpu_count())
    #
    # # Process each row in parallel
    # for _ in tqdm(pool.imap_unordered(process_row, rows), total=len(rows)):
    #     pass  # Just consume the iterator to update the tqdm progress bar
    #
    # # Close the pool and wait for the work to finish
    # pool.close()
    # pool.join()

    for row in tqdm(rows):
        try:
            process_row(row)
        except Exception as e:
            print('IMAGE NOT PROCESSED:', row['filepath_cfp'])
            print(e)


if __name__ == '__main__':

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a part of a DataFrame.")
    parser.add_argument('csvfile', type=str, help="AREDS csv file to process")
    parser.add_argument("part_num", type=int, default=1, help="The part number to process (1-indexed).")
    parser.add_argument("total_parts", type=int, default=12, help="The total number of parts to divide the DataFrame into.")
    parser.add_argument("output_res", type=int, default=2048, help="Resample image to standardised resolution.")
    parser.add_argument("data_root", type=str, help="Path to AREDS (original) images.")
    parser.add_argument("pp_root", type=str, help="Path to store pre-processed images.")
    args = parser.parse_args()

    # Setup parameters
    data_root = args.data_root
    pp_root = args.pp_root
    image_res = args.output_res

    # Load up train/valid/test
    data_df = pd.read_csv(args.csvfile)  # '/path/to/AREDS/[train/valid/test].csv'
    data_df['filepath_cfp'] = data_df['filepath_cfp'].str.replace('dcm', 'jpeg')

    # Calculate the number of rows in each part
    total_rows = len(data_df)
    part_size = total_rows // args.total_parts
    remainder = total_rows % args.total_parts

    # Calculate the start and end indices for the slice
    start = (args.part_num - 1) * part_size + min(args.part_num - 1, remainder)
    end = start + part_size + (1 if args.part_num <= remainder else 0)

    main(data_df.iloc[start:end])
