import glob

import cv2
import os
from deepface import DeepFace
import numpy as np
from PIL import Image
import shutil
import pandas as pd

sample_rate = 50

MAX_TRAIN_IMAGES = 1000

# auxiliary methods

def trim_dirs():
    # go over face dirs
    data_root = my_data_root
    out_root = my_out_root
    dirs = os.listdir(data_root)
    for dir in dirs:
        if os.path.isdir(os.path.join(data_root,dir)):
            print(f'processing dir {dir}')
            ims = glob.glob('*.png', root_dir=os.path.join(data_root,dir))
            print(f'dir has {len(ims)} images')
            if len(ims) >= MAX_TRAIN_IMAGES:
                os.makedirs(os.path.join(out_root,dir),exist_ok=True)
                for i in range(MAX_TRAIN_IMAGES):
                    shutil.copy(os.path.join(data_root,dir,ims[i]),os.path.join(out_root,dir,ims[i]))


# main methods
def frame_capture(in_path,ep):
    # Path to video file
    vidObj = cv2.VideoCapture(in_path)
    out_path, fname = os.path.split(in_path)
    frames_path = os.path.join(out_path, fname + "_frames")
    print(frames_path)
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    else:
        print("frames exist, skipping")
        return frames_path

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:
        success, image = vidObj.read()
        if count % sample_rate == 0:
            imname = f'{ep}frame{count}.jpg'
            # Saves the frames with frame-count
            try:
                cv2.imwrite(os.path.join(frames_path, imname), image)
            except:
                print("failed to write frame")
            print(imname)

        count += 1

    return frames_path


def find_faces(gallery_path, db_path):
    # we have a gallery of the main characters
    # for each face in the gallery we search through the extracted faces
    # and take all similar faces
    margin = 10
    gallery_faces = os.listdir(gallery_path)
    for face_im in gallery_faces:
        if os.path.isfile(os.path.join(gallery_path, face_im)):
            # create results directory
            face_name = face_im.split('.')[0]
            res_dir_name = os.path.join(gallery_path, face_name)
            os.makedirs(res_dir_name, exist_ok=True)
            # go over all image directories
            eps = os.listdir(db_path)
            for ep in eps:
                if os.path.isdir(os.path.join(db_path,ep)):
                    dirs = os.listdir(os.path.join(db_path,ep))
                    # make sure this is a frames dir
                    for content in dirs:
                        if content.endswith('_frames'):
                            cur_path = os.path.join(db_path,ep,content)
                            print(f'looking for face {face_name} in {cur_path}')
                            # find all faces in db dir
                            dfs = DeepFace.find(img_path=os.path.join(gallery_path, face_im),
                                                db_path=cur_path,
                                                model_name='ArcFace',
                                                detector_backend='retinaface',
                                                threshold=0.5,
                                                enforce_detection=False)
                            # assume we have only 1 face in the gallery image
                            if len(dfs) != 1:
                                raise ValueError('DeepFace.find did not return 1 df')

                            for index, row in dfs[0].iterrows():
                                if index == 0:
                                    source_x, source_y, source_w, source_h = row['source_x'], row['source_y'], row['source_w'], row[
                                        'source_h']
                                    # Crop source image with margin
                                    source_image = Image.open(os.path.join(gallery_path, face_im))
                                    source_crop_coordinates = (
                                        max(0, source_x - margin),
                                        max(0, source_y - margin),
                                        min(source_image.width, source_x + source_w + margin),
                                        min(source_image.height, source_y + source_h + margin)
                                    )
                                    source_cropped = source_image.crop(source_crop_coordinates)
                                    source_output_path = os.path.join(res_dir_name, "source.png")
                                    source_cropped.save(source_output_path)

                                target_image_path = row['identity']
                                # Open target image
                                target_image = Image.open(target_image_path)
                                # Extract coordinates
                                target_x, target_y, target_w, target_h = row['target_x'], row['target_y'], row['target_w'], row[
                                    'target_h']
                                # Crop target image with margin
                                target_crop_coordinates = (
                                    max(0, target_x - margin),
                                    max(0, target_y - margin),
                                    min(target_image.width, target_x + target_w + margin),
                                    min(target_image.height, target_y + target_h + margin)
                                )
                                target_cropped = target_image.crop(target_crop_coordinates)
                                # Save cropped images
                                target_fname = os.path.basename(target_image_path)
                                dist = row['distance']
                                target_output_path = (
                                    os.path.join(res_dir_name, f"{target_fname}_{index}_{dist:.2f}.png"))
                                target_cropped.save(target_output_path)



if __name__ == '__main__':

    print('video_extract_faces')

    # data_root = my_data_root
    # seasons = ['season1', 'season2', 'season3', 'season4', 'season5',
    #            'season6', 'season7', 'season8', 'season9', 'season10',
    #            'season11', 'season12']
    # for seas in seasons:
    #     episodes = os.listdir(os.path.join(data_root,seas))
    #     for ep in episodes:
    #         vf = os.listdir(os.path.join(data_root,seas,ep))
    #         video_file_path = os.path.join(data_root, seas, ep, vf[0])
    #         frames_path = frame_capture(video_file_path,ep)
    #
    # gallery_path = my_gallery_path
    # for seas in seasons:
    #     find_faces(gallery_path, os.path.join(data_root,seas))

    # trim_dirs()

