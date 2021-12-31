import cv2
import tensorflow as tf
import numpy as np
import uuid
from pymongo import MongoClient

from movenet.model_utils import (
    init_crop_region,
    run_inference,
    draw_prediction_on_image,
    determine_crop_region
)

def process_vid(user_id, vid, model, input_size):

    in_vid_path = "./cache/vid_in/" + vid.filename
    out_vid_path = "./cache/vid_out/" + vid.filename


    with open(in_vid_path, 'wb') as out_file:
        content = vid.file.read()  # async read
        out_file.write(content)  # async write
    unique_id = uuid.uuid4()
    vidcap = cv2.VideoCapture(in_vid_path)
    success,image = vidcap.read()
    count = 0
    image_height, image_width, _ = image.shape
    crop_region = init_crop_region(image_height, image_width)

    path_out = out_vid_path
    fps = 30.0
    out = cv2.VideoWriter(path_out,cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_width, image_height))
    keypoint_list = []
    
    while success:

        image = tf.io.encode_jpeg(image)
        image = tf.image.decode_jpeg(image)

        # Run model inference.
        keypoints_with_scores = run_inference(
        model, image, crop_region,
        crop_size=[input_size, input_size])

        # Save point data to MongoDB
        keypoint_list.append(keypoints_with_scores.tolist())

        # Visualize the predictions with image.
        output_overlay = draw_prediction_on_image(
        image.numpy().astype(np.int32),
        keypoints_with_scores, crop_region=None,
        close_figure=True, output_image_height=image_height)

        crop_region = determine_crop_region(
        keypoints_with_scores, image_height, image_width)
        
        # Write frame to new video
        out.write(output_overlay)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
            
    out.release()
    save_data(user_id, keypoint_list, unique_id)

    return out_vid_path

def save_data(user_id, keypoints_with_scores, unique_id):

    
    client = MongoClient()
    db = client['keypoints']
    keypoints = db.keypoints

    keypoints_with_scores = keypoints_with_scores
    save_data = {"user_id":user_id, "keypoints": keypoints_with_scores}
    keypoint = keypoints.insert_one(save_data).inserted_id

    print(keypoint)

    return True