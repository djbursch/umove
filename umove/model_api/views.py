import cv2
import tensorflow as tf
import numpy as np
from wsgiref.util import FileWrapper

from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view

from .models import Video
from .get_model import Model
from .model_utils import (
    init_crop_region,
    run_inference,
    draw_prediction_on_image,
    determine_crop_region
)

model = Model()
input_size = model.input_size

@api_view(["POST"])
def run_model(request):

    video = Video(video_name=request.data.get('video_name'), video=request.data.get('video'))
    video.save()

    vidcap = cv2.VideoCapture('./static_cache/sent_vid/' + request.data.get('video').name)
    success,image = vidcap.read()
    count = 0
    image_height, image_width, _ = image.shape
    crop_region = init_crop_region(image_height, image_width)

    path_out = './static_cache/out_vid/' + request.data.get('video').name
    fps = 30.0
    out = cv2.VideoWriter(path_out,cv2.VideoWriter_fourcc(*'mp4v'), fps, (image_width, image_height))

    
    while success:

        image = tf.io.encode_jpeg(image)
        image = tf.image.decode_jpeg(image)

        # Run model inference.
        keypoints_with_scores = run_inference(
        model, image, crop_region,
        crop_size=[input_size, input_size])

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
    
    file = FileWrapper(open(path_out, 'rb'))
    response = HttpResponse(file, content_type='video/mp4')
    response['Content-Disposition'] = 'attachment; filename=my_video.mp4'
    return response
