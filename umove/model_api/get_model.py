import tensorflow as tf
import tensorflow_hub as hub

class Model:
    
    def __init__(self, model_name="movenet_thunder", input_size=256):   
        self.model_name =  model_name
        self.module = hub.load("http://tfhub.dev/google/movenet/singlepose/thunder/4")
        self.input_size = input_size

    def movenet(self, input_image):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """
        model = self.module.signatures['serving_default']

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores