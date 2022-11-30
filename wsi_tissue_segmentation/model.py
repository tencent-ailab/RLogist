import os
import tensorflow
if tensorflow.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf
import math
import cv2
import numpy as np


class TfModel(object):
    def __init__(self, model_path, gpu_id=0):
        self.sess = None
        self.gpu_memory_fraction = 0.5
        self.input_name = 'input_img:0'
        self.output_name = 'output_1:0'

        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        self.input_tensor = None
        self.output_tensor = None

        self.model_path = model_path
        self.init_model()

    def init_model(self):
        with tf.device('/device:GPU:0'):
            with tf.Graph().as_default():
                graph_def = tf.GraphDef()
                with open(self.model_path, 'rb') as f:
                    first_line = f.readline()[:10]
                    if b'version' in first_line:
                        model_string = f.read()
                    else:
                        f.seek(0)
                        model_string = f.read()
                graph_def.ParseFromString(model_string)
                graph = tf.import_graph_def(graph_def, name='')

                with tf.Session(
                        graph=graph,
                        config=self.config).as_default() as self.sess:
                    self.input_tensor = self.sess.graph.get_tensor_by_name(
                        self.input_name)
                    self.output_tensor = self.sess.graph.get_tensor_by_name(
                        self.output_name)

    def release_model(self):
        self.sess.close()
        # cuda.select_device(0)
        # cuda.close()

    def inference(self, image_bgr):
        h, w = image_bgr.shape[:2]
        h_align = int(math.ceil(h / 32) * 32)
        w_align = int(math.ceil(w / 32) * 32)
        image_align = cv2.resize(image_bgr, (w_align, h_align))
        image_rgb = image_align[:, :, ::-1]
        image_input = np.reshape(image_rgb, (1,) + image_rgb.shape)

        predicts = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: image_input})
        predicts = (predicts*255).astype(np.uint8)

        predict = cv2.resize(predicts[0, :, :, 0], (w, h), cv2.INTER_NEAREST)
        return predict
