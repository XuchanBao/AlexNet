import numpy as np
import tensorflow as tf

from model import AlexNet


PARAM_NAME_MAPPING = {
    "L1_filter:0": "conv1/weights:0",
    "B1:0": "conv1/biases:0",
    "L2_filter:0": "conv2/weights:0",
    "B2:0": "conv2/biases:0",
    "L3_filter:0": "conv3/weights:0",
    "B3:0": "conv3/biases:0",
    "L4_filter:0": "conv4/weights:0",
    "B4:0": "conv4/biases:0",
    "L5_filter:0": "conv5/weights:0",
    "B5:0": "conv5/biases:0",
    "fully_connected/weights:0": "fc6/weights:0",
    "fully_connected/biases:0": "fc6/biases:0",
    "fully_connected_1/weights:0": "fc7/weights:0",
    "fully_connected_1/biases:0": "fc7/biases:0",
    "fully_connected_2/weights:0": "fc8/weights:0",
    "fully_connected_2/biases:0": "fc8/biases:0"
}
alexnet = AlexNet(None, 128, True)

alexnet.build_graph()

saver = tf.train.Saver()


def save_model_to_np(epoch_num):
    param_dict = {}
    model_dir = "model/permanent"
    with tf.Session() as sess:
        alexnet.restore_model(sess, saver, model_dir, 'epoch_{}.ckpt'.format(epoch_num))
        for param in tf.trainable_variables():
            print(param)
            mapped_name = PARAM_NAME_MAPPING[param.name].split('/')[0]

            if mapped_name not in param_dict:
                param_dict[mapped_name] = []
            param_value = sess.run(param)
            param_dict[mapped_name].append(param_value)

    save_path = "{}/epoch_{}.npy".format(model_dir, epoch_num)
    np.save(save_path, param_dict)
    print("Saved numpy model to {}".format(save_path))


for epoch in [0, 5, 10, 15, 20, 25]:
    save_model_to_np(epoch_num=epoch)
