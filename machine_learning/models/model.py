import tensorflow as tf


class Model:
    def __init__(self, graph_def, labels):
        self._labels = labels
        self._graph_def = graph_def

    @staticmethod
    def load_from_pb(pb_file, labels):
        """

        :param pb_file:
        :param labels:
        :return:
        """
        label_lines = [
            line.rstrip() for line in tf.gfile.GFile(labels),
        ]

        with tf.gfile.GFile(pb_file, 'rb') as stream:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(stream.read())

        return Model(graph_def, label_lines)

    def predict(self, image):
        img_data = tf.gfile.GFile(image, 'rb').read()

        tf.import_graph_def(self._graph_def, name='')

        with tf.Session() as sess:
            pass
