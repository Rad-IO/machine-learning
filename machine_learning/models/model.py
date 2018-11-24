import tensorflow as tf


class Model:
    """Class representing a machine learning model used in classification.

    Parameters
    ----------
    graph_def: tf.GraphDef
        With the definition of the model.
    labels: iterable of str
        With the labels the model is going to return.
    name: str
        The name of the model. Preferably, it should be what this model is
        supposed to classify.

    """
    def __init__(self, graph_def, labels, name):
        self._labels = labels
        self._graph_def = graph_def
        self.name = name

    @staticmethod
    def load_from_pb(pb_file, labels, name):
        """Load model from a .pb file.

        Parameters
        ----------
        pb_file: str
            Path to the pb file defining the model's parameters.
        labels: iterable of str
            With the labels used by our model.
        name: str
            The name of the model. Preferably, it should be what this model is
            supposed to classify.

        Returns
        -------
        machine_learning.models.Model
            With the initialised data.

        """
        with tf.gfile.GFile(pb_file, 'rb') as stream:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(stream.read())

        return Model(graph_def, labels, name)

    def predict(self, image):
        """Run prediction on an input image.

        Parameters
        ----------
        image: tf.gfile.GFile
            The data stream of the image.

        Returns
        ------
        dict (str -> float)
            With the prediction results, matching labels to probabilities.

        """
        result = {}
        img_data = image.read()

        tf.import_graph_def(self._graph_def, name='')

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': img_data})

            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            for node_id in top_k:
                label = self._labels[node_id]
                score = predictions[0][node_id]

                result[label] = score

        return result

    def __str__(self):
        return self.name
