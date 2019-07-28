"""
Base class for TensorFlow Checkpoint vocabulary importers
"""
import tensorflow as tf
from collections import OrderedDict
from os import path
from vocabulary_importers.vocabulary_importer import VocabularyImporter

class CheckpointVocabularyImporter(VocabularyImporter):
    

    def __init__(self, vocabulary_name, tokens_filename, embeddings_variable_name):
        super(CheckpointVocabularyImporter, self).__init__(vocabulary_name)
        

        self.tokens_filename = tokens_filename

        self.embeddings_variable_name = embeddings_variable_name

    def _read_vocabulary_and_embeddings(self, vocabulary_dir):
        
        tf.reset_default_graph()
        embeddings = tf.Variable(tf.contrib.framework.load_variable(vocabulary_dir, self.embeddings_variable_name), name = "embeddings")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loaded_embeddings_matrix = sess.run(embeddings)

        #Import vocabulary
        tokens_filepath = path.join(vocabulary_dir, self.tokens_filename)
        tokens_with_embeddings = OrderedDict()
        with open(tokens_filepath, encoding="utf-8") as file:
            for index, line in enumerate(file):
                token = line.strip()
                if token != "":
                    token = self._process_token(token)
                    tokens_with_embeddings[token] = loaded_embeddings_matrix[index]

        return tokens_with_embeddings