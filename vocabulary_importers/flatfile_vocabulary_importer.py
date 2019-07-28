"""
Base class for Flat File vocabulary importers
"""
import numpy as np
from collections import OrderedDict
from os import path
from vocabulary_importers.vocabulary_importer import VocabularyImporter

class FlatFileVocabularyImporter(VocabularyImporter):
    """Base class for Flat File vocabulary importers
    """

    def __init__(self, vocabulary_name, tokens_and_embeddings_filename, delimiter):
        super(FlatFileVocabularyImporter, self).__init__(vocabulary_name)
        

        self.tokens_and_embeddings_filename = tokens_and_embeddings_filename

        self.delimiter = delimiter

    def _read_vocabulary_and_embeddings(self, vocabulary_dir):
       
        
        tokens_and_embeddings_filepath = path.join(vocabulary_dir, self.tokens_and_embeddings_filename)
        tokens_with_embeddings = OrderedDict()
        with open(tokens_and_embeddings_filepath, encoding="utf-8") as file:
            for _, line in enumerate(file):
                values = line.split(self.delimiter)
                token = values[0].strip()
                if token != "":
                    token = self._process_token(token)
                    tokens_with_embeddings[token] = np.array(values[1:], dtype=np.float32)

        return tokens_with_embeddings