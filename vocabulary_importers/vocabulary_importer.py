"""
Base class for all vocabulary importers
"""
import abc
import numpy as np
from enum import Enum
from collections import OrderedDict
from vocabulary import Vocabulary

class VocabularyImportMode(Enum):
    
    External = 1
    ExternalIntersectDataset = 2
    ExternalUnionDataset = 3
    Dataset = 4

class VocabularyImportStats(object):
    """Contains information about the imported vocabulary.
    """

    def __init__(self):
        self.external_vocabulary_size = None
        self.dataset_vocabulary_size = None
        self.intersection_size = None

class VocabularyImporter(object):
    """Base class for all vocabulary importers
    """

    def __init__(self, vocabulary_name):
        
        self.vocabulary_name = vocabulary_name

    @abc.abstractmethod
    def _process_token(self, token):
        
        pass

    @abc.abstractmethod
    def _read_vocabulary_and_embeddings(self, vocabulary_dir):
        
        pass

    def import_vocabulary(self, vocabulary_dir, normalize = True, import_mode = VocabularyImportMode.External, dataset_vocab = None):
        

        if dataset_vocab is None and import_mode != VocabularyImportMode.External:
            raise ValueError("dataset_vocab must be provided if import_mode is not 'External'.")

        import_stats = VocabularyImportStats()
        
        #Read the external vocabulary tokens and embeddings
        tokens_with_embeddings = self._read_vocabulary_and_embeddings(vocabulary_dir)

        #If normalize flag is true, normalize casing of the external vocabulary and average embeddings for any resulting duplicate tokens
        if normalize:
            tokens_with_embeddings = self._normalize_tokens_with_embeddings(tokens_with_embeddings)
        
        import_stats.external_vocabulary_size = len(tokens_with_embeddings)
        
        #Apply dataset filters if applicable
        if dataset_vocab is not None:
            import_stats.dataset_vocabulary_size = dataset_vocab.size()

            if import_mode == VocabularyImportMode.ExternalIntersectDataset or import_mode == VocabularyImportMode.Dataset:
                #Get rid of all tokens that exist in the external vocabulary but don't exist in the dataset
                for token in list(tokens_with_embeddings.keys()):
                    if not dataset_vocab.word_exists(token):
                        del tokens_with_embeddings[token]
                import_stats.intersection_size = len(tokens_with_embeddings)

            if import_mode == VocabularyImportMode.ExternalUnionDataset or import_mode == VocabularyImportMode.Dataset:
                #Add any tokens that exist in the dataset but don't exist in the external vocabulary.
                #These added tokens will get word vectors sampled from the gaussian distributions of their components:
                #   where the mean of each component is the mean of that component in the external embedding matrix
                #   and the standard deviation of each component is the standard deviation of that component in the external embedding matrix
                embeddings_matrix = np.array(list(tokens_with_embeddings.values()), dtype=np.float32)
                emb_size = embeddings_matrix.shape[1]
                emb_mean = np.mean(embeddings_matrix, axis=0)
                emb_stdev = np.std(embeddings_matrix, axis=0)
                for i in range(dataset_vocab.size()):
                    dataset_token = dataset_vocab.int2word(i, capitalize_i=False)
                    if dataset_token not in tokens_with_embeddings:
                        tokens_with_embeddings[dataset_token] = np.random.normal(emb_mean, emb_stdev, emb_size)

        if len(tokens_with_embeddings) == 0:
            raise ValueError("Imported vocabulary size is 0. Try a different VocabularyImportMode (currently {0})".format(
                VocabularyImportMode(import_mode).name))

        tokens, embeddings_matrix = zip(*tokens_with_embeddings.items())
        embeddings_matrix = np.array(embeddings_matrix, dtype=np.float32)

        #Create the vocabulary instance
        vocabulary = Vocabulary(external_embeddings = embeddings_matrix)
        for i in range(len(tokens)):
            vocabulary.load_word(tokens[i], i)
        vocabulary.compile(loading = True)
        return vocabulary, import_stats

    def _normalize_tokens_with_embeddings(self, tokens_with_embeddings):
        """Convert all word tokens to lower case and then average the embedding vectors for any duplicate words
        """
        norm_tokens_with_embeddings = OrderedDict()
        for token, embedding in tokens_with_embeddings.items():
            if token not in Vocabulary.special_tokens:
                token = token.lower()
            if token in norm_tokens_with_embeddings:
                norm_tokens_with_embeddings[token].append(embedding)
            else:
                norm_tokens_with_embeddings[token] = [embedding]

        for token, embedding in norm_tokens_with_embeddings.items():
            norm_tokens_with_embeddings[token] = np.mean(embedding, axis=0)
        
        return norm_tokens_with_embeddings