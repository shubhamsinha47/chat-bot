
from os import path

from vocabulary_importers.checkpoint_vocabulary_importer import CheckpointVocabularyImporter
from vocabulary import Vocabulary

class NnlmEnVocabularyImporter(CheckpointVocabularyImporter):
    """Importer implementation for the nnlm english vocabulary
    """
    def __init__(self):
        super(NnlmEnVocabularyImporter, self).__init__("nnlm_en", "tokens.txt", "embeddings")
    
    def _process_token(self, token):

        if token == "<S>":
            token = Vocabulary.SOS
        elif token == "</S>":
            token = Vocabulary.EOS
        elif token == "<UNK>":
            token = Vocabulary.OUT
        elif token == "--":
            token = Vocabulary.PAD

        return token