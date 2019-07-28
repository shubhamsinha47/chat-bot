"""
Base class for dataset readers
"""
import abc
from os import path

from vocabulary_importers import vocabulary_importer_factory
from vocabulary_importers.vocabulary_importer import VocabularyImportMode
from vocabulary import Vocabulary
from dataset import Dataset

class DatasetReadStats(object):
    """Contains information about the read dataset.
    """

    def __init__(self):
        self.input_vocabulary_import_stats = None
        self.output_vocabulary_import_stats = None

class DatasetReader(object):
    """Base class for dataset readers
    """

    def __init__(self, dataset_name):
        """Initialize the DatasetReader.

        Args:
            dataset_name: Name of the dataset. Subclass must pass this in.
        """
        self.dataset_name = dataset_name

    @abc.abstractmethod
    def _get_dialog_lines_and_conversations(self, dataset_dir):
        
        pass
    
    def read_dataset(self, dataset_dir, model_dir, training_hparams, share_vocab = True, encoder_embeddings_dir = None, decoder_embeddings_dir = None):
        
        if share_vocab:
            if training_hparams.input_vocab_threshold != training_hparams.output_vocab_threshold and (encoder_embeddings_dir is None or training_hparams.input_vocab_import_mode != VocabularyImportMode.External):
                raise ValueError("Cannot share generated or joined imported vocabulary when the input and output vocab thresholds are different.")
            if encoder_embeddings_dir is not None:
                if training_hparams.input_vocab_import_mode != training_hparams.output_vocab_import_mode:
                    raise ValueError("Cannot share imported vocabulary when input and output vocab import modes are different.")
                if training_hparams.input_vocab_import_normalized != training_hparams.output_vocab_import_normalized:
                    raise ValueError("Cannot share imported vocabulary when input and output normalization modes are different.")
            if decoder_embeddings_dir is not None and decoder_embeddings_dir != encoder_embeddings_dir:
                raise ValueError("Cannot share imported vocabulary from two different sources or share import and generated vocabulary.")


        read_stats = DatasetReadStats()

        #Get dialog line and conversation collections
        id2line, conversations_ids = self._get_dialog_lines_and_conversations(dataset_dir)
        
        #Clean dialog lines
        for line_id in id2line:
            id2line[line_id] = Vocabulary.clean_text(id2line[line_id], training_hparams.max_question_answer_words, training_hparams.normalize_words)

        #Output cleaned lines for debugging purposes
        if training_hparams.log_cleaned_dataset:
            self._log_cleaned_dataset(model_dir, id2line.values())

        # Getting separately the questions and the answers
        questions_for_count = []
        questions = []
        answers = []
        for conversation in conversations_ids[:training_hparams.max_conversations]:
            for i in range(len(conversation) - 1):
                conv_up_to_question = ''
                for j in range(max(0, i - training_hparams.conv_history_length), i):
                    conv_up_to_question += id2line[conversation[j]] + " {0} ".format(Vocabulary.EOS)
                question = id2line[conversation[i]]
                question_with_history = conv_up_to_question + question
                answer = id2line[conversation[i+1]]
                if training_hparams.min_question_words <= len(question_with_history.split()):
                    questions.append(conv_up_to_question + question)
                    questions_for_count.append(question)
                    answers.append(answer)

        # Create the vocabulary object & add the question & answer words
        if share_vocab:
            questions_and_answers = []
            for i in range(len(questions_for_count)):
                question = questions_for_count[i]
                answer = answers[i]
                if i == 0 or question != answers[i - 1]:
                    questions_and_answers.append(question)
                questions_and_answers.append(answer)
            
            input_vocabulary, read_stats.input_vocabulary_import_stats = self._create_and_save_vocab(questions_and_answers,
                                                                                  training_hparams.input_vocab_threshold, 
                                                                                  model_dir, 
                                                                                  Vocabulary.SHARED_VOCAB_FILENAME,
                                                                                  encoder_embeddings_dir,
                                                                                  training_hparams.input_vocab_import_normalized,
                                                                                  training_hparams.input_vocab_import_mode)
            output_vocabulary = input_vocabulary
            read_stats.output_vocabulary_import_stats = read_stats.input_vocabulary_import_stats
        else:
            input_vocabulary, read_stats.input_vocabulary_import_stats = self._create_and_save_vocab(questions_for_count, 
                                                                                  training_hparams.input_vocab_threshold, 
                                                                                  model_dir, 
                                                                                  Vocabulary.INPUT_VOCAB_FILENAME,
                                                                                  encoder_embeddings_dir,
                                                                                  training_hparams.input_vocab_import_normalized,
                                                                                  training_hparams.input_vocab_import_mode)

            output_vocabulary, read_stats.output_vocabulary_import_stats = self._create_and_save_vocab(answers,
                                                                                    training_hparams.output_vocab_threshold, 
                                                                                    model_dir, 
                                                                                    Vocabulary.OUTPUT_VOCAB_FILENAME,
                                                                                    decoder_embeddings_dir,
                                                                                    training_hparams.output_vocab_import_normalized,
                                                                                    training_hparams.output_vocab_import_mode)
        
        # Adding the End Of String tokens to the end of every answer
        for i in range(len(answers)):
            answers[i] += " {0}".format(Vocabulary.EOS)
        
        #Create the Dataset object from the questions / answers lists and the vocab object.
        dataset = Dataset(questions, answers, input_vocabulary, output_vocabulary)
        
        return dataset, read_stats

    def _create_and_save_vocab(self, word_sequences, vocab_threshold, model_dir, vocab_filename, embeddings_dir, normalize_imported_vocab, vocab_import_mode):
        
        vocabulary = None
        if embeddings_dir is None or vocab_import_mode != VocabularyImportMode.External:
            vocabulary = Vocabulary()
            for i in range(len(word_sequences)):
                word_seq = word_sequences[i]
                vocabulary.add_words(word_seq.split())
            vocabulary.compile(vocab_threshold)
        
        vocabulary_import_stats = None
        if embeddings_dir is not None:
            vocabulary_importer = vocabulary_importer_factory.get_vocabulary_importer(embeddings_dir)
            vocabulary, vocabulary_import_stats = vocabulary_importer.import_vocabulary(embeddings_dir, 
                                                                                        normalize_imported_vocab, 
                                                                                        vocab_import_mode, 
                                                                                        vocabulary)

        vocab_filepath = path.join(model_dir, vocab_filename)
        vocabulary.save(vocab_filepath)
        return vocabulary, vocabulary_import_stats

    def _log_cleaned_dataset(self, model_dir, lines):
        """Write the cleaned dataset to disk
        """
        log_filepath = path.join(model_dir, "cleaned_dataset.txt")
        with open(log_filepath, mode="w", encoding="utf-8") as file:
            for line in lines:
                file.write(line)
                file.write('\n')