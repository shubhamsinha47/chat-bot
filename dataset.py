"""
Dataset class
"""
import math
import random
import numpy as np
from os import path

class Dataset(object):
    """Class representing a chatbot dataset with questions, answers, and vocabulary.
    """
    
    def __init__(self, questions, answers, input_vocabulary, output_vocabulary):
        
        if len(questions) != len (answers):
            raise RuntimeError("questions and answers lists must be the same length, as they are lists of input-output pairs.")

        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary
        #If the questions and answers are already integer encoded, accept them as is.
        #Otherwise use the Vocabulary instances to encode the question and answer sequences.
        if len(questions) > 0 and isinstance(questions[0], str):
            self.questions_into_int = [self.input_vocabulary.words2ints(q) for q in questions]
            self.answers_into_int = [self.output_vocabulary.words2ints(a) for a in answers]
        else:
            self.questions_into_int = questions
            self.answers_into_int = answers
    
    def size(self):
        """ The size (number of samples) of the Dataset.
        """
        return len(self.questions_into_int)
    
    def train_val_split(self, val_percent = 20, random_split = True, move_samples = True):

        if move_samples:
            questions = self.questions_into_int
            answers = self.answers_into_int
        else:
            questions = self.questions_into_int[:]
            answers = self.answers_into_int[:]
        
        num_validation_samples = int(len(questions) * (val_percent / 100))
        num_training_samples = len(questions) - num_validation_samples
        
        training_questions = []
        training_answers = []
        validation_questions = []
        validation_answers = []
        if random_split:
            for _ in range(num_validation_samples):
                random_index = random.randint(0, len(questions) - 1)
                validation_questions.append(questions.pop(random_index))
                validation_answers.append(answers.pop(random_index))
            
            for _ in range(num_training_samples):
                training_questions.append(questions.pop(0))
                training_answers.append(answers.pop(0))
        else:
            for _ in range(num_training_samples):
                training_questions.append(questions.pop(0))
                training_answers.append(answers.pop(0))
            
            for _ in range(num_validation_samples):
                validation_questions.append(questions.pop(0))
                validation_answers.append(answers.pop(0))
        
        training_dataset = Dataset(training_questions, training_answers, self.input_vocabulary, self.output_vocabulary)
        validation_dataset = Dataset(validation_questions, validation_answers, self.input_vocabulary, self.output_vocabulary)
        
        return training_dataset, validation_dataset
            
    def sort(self):
        """Sorts the dataset by the lengths of the questions. This can speed up training by reducing the
        amount of padding the input sequences need.
        """
        if self.size() > 0:
            self.questions_into_int, self.answers_into_int = zip(*sorted(zip(self.questions_into_int, self.answers_into_int), 
                                                                         key = lambda qa_pair: len(qa_pair[0])))

    def save(self, filepath):
        """Saves the dataset questions & answers exactly as represented by input_vocabulary and output_vocabulary.
        """
        filename, ext = path.splitext(filepath)
        questions_filepath = "{0}_questions{1}".format(filename, ext)
        answers_filepath = "{0}_answers{1}".format(filename, ext)

        with open(questions_filepath, mode="w", encoding="utf-8") as file:
            for question_into_int in self.questions_into_int:
                question = self.input_vocabulary.ints2words(question_into_int, is_punct_discrete_word = True, capitalize_i = False)
                file.write(question)
                file.write('\n')

        with open(answers_filepath, mode="w", encoding="utf-8") as file:
            for answer_into_int in self.answers_into_int:
                answer = self.output_vocabulary.ints2words(answer_into_int, is_punct_discrete_word = True, capitalize_i = False)
                file.write(answer)
                file.write('\n')


    
    def batches(self, batch_size):
        
        for batch_index in range(0, math.ceil(len(self.questions_into_int) / batch_size)):
                start_index = batch_index * batch_size
                questions_in_batch = self.questions_into_int[start_index : start_index + batch_size]
                answers_in_batch = self.answers_into_int[start_index : start_index + batch_size]
                
                seqlen_questions_in_batch = np.array([len(q) for q in questions_in_batch])
                seqlen_answers_in_batch = np.array([len(a) for a in answers_in_batch])
                
                padded_questions_in_batch = np.array(self._apply_padding(questions_in_batch, self.input_vocabulary))
                padded_answers_in_batch = np.array(self._apply_padding(answers_in_batch, self.output_vocabulary))
                
                yield padded_questions_in_batch, padded_answers_in_batch, seqlen_questions_in_batch, seqlen_answers_in_batch
    
    
    def _apply_padding(self, batch_of_sequences, vocabulary):
        
        max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
        return [sequence + ([vocabulary.pad_int()] * (max_sequence_length - len(sequence))) for sequence in batch_of_sequences]