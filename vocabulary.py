"""
Vocabulary class
"""
import re

class Vocabulary(object):
    
    
    SHARED_VOCAB_FILENAME = "shared_vocab.tsv"
    INPUT_VOCAB_FILENAME = "input_vocab.tsv"
    OUTPUT_VOCAB_FILENAME = "output_vocab.tsv"

    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    OUT = "<OUT>"
    special_tokens = [PAD, SOS, EOS, OUT]
    
    def __init__(self, external_embeddings = None):
        
        self._word2count = {}
        self._words2int = {}
        self._ints2word = {}
        self._compiled = False
        self.external_embeddings = external_embeddings

    def load_word(self, word, word_int, count = 1):
        
        self._validate_compile(False)

        self._word2count[word] = count
        self._words2int[word] = word_int
        self._ints2word[word_int] = word
        
    def add_words(self, words):
        
        self._validate_compile(False)

        for i in range(len(words)):
            word = words[i]
            if word in self._word2count:
                self._word2count[word] += 1
            else:
                self._word2count[word] = 1
    
    def compile(self, vocab_threshold = 1, loading = False):
        
        self._validate_compile(False)
        
        if not loading:
            #Add the special tokens to the lookup dictionaries
            for i, special_token in enumerate(Vocabulary.special_tokens):
                self._words2int[special_token] = i
                self._ints2word[i] = special_token

            
            word_int = len(self._words2int)
            for word, count in sorted(self._word2count.items()):
                if count >= vocab_threshold:
                    self._words2int[word] = word_int
                    self._ints2word[word_int] = word
                    word_int += 1
                else:
                    del self._word2count[word]
            
            #Add the special tokens to _word2count so they have count values for saving to disk
            self.add_words(Vocabulary.special_tokens)

        #The Vocabulary instance may now be used for integer encoding / decoding
        self._compiled = True



    def size(self):
        """The size (number of words) of the Vocabulary
        """
        self._validate_compile(True)
        return len(self._word2count)
    
    def word_exists(self, word):
        """Check if the given word exists in the vocabulary.

        Args:
            word: The word to check.
        """
        self._validate_compile(True)
        return word in self._words2int

    def words2ints(self, words):
        
        return [self.word2int(w) for w in words.split()]
    
    def word2int(self, word):
        
        self._validate_compile(True)
        return self._words2int[word] if word in self._words2int else self.out_int()

    def ints2words(self, words_ints, is_punct_discrete_word = False, capitalize_i = True):
        
        words = ""
        for i in words_ints:
            word = self.int2word(i, capitalize_i)
            if is_punct_discrete_word or word not in ['.', '!', '?']:
                words += " "
            words += word
        words = words.strip()
        return words

    def int2word(self, word_int, capitalize_i = True):
        
        self._validate_compile(True)
        word = self._ints2word[word_int]
        if capitalize_i and word == 'i':
            word = 'I'
        return word
    
    def pad_int(self):
        """Get the integer encoding of the PAD token
        """
        return self.word2int(Vocabulary.PAD)

    def sos_int(self):
        """Get the integer encoding of the SOS token
        """
        return self.word2int(Vocabulary.SOS)
    
    def eos_int(self):
        """Get the integer encoding of the EOS token
        """
        return self.word2int(Vocabulary.EOS)

    def out_int(self):
        """Get the integer encoding of the OUT token
        """
        return self.word2int(Vocabulary.OUT)
    
    def save(self, filepath):
        """Saves the vocabulary to disk.

        Args:
            filepath: The path of the file to save to
        """
        total_words = self.size()
        with open(filepath, "w", encoding="utf-8") as file:
            file.write('\t'.join(["word", "count"]))
            file.write('\n')
            for i in range(total_words):
                word = self._ints2word[i]
                count = self._word2count[word]
                file.write('\t'.join([word, str(count)]))
                if i < total_words - 1:
                    file.write('\n')

    def _validate_compile(self, expected_status):
        """Validate that the vocabulary is compiled or not based on the needs of the attempted operation

        Args:
            expected_status: The compilation status expected by the attempted operation
        """
        if self._compiled and not expected_status:
            raise ValueError("This vocabulary instance has already been compiled.")
        if not self._compiled and expected_status:
            raise ValueError("This vocabulary instance has not been compiled yet.")
    
    @staticmethod        
    def load(filepath):
        """Loads the vocabulary from disk.

        Args:
            filepath: The path of the file to load from
        """
        vocabulary = Vocabulary()
        
        with open(filepath, encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index > 0: #Skip header line
                    word, count = line.split('\t')
                    word_int = index - 1
                    vocabulary.load_word(word, word_int, int(count))
        
        vocabulary.compile(loading = True)
        return vocabulary
    
    @staticmethod
    def clean_text(text, max_words = None, normalize_words = True):
        
        text = text.lower()
        text = re.sub(r"'+", "'", text)
        if normalize_words:
            text = re.sub(r"i'm", "i am", text)
            text = re.sub(r"he's", "he is", text)
            text = re.sub(r"she's", "she is", text)
            text = re.sub(r"that's", "that is", text)
            text = re.sub(r"there's", "there is", text)
            text = re.sub(r"what's", "what is", text)
            text = re.sub(r"where's", "where is", text)
            text = re.sub(r"who's", "who is", text)
            text = re.sub(r"how's", "how is", text)
            text = re.sub(r"it's", "it is", text)
            text = re.sub(r"let's", "let us", text)
            text = re.sub(r"\'ll", " will", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"\'re", " are", text)
            text = re.sub(r"\'d", " would", text)
            text = re.sub(r"won't", "will not", text)
            text = re.sub(r"shan't", "shall not", text)
            text = re.sub(r"can't", "can not", text)
            text = re.sub(r"cannot", "can not", text)
            text = re.sub(r"n't", " not", text)
            text = re.sub(r"'", "", text)
        else:
            text = re.sub(r"(\W)'", r"\1", text)
            text = re.sub(r"'(\W)", r"\1", text)
        text = re.sub(r"[()\"#/@;:<>{}`+=~|$&*%\[\]_]", "", text)
        text = re.sub(r"[.]+", " . ", text)
        text = re.sub(r"[!]+", " ! ", text)
        text = re.sub(r"[?]+", " ? ", text)
        text = re.sub(r"[,-]+", " ", text)
        text = re.sub(r"[\t]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()
        
        if max_words is not None:
            text_parts = text.split()
            if len(text_parts) > max_words:
                truncated_text_parts = text_parts[:max_words]
                while len(truncated_text_parts) > 0 and not re.match("[.!?]", truncated_text_parts[-1]):
                    truncated_text_parts.pop(-1)
                if len(truncated_text_parts) == 0:
                    truncated_text_parts = text_parts[:max_words]
                text = " ".join(truncated_text_parts)
                
        return text

    @staticmethod
    def auto_punctuate(text):
        
        text = text.strip()
        if not (text.endswith(".") or text.endswith("?") or text.endswith("!") or text.startswith("--")):
            tmp = re.sub(r"'", "", text.lower())
            if (tmp.startswith("who") or tmp.startswith("what") or tmp.startswith("when") or 
                    tmp.startswith("where") or tmp.startswith("why") or tmp.startswith("how") or
                    tmp.endswith("who") or tmp.endswith("what") or tmp.endswith("when") or 
                    tmp.endswith("where") or tmp.endswith("why") or tmp.endswith("how") or
                    tmp.startswith("are") or tmp.startswith("will") or tmp.startswith("wont") or tmp.startswith("can")):
                text = "{}?".format(text)
            else:
                text = "{}.".format(text)
        return text