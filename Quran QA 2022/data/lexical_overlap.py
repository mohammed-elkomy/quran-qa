import string
from functools import partial

from datasets import load_dataset, DownloadConfig
# import regex
import re

import nltk

# import nltk for stopwords
from farasa.stemmer import FarasaStemmer
from nltk.corpus import stopwords
import stanza
from tqdm import tqdm

seen_strings = {}


def preprocess_text_squad(input_string, nlp):
    if input_string in seen_strings:
        return seen_strings[input_string]

    stop_words = set(stopwords.words('english'))

    # convert to lower case
    lower_string = input_string.lower()

    # remove numbers
    no_number_string = re.sub(r'\d+', '', lower_string)

    # remove all punctuation except words and space
    no_punc_string = re.sub(r'[^\w\s]', '', no_number_string)

    # remove white spaces
    no_wspace_string = no_punc_string.strip()

    # convert string to list of words
    lst_string = [no_wspace_string][0].split()

    # remove stopwords
    no_stpwords_string = "".join(
        i + ' ' for i in lst_string if not i in stop_words
    )

    # removing last space
    no_stpwords_string = no_stpwords_string[:-1]

    lemmatized_out = nlp(no_stpwords_string)
    lemmas = {word.lemma for sent in lemmatized_out.sentences for word in sent.words}

    seen_strings[input_string] = lemmas
    return lemmas


def preprocess_text_qrcd(input_string, stemmer):
    if input_string in seen_strings:
        return seen_strings[input_string]

    def remove_stopWords(text):
        terms = []
        stopWords = {'من', 'الى', 'إلى', 'عن', 'على', 'في', 'حتى'}
        for term in text.split():
            if term not in stopWords:
                terms.append(term)
        return " ".join(terms)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        # Arabic punctuation
        exclude.add('،')
        exclude.add('؛')
        exclude.add('؟')
        return ''.join(ch for ch in text if ch not in exclude)

    cleaned_string = white_space_fix(remove_stopWords(remove_punc(input_string)))
    stemmed = set(stemmer.stem(cleaned_string).split())
    seen_strings[input_string] = stemmed
    return stemmed


def lexical_matching(dataset, process_function):
    for split in dataset:
        # dataset[split] = dataset[split].select(range(100))
        dataset_size = dataset[split].num_rows
        average_overlap = 0
        for question, answers in tqdm(zip(dataset[split]["question"], dataset[split]["answers"]), total=dataset_size, desc=f"processing {split}"):
            question_lemmas = process_function(question)
            answers_lemmas = [process_function(answer) for answer in answers["text"]]
            all_answers_lemmas = {word for answer in answers_lemmas for word in answer}
            average_overlap += intersection_index(all_answers_lemmas, question_lemmas)
        average_overlap /= dataset_size
        print(f"for {split} split", average_overlap * 100)


def intersection_index(all_answers_lemmas, question_lemmas):
    return len(all_answers_lemmas.intersection(question_lemmas)) / (min(len(all_answers_lemmas), len(question_lemmas)) + 1e-7)


def lexical_overlap_squad():
    # download stpwords
    nltk.download('stopwords')
    stanza.download('en')

    nlp = stanza.Pipeline(lang='en', use_gpu=False, processors='tokenize,mwt,pos,lemma')

    dataset = load_dataset("squad_dataset_loader.py",
                           data_files={'train': 'squad/train-v1.1.json',
                                       'validation': 'squad/dev-v1.1.json',
                                       },
                           download_config=DownloadConfig(local_files_only=True)
                           )
    dataset["train"] =     dataset["train"].select(range(3))
    process_function = partial(preprocess_text_squad, nlp=nlp)
    lexical_matching(dataset, process_function)


def lexical_overlap_qrcd():
    stemmer = FarasaStemmer(interactive=True) # to be faster

    dataset = load_dataset("qrcd_dataset_loader.py",
                           data_files={'train': 'qrcd/qrcd_v1.1_train.jsonl',
                                       'validation': 'qrcd/qrcd_v1.1_dev.jsonl',
                                       },

                           download_config=DownloadConfig(local_files_only=True), preprocessor=None
                           )

    process_function = partial(preprocess_text_qrcd, stemmer=stemmer)
    lexical_matching(dataset, process_function)


if __name__ == "__main__":
    lexical_overlap_squad()
    """
    for train split 7.471337418110706
    for validation split 10.78994588299155
    
    """
    lexical_overlap_qrcd()
    """
    for train split 10.593566083208007
    for validation split 4.826343258288479
    """