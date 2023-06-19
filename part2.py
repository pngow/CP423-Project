import os
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

# NOTE: HOW MANY URLS PER TOPIC? 100 OR SPLIT THE 100?
def check_num_files(topics):
    PAGES_PER_TOPIC = 100

    # check if data folder exists
    if os.path.isdir('data'):
        # want folders to have at least 34+ urls each
        flag = False
        i = 0

        # check number of files in each topic folder
        while i < len(topics):
            count = 0

            # iterate through list of files in topic directory
            for file in os.listdir('data/' + topics[i]):
                # check if file path exists to the file listed in the directory
                if os.path.isfile(os.path.join('data', topics[i], file)):
                    count += 1

            # check if final count was more than 100
            if count >= PAGES_PER_TOPIC:
                flag = True

            i += 1

    return flag

def get_mapping():
    mapping = dict()

    try: 
        # get mapping
        with open('mapping.txt', 'r') as f:
            line = f.readline()

            while line:
                # 0 = url hash, 1 = doc id
                tokens = line.split(',')
                mapping[tokens[0].strip().split('/')[1]] = tokens[1].strip()

                line = f.readline()
    except:
        print("No mapping file exists. Re-run option 1.")
        # return to main?

    return mapping

def gen_soundex(term):
    # convert the word to uppercase
    term = term.upper()
 
    soundex = ""
 
    # keep first letter
    soundex += term[0]
 
    # create a dictionary which maps letters to respective soundexcodes. 
    # vowels and 'H', 'W' and 'Y' will be represented by '.' cause not coded
    codes = {"BFPV": "1", 
            "CGJKQSXZ": "2",
            "DT": "3",
            "L": "4", 
            "MN": "5", 
            "R": "6",
            "AEIOUHWY": "."}
 
    # encode based on the dictionary of codes
    for char in term[1:]:
        for key in codes.keys():
            if char in key:
                code = codes[key]
                if code != '.':
                    if code != soundex[-1]:
                        soundex += code
 
    # trim/pad to make soundex a 4-character code
    soundex = soundex[:4].ljust(4, "0")
 
    return soundex

def inverted_index(topics):
    print("\nCreating inverted index.")

    # check number of files downloaded
    if not check_num_files(topics):
        print("Insufficient number of files downloaded or data folder is empty. Re-run option 1.")
        return

    mapping = get_mapping()

    # create dictionary for inverted index if parameter was set to true  
    index = dict()
    # set of all unique terms
    terms = set()

    # iterate through topic directories
    for topic in topics:
        # iterate through files in directory
        for filename in os.listdir(f'data/{topic}'):
            # get doc id
            doc_id = mapping[filename.split('.')[0]]

            # get content downloaded from web page
            with open(f'data/{topic}/{filename}', 'r', encoding="utf-8") as f1:
                text = f1.read().splitlines()
                text = ' '.join(text)

            # loop through tokenized words in text ... create count for each 
            tokens = np.array(nltk.tokenize.word_tokenize(text.lower()))
            # numpy array ... unique counts ... dictionary
            unique_tokens, counts = np.unique(tokens, return_counts=True)

            # loop through unique tokens found to add to index dictionary 
            for term_index in range(len(unique_tokens)):
                # soundex doesn't convert numbers, so ignore in term list
                if re.search(r"\d", unique_tokens[term_index]) or re.match(r"^\d{1,}(?:\,\d{3})*(?:.\d{0,})?$|^\d+(?:.)?\d+$", unique_tokens[term_index]):
                    continue
                # ignore strings that only contain special characters
                elif re.match(r"^[\W_]+$", unique_tokens[term_index]):
                    continue
                # ignore strings that start with / ... likely just url
                elif unique_tokens[term_index].startswith('/') or unique_tokens[term_index].startswith('-'):
                    continue
                # ignore words longer than 20 characters adn doesn't have hyphens (-) ... likely not a word
                elif len(unique_tokens[term_index]) > 20 and "-" not in unique_tokens[term_index]:
                    continue

                # add term to set of all terms
                terms.add(unique_tokens[term_index])

                # generate soundex code
                soundex = gen_soundex(unique_tokens[term_index])

                # try except to get case where key word doesn't exist in the initial dictionary
                try:
                    index[unique_tokens[term_index]][doc_id] = int(counts[term_index])
                except:
                    # create nested dictionary for the word
                    index[unique_tokens[term_index]] = dict()
                    # store the soundex that was created
                    index[unique_tokens[term_index]]['soundex'] = soundex
                    # store the count for the term in the specific document id
                    index[unique_tokens[term_index]][doc_id] = int(counts[term_index])
            
    # make iterable
    terms = list(terms)

    with open('invertedindex.txt', mode='w', encoding="utf-8") as f2:
        # create formatted heading for inverted ndex
        f2.write("|Term|Soundex|Appearances (DocID, Frequency)|\n")
        f2.write("|---------|----------------------------------|\n")

        # for each unique term
        for term in terms:
            # get soundex value for the term
            soundex = index[term].pop('soundex')

            # format string of appearances of the term in documents
            appearances = [str(pair).replace("'", "") for pair in index[term].items()]
            formatted_appearances = ', '.join(appearances)

            # create output string
            out_string = f"|{term}|{soundex}|{formatted_appearances}|\n"
            f2.write(out_string)      

    print("Complete.\n") 
