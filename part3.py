import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
from itertools import islice
import ast

def load_inverted_index(file_path: str) -> Dict[str, Tuple[str, List[Tuple[str, int]]]]:
    inverted_index = {}
    with open(file_path, 'r') as f:

        for line in f.readlines():
            #skip processing header lines
            if line == "|Term|Soundex|Appearances (DocID, Frequency)|\n" or line == '|---------|----------------------------------|\n':
                continue

            #split each line by '|' + remove the '|' at the beginning and end of each line
            term, soundex, appearances = re.split(r'\s*\|\s*', line.strip())[1:-1]

            #example of appearances: "(H1, 2), (H4, 6), (H33, 44)" etc.
            # So ignoring first two characters and splitting by "), (" will give a list where each item is a string representing [DocID, Frequency] 
            appearances_list = appearances[1:-1].split('), (')

            #For each string item (DocID, Freq) in apperances_list split it to get an actual list representing (DocID, Freq)
            appearances_list = [tuple(x.split(', ')) for x in appearances_list]
            appearances_list = [(x[0], int(x[1])) for x in appearances_list]
            
            #map the soundex and list of apperances to the term 
            inverted_index[term] = (soundex, appearances_list)
    return inverted_index

def load_mapping(file_path: str) -> Dict[str, str]:

    '''
    Example of mapping file: 
            Technology/1d0ca036ed0767110cfc554b89f0b2f25c1d306586fe14574347db8275e573b8, H1
            Entertainment/1557c4d291a873db1bc57c92c5141c97fe65d1f6b031d462ec2fb59d3aa7b735, H2, ...

    Tracking folder/topic alongside hashID will help finding documents when printing top 3 documents matching query
    '''

    mapping = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            doc_hash, doc_id = line.strip().split(', ')
            mapping[doc_id] = doc_hash
    return mapping

def soundex(term):
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

def correct_query_terms(query: str, inverted_index: Dict[str, Tuple[str, List[Tuple[str, int]]]]) -> List[str]:
    words = query.split()
    corrected_words = []

    for word in words:
        if word in inverted_index:
            corrected_words.append(word)
        else:
            #if word is not spelled correctly, 
            
            word_soundex = soundex(word)
            best_word = None
            best_distance = float("inf")
            
            #   calculate soundex of misspelled word and compare vs soundex of terms in inverted index 
            for term, (term_soundex, _) in inverted_index.items():
                distance = sum(map(lambda pair: pair[0] != pair[1], zip(word_soundex, term_soundex)))
                if distance < best_distance:
                    best_distance = distance
                    best_word = term

            #Take the word with the closest soundex code as the corrected term
            corrected_words.append(best_word)

    return corrected_words

def term_at_a_time(query_terms: List[str], inverted_index: Dict[str, Tuple[str, List[Tuple[str, int]]]]) -> List[str]:
    found_docs = set()

    #for each term, make a list of documents where it appears using the apperances field in inverted_index 
    for term in query_terms:
        for appearance in inverted_index[term][1]:
            found_docs.add(appearance[0])
    return list(found_docs)

def vectorize_documents(doc_ids: List[str], mapping: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    vectorizer = TfidfVectorizer(input='filename', encoding='utf-8')
    file_names = [f'data/{mapping[doc_id]}.txt' for doc_id in doc_ids]
    tfidf_matrix = vectorizer.fit_transform(file_names)

    return tfidf_matrix, vectorizer.get_feature_names_out()

def search_query():

    inverted_index = load_inverted_index('invertedindex.txt')
    mapping = load_mapping('mapping.txt')

    
    query = input("Enter your query: ")

    #Check each term for corrections
    corrected_query_terms = correct_query_terms(query, inverted_index)
    # print(f"Corrected query terms: {corrected_query_terms}")

    #get list of documents containing at least 1 occurence of each term in query
    found_doc_ids = term_at_a_time(corrected_query_terms, inverted_index)
    # print(f"Found documents: {found_doc_ids}, {len(found_doc_ids)}")
    
    #Vectorize documents
    doc_vectors, feature_names = vectorize_documents(found_doc_ids, mapping)
    
    #Vectorize query
    vectorizer = TfidfVectorizer(vocabulary=feature_names)
    query_vector = vectorizer.fit_transform([" ".join(corrected_query_terms)])

    #Calculate cosine similarities to find top 3 matching documents
    cosine_similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    top_3_indices = cosine_similarities.argsort()[::-1]
    top_3_indices = top_3_indices[0:3]
    top_3_doc_ids = [found_doc_ids[i] for i in top_3_indices]

    #print results
    print("Top 3 most related documents:")
    for i, doc_id in enumerate(top_3_doc_ids):
        file_name = f'data/{mapping[doc_id]}.txt'
        with open(file_name, 'r') as f:
            content = f.read()
            #Highlight query terms in different color when printing results
            for term in corrected_query_terms:
                reColor = re.compile(re.escape(term), re.IGNORECASE)
                content = reColor.sub(f"\033[94m{term}\033[0m", content)
                
            print(f"\nDocument {i + 1} (DocID: {doc_id}):\n{content}\n")


if __name__ == "__main__":
    search_query()
