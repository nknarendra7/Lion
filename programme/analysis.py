# # load data
# # preprocess(tokenize,stemming,lemmatization)
# # create Dictionary from dataset
# # filter extremes(no_below and no_above)
# # convert doc2bow (Bag of Words on dataset ( number of times a word appears in the training set)
# # Running LDA using BoW

import pandas as pd
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
import multiprocessing
from gensim.models import CoherenceModel





df = pd.read_csv('train.csv')
# print(df)

stemmer = SnowballStemmer('english')
def stem_lemmatize(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text))

def preprocess(text):
    data = []
    for word in simple_preprocess(text):
        if word not in STOPWORDS and len(word)>3:
            d = stem_lemmatize(word)
            data.append(d)
    return data
process_doc = []
for i in df.Phrase:
    process_doc.append(preprocess(i))

dict_doc = Dictionary(process_doc)
print(dict_doc)
dict_doc.filter_extremes(no_below=2,no_above=.1,keep_n=100000)
bow_corpus = [dict_doc.doc2bow(x) for x in process_doc]
# print(bow_corpus)
# lda_model = LdaMulticore(bow_corpus,num_topics=8,id2word=dict_doc,workers=2,passes=8)
# lda_model = LdaMulticore(bow_corpus)
# print(lda_model)

#for unseen document
unseen_doc = "Javascript add row and calculate multiple rows from html"
bow_vec = dict_doc.doc2bow(preprocess(unseen_doc))
# print(bow_vec)



if __name__ == '__main__':
    multiprocessing.freeze_support() #for windows system
    lda_model = LdaMulticore(corpus=bow_corpus, id2word=dict_doc, num_topics=4)
    # print(lda)
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic))
        print("\n")

#     print("*******************************************")
#     for index, score in sorted(lda_model[bow_vec], key=lambda tup: -1 * tup[1]):
#         print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    print((lda_model[bow_vec]))
    output = lda_model[bow_vec]


    b = sorted(output,key = lambda x:x[1],reverse = True)
    print(b[0][0])

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(bow_corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=process_doc, dictionary=dict_doc, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

