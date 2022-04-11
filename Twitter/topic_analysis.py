import re
import nltk
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import itertools

from helpers.csv import load_tweets_csv

USERS_TWEETS = "../data/users_tweets.csv"
LDA_MODEL = "../data/models/simple_lda.model"
LDA_MODEL_DIC = "../data/models/simple_lda_dic"
IDF_LDA_MODEL = "../data/models/tf_idf_lda.model"
IDF_LDA_DIC = "../data/models/tf_idf_lda_dic"


def process_cloud_words(model, top, num_topics):
    words = dict()
    for t in range(num_topics):
        d = dict(model.show_topic(t, top))
        x = list(itertools.islice(d.items(), 0, top))
        for word in x:
            words[word[0]] = word[1]
    return " ".join(words) + " "


def generate_word_cloud(model, top, num_topics):
    words = process_cloud_words(model, top, num_topics)
    print(words)
    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10)
    wordcloud.generate(words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


################
#
# Tokenization
# Stopwords
# Punctuation
# Lemmanization
#
################

def tokenize_docs(docs):
    tokenized_docs = []
    for doc in docs:
        tokenized_docs.append(nltk.word_tokenize(doc.lower()))
    return tokenized_docs


def clean_process(doc):
    tokenized = nltk.word_tokenize(doc.lower())
    stop_free = [token for token in tokenized if token not in stop]
    punc_free = [ch for ch in stop_free if ch not in exclude]
    normalized = [lemma.lemmatize(word) for word in punc_free]
    return normalized


def clean_option_2(doc):
    valid_tags = ["NN", "NNS", "NNP", "VB", "VBG", "VBD", "VBN", "VBP", "VBZ", "JJ", "JJS", "JJR", "RB"]
    doc_nonum = re.sub(r'\d+', '', doc)
    doc_no_urls = re.sub(r"http\S+", "", doc_nonum)
    tokenized = nltk.word_tokenize(doc_no_urls.lower())
    no_small = [x for x in tokenized if len(x) > 2]
    tagged = nltk.pos_tag(no_small)
    tags = [t[0] for t in tagged if t[1] in valid_tags]
    stop_free = [tag for tag in tags if tag not in stop]
    punc_free = [ch for ch in stop_free if ch not in exclude]
    # normalized = [lemma.lemmatize(word) for word in punc_free]
    return punc_free


def apply_tf_idf(docs, prints):
    dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = gensim.models.tfidfmodel.TfidfModel
    tfidfmodel = tfidf(doc_term_matrix, id2word=dictionary)
    if (prints):
        print(tfidfmodel.id2word)
        print(tfidfmodel.dfs)
    voc = {}
    for i in range(len(tfidfmodel.id2word)):
        if tfidfmodel.dfs[i] > 1:
            voc[tfidfmodel.id2word[i]] = tfidfmodel.idfs[i]
    if (prints): print(len(voc), voc)
    sel_features = sorted(voc, key=voc.__getitem__, reverse=False)[:(len(voc))]
    if (prints): print(sel_features)
    new_doc_clean = [[w for w in doc if w in sel_features] for doc in docs]
    if (prints): print(new_doc_clean)
    return new_doc_clean


def model_lda(doc_clean, n_topics, words, coherence):
    dictionary, doc_term_matrix = get_doc_term_matrix(doc_clean)
    # generate LSA model
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(doc_term_matrix, num_topics=n_topics, id2word=dictionary, passes=100)
    print(ldamodel.print_topics(num_topics=n_topics, num_words=words))
    if (coherence):
        coherence_model_lsa = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_lsa = coherence_model_lsa.get_coherence()
        print('Coherence Score: ', coherence_lsa)
    return ldamodel, dictionary


def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start, step):
    coherence_values = []
    model_list = []
    print("coh start")
    for num_topics in range(start, stop, step):
        print("coh num_topics", num_topics)
        lda = gensim.models.ldamodel.LdaModel
        model = lda(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=100)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


# %%
def plot_graph(doc_clean, start, stop, step):
    print("coh dic")
    dictionary, doc_term_matrix = get_doc_term_matrix(doc_clean)

    print("coh lda")
    model_list_lda, coherence_values_lda = compute_coherence_values(dictionary, doc_term_matrix, doc_clean,
                                                                    stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values_lda, label="LDA")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.title("Coherence score for different number of topics")
    plt.show()

    num_topics_lda = x[coherence_values_lda.index(max(coherence_values_lda))]
    return num_topics_lda


def get_doc_term_matrix(docs):
    dictionary = corpora.Dictionary(docs)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
    return dictionary, doc_term_matrix


def test_baseline(model, dictionary, num_words, original_texts):
    for i in range(len(original_texts)):
        doc = original_texts[i]
        print(doc)
        docrep = dictionary.doc2bow(doc.split())
        max_res = max(model[docrep], key=lambda item: item[1])
        e = dict(model.show_topic(max_res[0], num_words))
        x = list(itertools.islice(e.items(), 0, num_words))
        f = [j[0] for j in x]
        print("max res: ", max_res, " | topic:", f, '\n')


def get_topic_analysis():
    tweets_texts = [tweet.text for tweet in users_tweets]
    cleaned_text = [clean_option_2(doc) for doc in tweets_texts]
    print("Cleaned tweets example:", cleaned_text[:5])

    # num_topics_lda = plot_graph(cleaned_text, 40, 60, 2)  # optimal is 54 for clean_type 1
    # print("Optimal number of topics for LDA model:", num_topics_lda)
    lda_model, lda_model_dic = model_lda(cleaned_text, 54, 5, False)
    test_baseline(lda_model, lda_model_dic, 20, original_texts=tweets_texts)
    print("\n")

    doc_clean_idf = apply_tf_idf(cleaned_text, False)
    # num_topics_lda = plot_graph(doc_clean_idf, 40, 60, 2)  # optimal is 52 for clean_type 1
    # print("Optimal number of topics for LDA model with TF_IDF:", num_topics_lda)
    idf_model_lda, idf_dic_lda = model_lda(doc_clean_idf, 52, 5, False)
    test_baseline(idf_model_lda, idf_dic_lda, 20, tweets_texts)

    lda_model.save(LDA_MODEL)
    lda_model_dic.save_as_text(LDA_MODEL_DIC)

    idf_model_lda.save(IDF_LDA_MODEL)
    idf_dic_lda.save_as_text(IDF_LDA_DIC)


if __name__ == '__main__':
    stop = set(stopwords.words('portuguese'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    users_tweets = load_tweets_csv(USERS_TWEETS)
    get_topic_analysis()
