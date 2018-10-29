import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_pandas import DataFrameMapper
from sklearn.naive_bayes import MultinomialNB

testCSV = pd.read_csv(r'Dataset(Analysis)(processed lyrics).csv')


bow = CountVectorizer(max_features=1000,
                      lowercase=True,
                      ngram_range=(1,1),
                      analyzer="word").fit(testCSV['Lyrics'].values.astype(str))
len(bow.vocabulary_)


lyrics_bow = bow.transform(testCSV['Lyrics'].values.astype(str))
# print('Shape of Sparse Matrix: ', lyrics_bow.shape)
print("Back end begin...")

tfidf_transformer = TfidfTransformer().fit(lyrics_bow)
lyrics_tfidf = tfidf_transformer.transform(lyrics_bow)

df = pd.DataFrame()
df['lyrics']=list(lyrics_tfidf.toarray())
df['hit'] =  testCSV['Hit']


model = MultinomialNB()
model.fit(df['lyrics'].tolist(), df['hit'].tolist())


def process(userLyrics):
    lowerCase = lambda x: " ".join(x.lower() for x in str(x).split())
    userLyrics = lowerCase(userLyrics)

    # Removing punctuation that does not add meaning to the song
    userLyrics = userLyrics.replace('[^\w\s]', '')

    # Removing of stop words
    from nltk.corpus import stopwords

    stop = stopwords.words('english')
    removeStopWords = lambda x: " ".join(x for x in str(x).split() if x not in stop)
    userLyrics = removeStopWords(userLyrics)

    # Correction of Spelling mistakes
    from textblob import TextBlob
    spellingMistake = lambda x: str(TextBlob(x).correct())
    userLyrics = spellingMistake(userLyrics)

    # Lemmatization is basically converting a word into its root word. It is preferred over Stemming.
    from textblob import Word
    lemmatize = lambda x: " ".join([Word(word).lemmatize() for word in x.split()])
    userLyrics = lemmatize(userLyrics)
    print(userLyrics)
    return userLyrics