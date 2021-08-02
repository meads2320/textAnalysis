#Loading NLTK - Natural Language Toolkit https://www.nltk.org/
#NLTK is a leading platform for building Python programs to work with human language data
import nltk

#NLTK Downloader to use specific resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

#Tokenization is the first step in text analytics.

#Tokenization: The process of breaking down a text paragraph into smaller chunks such as words or sentence

#Sentence Tokenization - breaks text paragraph into sentences.

from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)

#Output: ['Hello Mr. Smith, how are you doing today?', 'The weather is great, and city is awesome.', 'The sky is pinkish-blue.', "You shouldn't eat cardboard"]

#Word Tokenization - breaks text paragraph into words.

from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

#Output: ['Hello', 'Mr.', 'Smith', ',', 'how', 'are', 'you', 'doing', 'today', '?', 'The', 'weather', 'is', 'great', ',', 'and', 'city', 'is', 'awesome', '.', 'The', 'sky', 'is', 'pinkish-blue', '.', 'You', 'should', "n't", 'eat', 'cardboard']


# Frequency Distribution

from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)

#Output: <FreqDist with 25 samples and 30 outcomes>

fdist.most_common(2)
#Output: [('is', 3), (',', 2)]

# Frequency Distribution Plot
#----------------------------------------------------------------
# import matplotlib.pyplot as plt
# fdist.plot(30,cumulative=False)
# plt.show()
#----------------------------------------------------------------

#Stopwords - considered as noise in the text

#Examples: [as, is, am, are, this, a, an, the, etc.]

#In NLTK for removing stopwords, you need to create a list of stopwords and filter out your list of tokens from these words.

from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)

#Output: {'out', 'off', 'same', "haven't", 'or', 'again', "you've", 'himself', 'how', 'just', "shouldn't", 'these', "needn't", 'some', 'aren', 'there', 're', 'through', 'his', 'its', "wouldn't", "wasn't", 'not', 'me', 'that', 'while', 'where', 'own', 'are', 'if', 'was', 'having', 'be', 'does', 'the', 'only', 'too', 'above', 'under', 'd', 'both', 'herself', "isn't", 'am', "you're", 't', 'on', "don't", 'theirs', 'her', 'haven', 'into', 'from', 'further', 'as', 'below', 'm', 'i', 'who', 'y', "aren't", 'hasn', 'it', "should've", "she's", 'their', 'my', 'have', 'hers', 'no', 'did', 'at', 'most', "won't", "doesn't", 'they', 'what', 'wouldn', 'weren', 'been', 'has', 'can', 'o', 'he', 'won', 'should', 'ours', 'few', 'didn', 'had', 've', 'were', 'them', 'here', 'll', 'ain', 'than', "mightn't", 'all', 'other', 'hadn', 'now', 'yourselves', 'by', 'our', 'why', 'any', 'whom', "shan't", 'isn', 'shan', "you'll", 'myself', "that'll", 'each', "weren't", 'during', "didn't", 'in', 'itself', "hadn't", 'after', 'your', 'those', 'and', 'then', 'once', 'so', 'she', 'more', 'shouldn', 'which', 'a', 'doesn', 'mightn', 'doing', 'this', 'yourself', "hasn't", 'about', 'ourselves', "couldn't", 'of', 'but', 'up', 'is', 'do', 'against', 'mustn', "it's", 'will', 'themselves', 'ma', "mustn't", 'when', 'yours', 'you', 'couldn', 's', 'don', 'before', 'nor', 'wasn', 'because', 'such', 'very', 'over', 'down', 'for', 'between', 'needn', 'him', 'with', 'we', 'to', 'until', 'an', "you'd", 'being'}

# Removing Stopwords

filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filtered Sentence:",filtered_sent)

#Tokenized Sentence: ['Hello', 'Mr.', 'Smith', ',', 'how', 'are', 'you', 'doing', 'today', '?']
#Filtered Sentence: ['Hello', 'Mr.', 'Smith', ',', 'today', '?']

#Lexicon Normalization -  considers another type of noise in the text

#For example, [connection, connected, connecting] word reduce to a common word "connect"

#Stemming

# Stemming Removes affixes from words - [connection, connected, connecting] would reduce to a common word "connect".
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

#Filtered Sentence: ['Hello', 'Mr.', 'Smith', ',', 'today', '?']
#Stemmed Sentence: ['hello', 'mr.', 'smith', ',', 'today', '?']


#Lemmatization - reduces words to their base word, which is linguistically correct lemmas
#Lemmatization is usually more sophisticated than stemming
#For example, The word "better" has "good" as its lemma

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))


#POS Tagging - The primary target of Part-of-Speech(POS)
#Tags: NOUN, PRONOUN, ADJECTIVE, VERB, ADVERBS, etc. based on the context

sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nltk.word_tokenize(sent)
print(tokens)

#Output: ['Albert', 'Einstein', 'was', 'born', 'in', 'Ulm', ',', 'Germany', 'in', '1879', '.']

tagged_tokens = nltk.pos_tag(tokens)

print(tagged_tokens)

#Output: [('Albert', 'NNP'), ('Einstein', 'NNP'), ('was', 'VBD'), ('born', 'VBN'), ('in', 'IN'), ('Ulm', 'NNP'), (',', ','), ('Germany', 'NNP'), ('in', 'IN'), ('1879', 'CD'), ('.', '.')]

#Sentiment Analysis - Sentiments are combination words, tone, and writing style


#Type 1 - Lexicon-based

#count number of positive and negative words in given text and the larger count will be the sentiment of text.

#Type 2 - Machine learning based approach:

#Develop a classification model, which is trained using the pre-labeled dataset of positive, negative, and neutral.
