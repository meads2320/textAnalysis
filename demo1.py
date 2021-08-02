#Loading NLTK
import nltk
nltk.download('punkt')

#Tokenization is the first step in text analytics.
#Tokenization: The process of breaking down a text paragraph into smaller chunks such as words or sentence

#Sentence tokenizer breaks text paragraph into sentences.

from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)
