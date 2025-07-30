import pandas as pd
import re
import spacy
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords si nécessaire
nltk.download("stopwords", quiet=True)

def get_stopwords():
    base_stopwords = set(stopwords.words('english'))
    custom_stopwords = set(['a','about','above','after','again','against','all','am','an','and','any','are','aren\'t','as','at','be','because','been','before','being','below','between','both','but','by','can','can\'t','cannot','could','couldn\'t','did','didn\'t','do','does','doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t','have','haven\'t','having','he','he\'d','he\'ll','he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','i','i\'d','i\'ll','i\'m','i\'ve','if','in','into','is','isn\'t','it','it\'s','its','itself','let\'s','me','more','most','mustn\'t','my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own','same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than','that','that\'s','the','their','theirs','them','themselves','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those','through','to','too','under','until','up','very','was','wasn\'t','we','we\'d','we\'ll','we\'re','we\'ve','were','weren\'t','what','what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would','wouldn\'t','you','you\'d','you\'ll','you\'re','you\'ve','your','yours','yourself','yourselves','https','http','www','.com','.org','.net'])
    return base_stopwords.union(custom_stopwords)

def load_spacy_model():
    return spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def clean_comment(text, nlp, stopwords_set):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if token.lemma_ not in stopwords_set and token.lemma_ != '-PRON-'])

def run_cleaning_pipeline(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    stopwords_set = get_stopwords()
    nlp = load_spacy_model()
    df['cleaned'] = df['comment'].apply(lambda x: clean_comment(x, nlp, stopwords_set))
    df.to_csv(output_path, index=False)