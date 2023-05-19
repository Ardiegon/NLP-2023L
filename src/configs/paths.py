from os.path import join

DATA_DIR = join("data")
DATA_NLP2023 = join(DATA_DIR, "nlp2023_v2.csv")
DATA_DOUBLEQUALITY = join(DATA_DIR, "doubleQuality.csv")
POLISH_STOPWORDS = join(DATA_DIR, "polish.stopwords.txt")

MODELS_DIR = join("src","models")
SVM_POLISH = join(MODELS_DIR, "svm_polish.pickle")
SVM_GERMAN = join(MODELS_DIR, "svm_german.pickle")
