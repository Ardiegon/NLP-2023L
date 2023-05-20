# Project NLP 2023L
Double quality assesment using anomaly detection. 

# Install Env
```bash
conda env create -n nlp-env -f env.yml
conda activate nlp-env
conda develop .
conda develop src
```

or 

```bash
pip install -r requirements.txt
```
and then set your PYTHONPATH to "." and "./src" in some way

# Using
```bash
python classify.py "comment 1" "comment 2" ... "comment n" --language polish --model svm
```  
Current Languages:
- polish
- german

Current Models:
- bayes: Complement Naive Bayes (https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB)
- svm: OneClassSVM (https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)

exrun:
```bash
python .\classify.py "Proszek lepszy niż ten w polsce" "Używamy tego detergentu od wielu lat i jesteśmy bardzo zadowoleni."
```
Should return positive, and negative

# Literature
Training German Doc2Vec - https://devmount.github.io/GermanWordEmbeddings/  

German genral use BERT - https://huggingface.co/bert-base-german-cased  

Polish general use BERT - https://huggingface.co/dkleczek/bert-base-polish-cased-v1  

hugging face transformers tutorial - https://huggingface.co/docs/transformers/index  

Possible sentence encoding techniques - https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/  

