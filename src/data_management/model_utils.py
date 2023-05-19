from joblib import dump, load

def save_model(path, model):
    dump(model, path)

def load_model(path):
    return load(path) 