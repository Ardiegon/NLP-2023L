import time
import numpy as np
import pandas as pd
import translators as ts

from enum import Enum
from tqdm import tqdm
from os.path import join
from typing import Callable
from data.load_data import load_raw_data_from_path
from configs.paths import DATA_DIR, DATA_NLP2023
from requests.exceptions import HTTPError, JSONDecodeError

USE_PREACCELERATION = False
HTTP_ERROR_PROOFING = True
MAX_NUMBER_OF_RETRIES = 20
TRANSLATOR = "alibaba"

class TranslationType(Enum):
    PTG = 1
    GTP = 2

def translator(x:str, from_lang:str, to_lang:str, use_delay=False)->str:
    """
    Translate with delay. Delay is counter for 429 Error code for too many requests
    """
    text =ts.translate_text(query_text=x, translator=TRANSLATOR, from_language=from_lang, to_language=to_lang, if_use_preacceleration=USE_PREACCELERATION)
    if use_delay:
        time.sleep(2)
    return text

def get_translator(translation_type:TranslationType, use_delay=True)->Callable:
    if translation_type == TranslationType.GTP:
        return lambda x: translator(x, "de", "pl", use_delay=use_delay)
    elif translation_type == TranslationType.PTG:
        return lambda x: translator(x, "pl", "de", use_delay=use_delay)

def get_worker()->Callable:    
    return lambda x: x[::-1]

def translate_series(x:pd.Series, translation_type:TranslationType, use_http_error_proofing:bool = HTTP_ERROR_PROOFING)->pd.Series:
    if not use_http_error_proofing:
        return x.apply(get_translator(translation_type))
    else:
        result = x.copy()
        translator = get_translator(translation_type, use_delay=False)
        time_to_sleep = 1

        for i in tqdm(range(len(x))):
            job_done = False
            while not job_done:
                try:
                    result[i] = translator(str(x[i]))
                    time_to_sleep = 1
                    job_done = True
                except HTTPError as e:
                    print(f"{e}, sleeping for {time_to_sleep} seconds")
                    time.sleep(time_to_sleep)
                    time_to_sleep +=1
                except JSONDecodeError:
                    time_to_sleep +=1
                finally:
                    if MAX_NUMBER_OF_RETRIES == time_to_sleep:
                        print("Timeout")
                        break
    return result

def main():
    if USE_PREACCELERATION:
        ts.preaccelerate(timeout=5)
    data = load_raw_data_from_path(DATA_NLP2023)[:150]
    data.to_csv(join(DATA_DIR, "german_negative.csv"))

    data["comment"] = translate_series(data["comment"], TranslationType.GTP)
    print(data)
    data.to_csv(join(DATA_DIR, "polish_negative.csv"))

if __name__ == "__main__":
    main()
    
