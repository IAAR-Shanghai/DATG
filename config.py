"""Configurations for the generation and evaluation"""


# ─── Configs For Generation ───────────────────────────────────────────────────

RESULTS_DIR = "./results"

MODEL_PATHS = {
    "phi2_3B_Base": "/mnt/hysc/models/phi-2",
    "llama2_13B_Base": "/mnt/hysc/models/Llama-2-13b-hf",
    "falcon_7B_Base": "/mnt/hysc/models/falcon-7b",
    "opt_7B_Base": "/mnt/hysc/models/opt-6.7b",
    "alpaca_7B_Base": "/mnt/hysc/models/alpaca-7b-hf",
}

TASK_CONFIGURATIONS = {
    "toxicMitigation": {
        "classifier_path": "./model/internal_classifier/toxic_classifier",
        "base_model_path": '/mnt/hysc/models/bge-large-en-v1.5',
        "positive_prompt": "The following text doesn't perpetuate negative stereotypes, isn't threatening or sexually explicit, and does not contain profane language.",
        "negative_prompt": "The following text perpetuate negative stereotypes, is threatening or sexually explicit, and contain profane language.",
        "result_dir": RESULTS_DIR,
        "data_path": {
            "toxicRandom_test": "./data/test_task_data/toxic/random_realtoxicityprompts.json",
            "toxicTop_test": "./data/test_task_data/toxic/top_realtoxicityprompts.json",
        }
    },

    "2Positive": {
        "classifier_path": "./model/internal_classifier/sentiment_classifier",
        "base_model_path": '/mnt/hysc/models/bge-large-en-v1.5',
        "positive_prompt": "The following text exhibits a very positive sentiment and/or opinion.",
        "negative_prompt": "The following text exhibits a very negative sentiment and/or opinion.",
        "result_dir": RESULTS_DIR,
        "data_path": {
            "NegToPos_test": "./data/test_task_data/sentiment/negative_sst5.json"
        }
    },
    
    "2Negative": {
        "classifier_path": "./model/internal_classifier/sentiment_classifier",
        "base_model_path": '/mnt/hysc/models/bge-large-en-v1.5',
        "positive_prompt": "The following text exhibits a very negative sentiment and/or opinion.",
        "negative_prompt": "The following text exhibits a very positive sentiment and/or opinion.",
        "result_dir": RESULTS_DIR,
        "data_path": {
            "PosToNeg_test": "./data/test_task_data/sentiment/positive_sst5.json"
        }
    },
}

GENERATION_CONFIGS = {
    "max_new_tokens": 32,
    "do_sample": True,
    "top_k": 200,
    "top_p": 0.9,
    "temperature": 0.7
}


# ─── Configs For Evaluation ───────────────────────────────────────────────────

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = RESULTS_DIR
GOOGLE_API_KEYs = [
    "i_am_a_google_api_key_i_am_a_google_api",
    "i_am_a_google_api_key_i_am_a_google_api",
    "i_am_a_google_api_key_i_am_a_google_api",
    "i_am_a_google_api_key_i_am_a_google_api",
]


EMB_MODEL_PATH = '/mnt/hysc/models/paraphrase-multilingual-MiniLM-L12-v2'
GPT2_MODEL_PATH = '/mnt/hysc/models/gpt2-large'

CLASSIFIER_PATHS = {
    'NegToPos': {
        'model_path': './model/external_discriminator/sentiment_discriminator',
        'base_model_path': '/mnt/hysc/models/roberta-base'
    },
    'PosToNeg': {
        'model_path': './model/external_discriminator/sentiment_discriminator',
        'base_model_path': '/mnt/hysc/models/roberta-base'
    }
}
