import re

import torch
import numpy as np
import torch.nn.functional as F
from .config import DanGamConfig
from collections import defaultdict
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
# lingua_languages = Language.all()
lingua_languages = [Language.ENGLISH, Language.KOREAN]
lingua_detection = LanguageDetectorBuilder.from_languages(*lingua_languages).build()


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


emotion_labels_ko = {
    0: "rage", 1: "whining", 2: "frustrated", 3: "irritated",
    4: "defensive", 5: "spiteful", 6: "restless", 7: "disgusted",
    8: "displeased", 9: "annoyed", 10: "sad", 11: "disappointed",
    12: "heartbreaking", 13: "regret", 14: "depressed", 15: "paralyzed",
    16: "pessimistic", 17: "tearful", 18: "dejected", 19: "disillusioned",
    20: "anxious", 21: "frightened", 22: "stressful", 23: "vulnerable",
    24: "confused", 25: "embarrassing", 26: "skeptical", 27: "worried",
    28: "cautious", 29: "nervous", 30: "wound", 31: "jealous",
    32: "betrayed", 33: "isolated", 34: "shocked", 35: "poor",
    36: "victimized", 37: "unfair", 38: "afflicted", 39: "abandoned",
    40: "bewildered", 41: "beleaguered", 42: "self conscious", 43: "lonely",
    44: "inferiority", 45: "guilty", 46: "ashamed", 47: "abominable",
    48: "pathetic", 49: "baffled", 50: "joyful", 51: "grateful",
    52: "trusting", 53: "comfortable", 54: "satisfying", 55: "thrilled",
    56: "relaxed", 57: "relieved", 58: "excited", 59: "confident"
}

emotion_labels_en_ko = {
    "rage": "분노", "whining": "툴툴대는", 'frustrated': "좌절한", 'irritated': "짜증내는",
    'defensive': "방어적인", 'spiteful': "악의적인", 'restless': "안달하는", 'disgusted': "구역질 나는",
    'displeased': "노여워하는", 'annoyed': "성가신", 'sad': "슬픔", 'disappointed': "실망한",
    'heartbreaking': "비통한", 'regret': "후회되는", 'depressed': "우울한", 'paralyzed': "마비된",
    'pessimistic': "염세적인", 'tearful': "눈물이 나는", 'dejected': "낙담한", 'disillusioned': "환멸을 느끼는",
    'anxious': "불안", 'frightened': "두려운", 'stressful': "스트레스 받는", 'vulnerable': "취약한",
    'confused': "혼란스러운", 'embarrassing': "당혹스러운", 'skeptical': "회의적인", 'worried': "걱정스러운",
    'cautious': "조심스러운", 'nervous': "초조한", 'wound': "상처", 'jealous': "질투하는",
    'betrayed': "배신당한", 'isolated': "고립된", 'shocked': "충격 받은", 'poor': "가난한 불우한",
    'victimized': "희생된", 'unfair': "억울한", 'afflicted': "괴로워하는", 'abandoned': "버려진",
    'bewildered': "당황", 'beleaguered': "고립된 당황한)", 'self conscious': "남의 시선을 의식하는", 'lonely': "외로운",
    'inferiority': "열등감", "guilty": "죄책감의", 'ashamed': "부끄러운", 'abominable': "혐오스러운",
    'pathetic': "한심한", 'baffled': "혼란스러운 (당황한)", 'joyful': "기쁨", 'grateful': "감사하는",
    'trusting': "신뢰하는", 'comfortable': "편안한", 'satisfying': "만족스러운", 'thrilled': "흥분",
    'relaxed': "느긋한", 'relieved': "안도하는", 'excited': "신이 난", 'confident': "자신하는"
}
neg_tag_ko = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
           10, 11, 12, 13, 14, 15, 16, 18, 19,
           20, 21, 22, 23, 24, 25, 26, 27,
           30, 32, 34, 35, 36, 37, 38, 39,
           40, 41, 43, 44, 45, 46, 47, 48, 49]
pos_tag_ko = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
neut_tag_ko = [17, 28, 29, 31, 33, 42]

emotion_labels_en = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance",
    4: "approval", 5: "caring", 6: "confusion", 7: "curiosity",
    8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
    12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude",
    16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "optimism", 21: "pride", 22: "realization", 23: "relief",
    24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}
neg_tag_en = [2, 3, 6, 9, 10, 11, 12, 14, 16, 24, 25]
pos_tag_en = [0, 1, 4, 5, 8, 13, 15, 17, 18, 20, 21, 23, 26]
neut_tag_en = [7, 19, 22, 27]


# noinspection PyTypeChecker
class DanGam:
    """
    The DanGam class is designed for advanced emotion analysis in textual data. It utilizes
    sophisticated NLP models to accurately identify and categorize emotions at both sentence
    and word levels, providing nuanced insights into the emotional tone of text.

    Attributes:
        cfg (DotDict): Configuration settings for the DanGam instance.
        ko_model_name (AutoModelForSequenceClassification): Primary model for general Korean emotion classification.
        ko_sub_model_name (AutoModelForSequenceClassification): Secondary model for more detailed English emotion classification.
        ko_word_senti_model_name (AutoModel): Model dedicated to Korean word-level sentiment analysis.
        en_model_name (AutoModelForSequenceClassification): Primary model for general English emotion classification.
        en_sub_model_name (AutoModelForSequenceClassification): Secondary model for more detailed English emotion classification.
        en_word_senti_model_name (AutoModel): Model dedicated to English word-level sentiment analysis.

    Methods:
        cfg_info(): Displays the current configuration settings of the DanGam instance.
        check_default(): Outputs the default configuration values for reference.
        check_config(): Returns the current configuration of DanGam as a dictionary.
        chunk_text(text, max_length): Splits a given text into manageable chunks for model processing.
        get_emotion(...): Analyzes and returns the overall emotion of a sentence based on model predictions.
        normalize_sentiment_scores(dissimilarities): Normalizes sentiment dissimilarity scores.
        word_emotions(sentence, emotion_label, specific_emotion_label): Segments words in a sentence and assigns them emotions.
        noun_emotions(sentence, noun_list): Analyzes and categorizes emotions associated with specific nouns in a sentence.

    Usage:
        - Initialize the class with default or custom configuration.
        - Use its methods to perform detailed emotion segmentation and analysis in textual content.
    """
    VERSION = '0.0.136'
    CREATED_BY = 'jasonheesanglee\thttps://github.com/jasonheesanglee'

    def __init__(self, cfg=None):
        """

    Args:
        cfg (dict, optional): A dictionary containing custom configuration settings. If provided, these settings will
                              override the default configuration. The configuration should specify details like model names,
                              tokenizer preferences, column names for data processing, and other relevant parameters.
                              If None, the default configuration is used.

    Attributes Initialization:
        - Initializes the model and tokenizer based on the provided or default configuration.
        - Sets up various class attributes like separator tokens and emotion labels.
        - Prepares lists of tags for categorizing emotions as positive, neutral, or negative.
    """
        print('''
CAUTION
This logic performs the best with the models that are pretrained with
AI HUB Dataset https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86
or any Dataset that has 60 sentiment tags listed as described in https://huggingface.co/hun3359/klue-bert-base-sentiment/blob/main/config.json\n
You can also modify configuration by calling update_config()
        '''
              )
        if cfg is not None:
            self.cfg = DanGamConfig(cfg)
            # print('why?')
        else:
            self.cfg = DanGamConfig()
            # print('good')
        self.initialize_models()
        self.sep_token = ' >>>> '
        self.emotion_label_ko = emotion_labels_ko
        self.emotion_label_en = emotion_labels_en
        self.pos_tag_ko = pos_tag_ko
        self.neut_tag_ko = neut_tag_ko
        self.neg_tag_ko = neg_tag_ko
        self.pos_tag_en = pos_tag_en
        self.neut_tag_en = neut_tag_en
        self.neg_tag_en = neg_tag_en

    def config_info(self):
        """
        Prints the current configuration information of the DanGam.
        Includes details about the models used, text and emotion column names, and other settings.
        """
        print(f"""
'ko_model_name' - The model that will run through the first loop of the Korean sentence segmentation.
'ko_sub_model_name' - The model that will run through the second loop of the Korean sentence segmentation.
'ko_word_senti_model_name' - The model that will through the loop of the Korean word segmentation.
'en_model_name' - The model that will run through the first loop of the English sentence segmentation.
'en_sub_model_name' - The model that will run through the second loop of the English sentence segmentation.
'en_word_senti_model_name' - The model that will through the loop of the English word segmentation.
'truncation' - Bool : Turning on and off Truncation throughout the module.\n 
'sent_emo_col' - The column name of sentence emotion (pos/neg/neut) you want this module to set.\n
'sent_spec_emo_col' - The column name of sentence emotion (pos/neg/neut) you want this module to set.\n
'max_length' - Max length for chunk_text.
        """)

    def initialize_models(self):
        """
        Initialize models based on the current configuration.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ### Korean

        self.ko_tokenizer = AutoTokenizer.from_pretrained(self.cfg.ko_model_name)
        self.ko_model = AutoModelForSequenceClassification.from_pretrained(self.cfg.ko_model_name)
        self.ko_model.to(self.device)

        self.ko_sub_tokenizer = AutoTokenizer.from_pretrained(self.cfg.ko_sub_model_name)
        self.ko_sub_model = AutoModelForSequenceClassification.from_pretrained(self.cfg.ko_sub_model_name)
        self.ko_sub_model.to(self.device)

        self.ko_word_senti_tokenizer = AutoTokenizer.from_pretrained(self.cfg.ko_word_senti_model_name)
        self.ko_word_senti_model = AutoModel.from_pretrained(self.cfg.ko_word_senti_model_name)
        self.ko_word_senti_model.to(self.device)

        ### English

        self.en_tokenizer = AutoTokenizer.from_pretrained(self.cfg.en_model_name)
        self.en_model = AutoModelForSequenceClassification.from_pretrained(self.cfg.en_model_name)
        self.en_model.to(self.device)

        self.en_sub_tokenizer = AutoTokenizer.from_pretrained(self.cfg.en_sub_model_name, add_prefix_space=True)
        self.en_sub_model = AutoModelForSequenceClassification.from_pretrained(self.cfg.en_sub_model_name)
        self.en_sub_model.to(self.device)

        self.en_word_senti_tokenizer = AutoTokenizer.from_pretrained(self.cfg.en_word_senti_model_name,
                                                                     add_prefix_space=True)
        self.en_word_senti_model = AutoModel.from_pretrained(self.cfg.en_word_senti_model_name)
        self.en_word_senti_model.to(self.device)

        ### model configurations

        self.truncation = self.cfg.truncation
        self.padding = self.cfg.padding
        self.max_length = self.cfg.max_length
        self.align_th = self.cfg.alignment_threshold
        self.emo_th = self.cfg.emotion_threshold
        self.emo_wt_reach_th = self.cfg.emotion_weight_reach_threshold
        self.emo_wt_n_reach_th = self.cfg.emotion_weight_not_reach_threshold
        self.spec_wt_reach_th = self.cfg.specific_weight_reach_threshold
        self.spec_wt_n_reach_th = self.cfg.specific_weight_not_reach_threshold
        self.noun_th = self.cfg.noun_threshold

    def lang_conf_rate(self, text):
        conf_rate = {}
        try:
            for word in text.split():
                possible_lang = lingua_detection.compute_language_confidence_values(word)
                for confidence in possible_lang:
                    try:
                        conf_rate[confidence.language.name] += confidence.value
                    except KeyError:
                        conf_rate[confidence.language.name] = confidence.value

            conf_rate_per = {}
            for key, value in conf_rate.items():
                if sum(conf_rate.values()) != 0:
                    conf_rate_per[key] = value / sum(conf_rate.values())
                else:
                    conf_rate_per[key] = 0
            return conf_rate_per
        except:
            pass

    def lang_detector(self, text):
        text_lang = None
        try:
            lang_ratio = self.lang_conf_rate(text)

            max_rate = 0
            lang_temp = {}
            for lang, rate in lang_ratio.items():
                if rate > max_rate:
                    max_rate = rate
                    text_lang = lang
                    lang_temp = {lang: rate}
                elif rate == max_rate:
                    lang_temp[lang] = rate
            if 'JAPANESE' in lang_temp and 'CHINESE' in lang_temp and lang_temp['JAPANESE'] == lang_temp['CHINESE']:
                text_lang = 'JAPANESE'
            return text_lang
        except:
            return 'unidentified'

    def update_config(self, new_config):
        """
        Update the configuration of DanGam and reinitialize components as necessary.

        Args:
            new_cfg (dict): A dictionary containing the new configuration settings.
        """
        self.cfg = DanGamConfig(new_config)
        self.initialize_models()

    def check_default(self):
        print(self.cfg.check_default())

    def check_config(self):
        return self.cfg.get_config()

    def chunk_text(self, text: str, max_length=512, language='KOREAN'):
        """
        Splits a given text into chunks for processing.
        Ensures that each chunk is within the specified maximum length.

        Args:
            text (str): The text to be chunked.
            max_length (int): Maximum length of each text chunk.

        Yields:
            str: Chunks of the original text.
        """
        if language == 'KOREAN':
            tokens = self.ko_tokenizer.tokenize(text)

        elif language == 'ENGLISH':
            tokens = self.en_tokenizer.tokenize(text)

        chunk_size = max_length - 2
        for i in range(0, len(tokens), chunk_size):
            yield ' '.join(tokens[i:i + chunk_size])

    def get_emotion(self,
                    sentence: str,
                    original_emotion: str = None,
                    default_specific_emotion: str = None,
                    normalized_emotion: str = None,
                    language: str = None,
                    ):
        """
        Determines the overall emotion of a given sentence by analyzing it in chunks.
        Considers both the general and specific emotions to enhance accuracy.

        Args:
            sentence : The sentence to extract emotion from.
            original_emotion (str) -> optional : The pre-segmented emotion (positive, negative, neutral)
            default_specific_emotion (str) -> optional : The pre-segmented specific emotion (love, thrilled, happy, sad, etc..)
            normalized_emotion (str) -> optional : Normalized User input emotion (good food, bad person, lovely day, etc..)

        Returns:
            emotion (str) : string of overall emotion of the sentence. (positive, neutral, negative)
            specific_emotion (str) : string of specific emotion of the sentence. (one out of 60 emotions)
        """
        pattern = '[^a-zA-Zㄱ-ㅎㅏ-ㅢ가-힣0-9+]'
        sentence = re.sub(pattern, ' ', sentence)
        # if language == None:
        #     language = self.lang_detector(sentence)
        if original_emotion != None or default_specific_emotion != None or normalized_emotion != None:
            chunks = list(self.chunk_text(sentence, language=language))
            sum_prob = None
            num_chunks = 0

            for chunk in chunks:
                if language == 'KOREAN':
                    if normalized_emotion != '-':
                        inputs = self.ko_tokenizer(normalized_emotion + self.sep_token + chunk, return_tensors='pt',
                                                   padding=self.padding, truncation=self.truncation,
                                                   max_length=self.max_length)
                    else:
                        inputs = self.ko_tokenizer(default_specific_emotion + self.sep_token + chunk,
                                                   return_tensors='pt',
                                                   padding=self.padding, truncation=self.truncation,
                                                   max_length=self.max_length)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        logits = self.ko_model(**inputs).logits

                elif language == 'ENGLISH':
                    if normalized_emotion != '-':
                        inputs = self.en_tokenizer(normalized_emotion + self.sep_token + chunk, return_tensors='pt',
                                                   padding=self.padding, truncation=self.truncation,
                                                   max_length=self.max_length)
                    else:
                        inputs = self.en_tokenizer(default_specific_emotion + self.sep_token + chunk,
                                                   return_tensors='pt',
                                                   padding=self.padding, truncation=self.truncation,
                                                   max_length=self.max_length)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        logits = self.en_model(**inputs).logits

                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                if sum_prob is None:
                    sum_prob = probabilities
                else:
                    sum_prob += probabilities
                num_chunks += 1

            if num_chunks != 0 and sum_prob is not None:
                avg_prob = sum_prob / num_chunks
                sentiment_idx = avg_prob.argmax().item()
            else:
                avg_prob = 0
                sentiment_idx = avg_prob

            if language == 'KOREAN':
                if sentiment_idx in self.neg_tag_ko:
                    emotion = 'negative'
                    specific_emotion = self.emotion_label_ko[sentiment_idx]
                elif sentiment_idx in self.pos_tag_ko:
                    emotion = 'positive'
                    specific_emotion = self.emotion_label_ko[sentiment_idx]
                else:
                    emotion = 'neutral'
                    specific_emotion = self.emotion_label_ko[sentiment_idx]

            elif language == 'ENGLISH':
                if sentiment_idx in self.neg_tag_en:
                    emotion = 'negative'
                    specific_emotion = self.emotion_label_en[sentiment_idx]
                elif sentiment_idx in self.pos_tag_en:
                    emotion = 'positive'
                    specific_emotion = self.emotion_label_en[sentiment_idx]
                else:
                    emotion = 'neutral'
                    specific_emotion = self.emotion_label_en[sentiment_idx]


            if original_emotion.strip() == emotion.strip():
                return emotion, specific_emotion
            else:
                emotion_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

            for chunk in chunks:
                if language == 'KOREAN':
                    if normalized_emotion != '-':
                        inputs = self.ko_sub_tokenizer(normalized_emotion + self.sep_token + chunk,
                                                       return_tensors='pt',
                                                       padding=self.padding, truncation=self.truncation,
                                                       max_length=self.max_length)
                    else:
                        inputs = self.ko_sub_tokenizer(default_specific_emotion + self.sep_token + chunk,
                                                       return_tensors='pt',
                                                       padding=self.padding, truncation=self.truncation,
                                                       max_length=self.max_length)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = self.ko_sub_model(**inputs).logits

                elif language == 'ENGLISH':
                    if normalized_emotion != '-':
                        inputs = self.en_sub_tokenizer(normalized_emotion + self.sep_token + chunk,
                                                       return_tensors='pt',
                                                       padding=self.padding, truncation=self.truncation,
                                                       max_length=self.max_length)
                    else:
                        inputs = self.en_sub_tokenizer(default_specific_emotion + self.sep_token + chunk,
                                                       return_tensors='pt',
                                                       padding=self.padding, truncation=self.truncation,
                                                       max_length=self.max_length)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = self.en_sub_model(**inputs).logits

                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                emotion_pred = probabilities.argmax().item()  ##### majority votes

                if emotion_pred == 0:
                    emotion_counts['negative'] += 1
                elif emotion_pred == 1:
                    emotion_counts['positive'] += 1
                else:
                    emotion_counts['neutral'] += 1

            if emotion_counts['neutral'] > emotion_counts['positive'] + emotion_counts['negative']:
                emotion = 'neutral'
            elif emotion_counts['negative'] > emotion_counts['positive']:
                emotion = 'negative'
            elif emotion_counts['negative'] < emotion_counts['positive']:
                emotion = 'positive'
            else:
                emotion = 'neutral'

            specific_emotion = normalized_emotion
            if original_emotion.strip() == emotion.strip():
                return emotion, specific_emotion
            else:  ############ 여기에 아예 catboost를 써볼까....??
                sum_prob = None
                num_chunks = 0
                for chunk in chunks:
                    if language == 'KOREAN':
                        if normalized_emotion != '-':
                            inputs = self.ko_sub_tokenizer(normalized_emotion + self.sep_token + chunk,
                                                           return_tensors='pt',
                                                           padding=self.padding, truncation=self.truncation,
                                                           max_length=self.max_length)
                        else:
                            inputs = self.ko_sub_tokenizer(default_specific_emotion + self.sep_token + chunk,
                                                           return_tensors='pt', padding=self.padding,
                                                           truncation=self.truncation, max_length=self.max_length)

                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            logits = self.ko_sub_model(**inputs).logits

                    elif language == 'ENGLISH':
                        if normalized_emotion != '-':
                            inputs = self.en_sub_tokenizer(normalized_emotion + self.sep_token + chunk,
                                                           return_tensors='pt',
                                                           padding=self.padding, truncation=self.truncation,
                                                           max_length=self.max_length)
                        else:
                            inputs = self.en_sub_tokenizer(default_specific_emotion + self.sep_token + chunk,
                                                           return_tensors='pt', padding=self.padding,
                                                           truncation=self.truncation, max_length=self.max_length)

                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            logits = self.en_sub_model(**inputs).logits

                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    if sum_prob is None:
                        sum_prob = probabilities
                    else:
                        sum_prob += probabilities
                    num_chunks += 1

                if num_chunks != 0 and sum_prob is not None:
                    avg_prob = sum_prob / num_chunks
                    emotion_ = avg_prob.argmax().item()

                else:
                    emotion_ = None

                if emotion_ == 0:
                    emotion = 'negative'

                elif emotion_ == 1:
                    emotion = 'positive'
                else:
                    emotion = 'neutral'
                return emotion, specific_emotion

        else:
            chunks = list(self.chunk_text(sentence, max_length=int(self.max_length / 5), language=language))
            sum_prob = None
            num_chunks = 0

            for chunk_no in range(len(chunks)):
                if language == 'KOREAN':
                    if chunk_no != len(chunks) - 1:
                        inputs = self.ko_tokenizer(chunks[chunk_no] + self.sep_token + chunks[chunk_no + 1],
                                                   return_tensors='pt',
                                                   padding=self.padding, truncation=self.truncation,
                                                   max_length=self.max_length)
                    else:
                        inputs = self.ko_tokenizer(chunks[chunk_no] + self.sep_token, return_tensors='pt',
                                                   padding=self.padding, truncation=self.truncation,
                                                   max_length=self.max_length)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = self.ko_model(**inputs).logits

                elif language == 'ENGLISH':
                    if chunk_no != len(chunks) - 1:
                        inputs = self.en_tokenizer(chunks[chunk_no] + self.sep_token + chunks[chunk_no + 1],
                                                   return_tensors='pt',
                                                   padding=self.padding, truncation=self.truncation,
                                                   max_length=self.max_length)
                    else:
                        inputs = self.en_tokenizer(chunks[chunk_no] + self.sep_token, return_tensors='pt',
                                                   padding=self.padding, truncation=self.truncation,
                                                   max_length=self.max_length)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = self.en_model(**inputs).logits

                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                if sum_prob is None:
                    sum_prob = probabilities
                else:
                    sum_prob += probabilities
                num_chunks += 1

            if num_chunks != 0 and sum_prob is not None:
                avg_prob = sum_prob / num_chunks
                sentiment_idx = avg_prob.argmax().item()
            else:
                avg_prob = 0
                sentiment_idx = avg_prob
            if language == 'KOREAN':
                if sentiment_idx in self.neg_tag_ko:
                    emotion = 'negative'
                    specific_emotion = self.emotion_label_ko[sentiment_idx]
                elif sentiment_idx in self.pos_tag_ko:
                    emotion = 'positive'
                    specific_emotion = self.emotion_label_ko[sentiment_idx]
                else:
                    emotion = 'neutral'
                    specific_emotion = self.emotion_label_ko[sentiment_idx]

            elif language == 'ENGLISH':
                if sentiment_idx in self.neg_tag_en:
                    emotion = 'negative'
                    specific_emotion = self.emotion_label_en[sentiment_idx]
                elif sentiment_idx in self.pos_tag_en:
                    emotion = 'positive'
                    specific_emotion = self.emotion_label_en[sentiment_idx]
                else:
                    emotion = 'neutral'
                    specific_emotion = self.emotion_label_en[sentiment_idx]

            sum_prob = None
            num_chunks = 0

            for chunk in chunks:
                if language == 'KOREAN':
                    inputs = self.ko_tokenizer(emotion + self.sep_token + specific_emotion + self.sep_token + chunk,
                                               return_tensors='pt',
                                               padding=self.padding, truncation=self.truncation,
                                               max_length=self.max_length)

                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = self.ko_model(**inputs).logits


                elif language == 'ENGLISH':
                    inputs = self.en_tokenizer(emotion + self.sep_token + specific_emotion + self.sep_token + chunk,
                                               return_tensors='pt',
                                               padding=self.padding, truncation=self.truncation,
                                               max_length=self.max_length)

                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        logits = self.en_model(**inputs).logits

                probabilities = torch.nn.functional.softmax(logits, dim=-1)

                if sum_prob is None:
                    sum_prob = probabilities
                else:
                    sum_prob += probabilities
                num_chunks += 1

            if num_chunks != 0 and sum_prob is not None:
                avg_prob = sum_prob / num_chunks
                sentiment_idx = avg_prob.argmax().item()
            else:
                avg_prob = 0
                sentiment_idx = avg_prob
            if language == 'KOREAN':
                if sentiment_idx in self.neg_tag_ko:
                    new_emotion = 'negative'
                    new_specific_emotion = self.emotion_label_ko[sentiment_idx]
                elif sentiment_idx in self.pos_tag_ko:
                    new_emotion = 'positive'
                    new_specific_emotion = self.emotion_label_ko[sentiment_idx]
                else:
                    new_emotion = 'neutral'
                    new_specific_emotion = self.emotion_label_ko[sentiment_idx]

            elif language == 'ENGLISH':
                if sentiment_idx in self.neg_tag_en:
                    new_emotion = 'negative'
                    new_specific_emotion = self.emotion_label_en[sentiment_idx]
                elif sentiment_idx in self.pos_tag_en:
                    new_emotion = 'positive'
                    new_specific_emotion = self.emotion_label_en[sentiment_idx]
                else:
                    new_emotion = 'neutral'
                    new_specific_emotion = self.emotion_label_en[sentiment_idx]

            if new_emotion.strip() == emotion.strip():
                return new_emotion, new_specific_emotion
            else:
                # return emotion, specific_emotion
                emotion_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

                for chunk in chunks:
                    if language == 'KOREAN':
                        inputs = self.ko_sub_tokenizer(
                            emotion + self.sep_token + specific_emotion + self.sep_token + chunk, return_tensors='pt',
                            padding=self.padding, truncation=self.truncation, max_length=self.max_length)

                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            logits = self.ko_sub_model(**inputs).logits

                    if language == 'ENGLISH':
                        inputs = self.en_sub_tokenizer(
                            emotion + self.sep_token + specific_emotion + self.sep_token + chunk, return_tensors='pt',
                            padding=self.padding, truncation=self.truncation, max_length=self.max_length)

                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            logits = self.en_sub_model(**inputs).logits

                    probabilities = torch.nn.functional.softmax(logits, dim=-1)
                    emotion_pred = probabilities.argmax().item()  ##### majority votes
                    if emotion_pred == 0:
                        emotion_counts['negative'] += 1
                    elif emotion_pred == 1:
                        emotion_counts['positive'] += 1
                    else:
                        emotion_counts['neutral'] += 1

                if emotion_counts['neutral'] > emotion_counts['positive'] + emotion_counts['negative']:
                    new_emotion = 'neutral'
                elif emotion_counts['negative'] > emotion_counts['positive']:
                    new_emotion = 'negative'
                elif emotion_counts['negative'] < emotion_counts['positive']:
                    new_emotion = 'positive'
                else:
                    new_emotion = 'neutral'

                if new_emotion.strip() == emotion.strip():
                    return new_emotion, new_specific_emotion
                else:  ############ 여기에 아예 catboost를 써볼까....??
                    sum_prob = None
                    num_chunks = 0
                    for chunk in chunks:
                        if language == 'KOREAN':
                            inputs = self.ko_sub_tokenizer(
                                emotion + self.sep_token + specific_emotion + self.sep_token + chunk,
                                return_tensors='pt',
                                padding=self.padding, truncation=self.truncation, max_length=self.max_length)

                            inputs = {k: v.to(self.device) for k, v in inputs.items()}

                            with torch.no_grad():
                                logits = self.ko_sub_model(**inputs).logits

                        if language == 'ENGLISH':
                            inputs = self.en_sub_tokenizer(
                                emotion + self.sep_token + specific_emotion + self.sep_token + chunk,
                                return_tensors='pt',
                                padding=self.padding, truncation=self.truncation, max_length=self.max_length)

                            inputs = {k: v.to(self.device) for k, v in inputs.items()}

                            with torch.no_grad():
                                logits = self.en_sub_model(**inputs).logits

                        probabilities = torch.nn.functional.softmax(logits, dim=-1)
                        if sum_prob is None:
                            sum_prob = probabilities
                        else:
                            sum_prob += probabilities
                        num_chunks += 1

                    if num_chunks != 0 and sum_prob is not None:
                        avg_prob = sum_prob / num_chunks
                        emotion_ = avg_prob.argmax().item()

                    else:
                        emotion_ = None

                    if emotion_ == 0:
                        new_emotion = 'negative'

                    elif emotion_ == 1:
                        new_emotion = 'positive'
                    else:
                        new_emotion = 'neutral'

                    return new_emotion, new_specific_emotion

    def get_word_embeddings(self, sentence, language):
        """
        Retrieves word embeddings for a given sentence using the specified tokenizer and model.

        Args:
            sentence (str): The sentence for which to get word embeddings.
            max_length (int): Maximum length for tokenization.

        Returns:
            tuple: A tuple containing word embeddings and offset mappings.
        """
        # pattern = '[^a-zA-Zㄱ-ㅎㅏ-ㅢ가-힣0-9+]'
        # sentence = re.sub(pattern, ' ', sentence)

        if language == 'KOREAN':
            inputs = self.ko_word_senti_tokenizer(sentence, return_tensors="pt", max_length=self.max_length,
                                                  truncation=self.truncation, padding='max_length',
                                                  return_offsets_mapping=True)
        if language == 'ENGLISH':
            sentence = sentence.split()
            # print(f'here here here : {sentence}')
            inputs = self.en_word_senti_tokenizer(sentence, return_tensors="pt", max_length=self.max_length,
                                                  truncation=self.truncation, padding='max_length',
                                                  return_offsets_mapping=True,
                                                  is_split_into_words=True)  # , return_special_tokens_mask=False)
        offset_mapping = inputs.pop('offset_mapping')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # print(f'inputs \n{inputs}\n')
        if language == 'KOREAN':
            with torch.no_grad():
                outputs = self.ko_word_senti_model(**inputs)
        if language == 'ENGLISH':
            with torch.no_grad():
                outputs = self.en_word_senti_model(**inputs)
        # print(f'outputs \n{outputs}\n')
        embeddings = outputs.last_hidden_state
        # print(f'embeddings\n{embeddings}\n')
        # print(f'offset_mapping\n{offset_mapping}\n')
        return embeddings, offset_mapping

    def get_sentence_embedding(self, sentence, language):
        """
        Computes the embedding of a sentence by averaging the embeddings of its constituent tokens.

        Args:
            sentence (str): The sentence to compute embedding for.
            max_length (int): Maximum length for tokenization.

        Returns:
            torch.Tensor: The computed sentence embedding.
        """
        # pattern = '[^a-zA-Zㄱ-ㅎㅏ-ㅢ가-힣0-9+]'
        # sentence = re.sub(pattern, ' ', sentence)
        if language == 'KOREAN':
            inputs = self.ko_word_senti_tokenizer(sentence, return_tensors="pt", max_length=self.max_length,
                                                  truncation=self.truncation,
                                                  padding="max_length")
        if language == 'ENGLISH':
            inputs = self.en_word_senti_tokenizer(sentence, return_tensors="pt", max_length=self.max_length,
                                                  truncation=self.truncation,
                                                  padding="max_length")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if language == 'KOREAN':
            with torch.no_grad():
                outputs = self.ko_word_senti_model(**inputs)
        if language == 'ENGLISH':
            with torch.no_grad():
                outputs = self.en_word_senti_model(**inputs)

        # Average the token embeddings to get the sentence embedding
        sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding

    def get_emotion_embedding(self, emotion, language):
        """
        Computes the embedding for a general emotion label.

        Args:
            emotion (str): The emotion label for which to compute the embedding.

        Returns:
            torch.Tensor: The embedding of the specified emotion.
        """
        # pattern = '[^a-zA-Zㄱ-ㅎㅏ-ㅢ가-힣0-9+]'
        # sentence = re.sub(pattern, ' ', emotion)
        if language == 'KOREAN':
            inputs = self.ko_word_senti_tokenizer(emotion, return_tensors='pt', padding=self.padding,
                                                  truncation=self.truncation, max_length=self.max_length)
        elif language == 'ENGLISH':
            inputs = self.en_word_senti_tokenizer(emotion, return_tensors='pt', padding=self.padding,
                                                  truncation=self.truncation, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if language == 'KOREAN':
            with torch.no_grad():
                outputs = self.ko_word_senti_model(**inputs)

        if language == 'ENGLISH':
            with torch.no_grad():
                outputs = self.en_word_senti_model(**inputs)

        return outputs.last_hidden_state.mean(dim=1)

    def get_specific_emotion_embedding(self, specific_emotion, language):
        """
        Computes the embedding for a specific emotion label.

        Args:
            specific_emotion (str): The specific emotion label for which to compute the embedding.

        Returns:
            torch.Tensor: The embedding of the specified specific emotion.
        """
        # pattern = '[^a-zA-Zㄱ-ㅎㅏ-ㅢ가-힣0-9+]'
        # sentence = re.sub(pattern, ' ', specific_emotion)
        if language == 'KOREAN':
            inputs = self.ko_word_senti_tokenizer(specific_emotion, return_tensors='pt', padding=self.padding,
                                                  truncation=self.truncation, max_length=self.max_length)
        if language == 'ENGLISH':
            inputs = self.en_word_senti_tokenizer(specific_emotion, return_tensors='pt', padding=self.padding,
                                                  truncation=self.truncation, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if language == 'KOREAN':
            with torch.no_grad():
                outputs = self.ko_word_senti_model(**inputs)

        if language == 'ENGLISH':
            with torch.no_grad():
                outputs = self.en_word_senti_model(**inputs)

        return outputs.last_hidden_state.mean(dim=1)

    def calculate_vector_differences(self, sentence, emotion, specific_emotion, language):
        """
        Calculates vector differences (dissimilarities) between words in a sentence and a combined emotion embedding.

        Args:
            sentence (str): The sentence for analysis.
            emotion (str): General emotion label for the sentence.
            specific_emotion (str): Specific emotion label for the sentence.

        Returns:
            list: A list of dissimilarity scores for each word in the sentence.
        """
        pattern = '[^a-zA-Zㄱ-ㅎㅏ-ㅢ가-힣0-9+]'
        sentence = re.sub(pattern, ' ', sentence)
        word_embeddings, offset_mapping = self.get_word_embeddings(sentence, language)
        sentence_embedding = self.get_sentence_embedding(sentence, language)
        emotion_embedding = self.get_emotion_embedding(emotion, language)
        specific_emotion_embedding = self.get_emotion_embedding(specific_emotion, language)

        general_alignment = F.cosine_similarity(sentence_embedding, emotion_embedding, dim=1)
        specific_alignment = F.cosine_similarity(sentence_embedding, specific_emotion_embedding, dim=1)

        emotion_weight = self.emo_wt_reach_th if general_alignment.item() > self.emo_th else self.emo_wt_n_reach_th
        specific_weight = self.spec_wt_reach_th if specific_alignment.item() > self.emo_th else self.spec_wt_n_reach_th
        sentence_weight = 1 - (emotion_weight + specific_weight)

        combined_embedding = (emotion_embedding * emotion_weight -
                              sentence_embedding * sentence_weight +
                              specific_emotion_embedding * specific_weight) / 2

        # combined_embedding = (sentence_embedding - emotion_embedding) / 2

        dissimilarities = []

        for word_embedding in word_embeddings.squeeze():
            sim = F.cosine_similarity(combined_embedding, word_embedding.unsqueeze(0))
            if sim > self.align_th:
                adjust_sim = sim.item()
            else:
                adjust_sim = 1 - sim.item()

            dissimilarities.append(adjust_sim)

        return dissimilarities

    def normalize_sentiment_scores(self, dissimilarities):
        """
        Normalizes sentiment dissimilarity scores to a standard range for consistency.

        Args:
            dissimilarities (list): A list of dissimilarity scores.

        Returns:
            list: Normalized sentiment scores.
        """
        scores = np.array(dissimilarities)
        mean = np.mean(scores)
        std = np.std(scores)
        if std != 0:
            normalized_scores = (scores - mean) / std
        else:
            normalized_scores = scores - mean

        # Clip scores to be within -1 and 1
        normalized_scores = np.clip(normalized_scores, -1, 1)
        return normalized_scores.tolist()

    def assign_word_sentiment_scores(self, sentence, normalized_scores, language):
        """
        Assigns sentiment scores to each word in a sentence based on normalized dissimilarities.

        Args:
            sentence (str): The sentence to assign word sentiment scores.
            normalized_scores (list): Normalized sentiment scores for each token in the sentence.

        Returns:
            dict: A dictionary mapping each word in the sentence to its sentiment score.
        """
        if language == 'KOREAN':
            encoded = self.ko_word_senti_tokenizer.encode_plus(sentence, return_tensors='pt',
                                                               truncation=self.truncation,
                                                               padding='max_length', max_length=512)
            tokenized_words = self.ko_word_senti_tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            word_scores = []
            current_word = ""
            current_score = 0
            num_tokens = 0
            for token, score in zip(tokenized_words, normalized_scores):

                if token.startswith("##"):
                    current_word += token.lstrip("##")
                    current_score += score
                    num_tokens += 1
                elif token.startswith('▁'):
                    current_word += token.lstrip("▁")
                    current_score += score
                    num_tokens += 1
                else:
                    if current_word and not current_word.startswith('['):
                        average_score = current_score / num_tokens if num_tokens > 0 else current_score
                        word_scores.append({current_word: average_score})
                    current_word = token
                    current_score = score
                    num_tokens = 1

            # Add the last word
            if current_word and not current_word.startswith('['):
                average_score = current_score / num_tokens if num_tokens > 0 else current_score
                word_scores.append({current_word: average_score})

            return word_scores

        elif language == 'ENGLISH':
            sentence = sentence.split()
            encoded = self.en_word_senti_tokenizer.encode_plus(sentence, return_tensors='pt',
                                                               truncation=self.truncation,
                                                               padding='max_length', is_split_into_words=True,
                                                               return_special_tokens_mask=True,
                                                               max_length=self.max_length)
            tokenized_words = self.en_word_senti_tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            special_tokens_mask = encoded['special_tokens_mask'][0]

            word_scores = []
            current_word = ""
            current_score = 0
            num_tokens = 0
            for token, score, is_special_token in zip(tokenized_words, normalized_scores, special_tokens_mask):
                if is_special_token:
                    continue
                if token.startswith("##"):
                    current_word += token.lstrip("##")
                    current_score += score
                    num_tokens += 1
                # elif token.startswith('Ġ'):
                #     current_word += token.lstrip("Ġ")
                #     current_score += score
                #     num_tokens += 1
                #     # current_word=''

                else:
                    if current_word and not current_word.startswith('['):
                        average_score = current_score / num_tokens if num_tokens > 0 else current_score
                        word_scores.append({current_word: average_score})
                    current_word = token
                    current_score = score
                    num_tokens = 1
                current_word = current_word.replace('Ġ', '')
            # Add the last word
            if current_word and not current_word.startswith('['):
                average_score = current_score / num_tokens if num_tokens > 0 else current_score
                word_scores.append({current_word: average_score})

            return word_scores

    def word_emotions(self, sentence: str, emotion: str = None, specific_emotion: str = None, language=None):
        """
        Segments a sentence and assigns emotions to each word based on the overall sentence emotion and specific emotion.

        Args:
            sentence (str): The sentence for segmentation.
            emotion (str) -> Optional: The general emotion of the sentence.
            specific_emotion (str) -> Optional: The specific emotion of the sentence.

        Returns:
            dict: A dictionary mapping each word in the sentence to its assigned emotion.
        """

        pattern = '[^ㄱ-ㅣ가-힣a-zA-Z0-9+]'
        sentence = re.sub(pattern, ' ', sentence)
        if emotion != None:
            emotion = re.sub(pattern, ' ', emotion)
        if specific_emotion != None:
            specific_emotion = re.sub(pattern, ' ', specific_emotion)
        if language == None:
            language = self.lang_detector(sentence)
        if emotion == None or specific_emotion == None:
            emotion, specific_emotion = self.get_emotion(sentence, language=language)
        ################################### 여기까지 ok #################################################
        dissimilarities = self.calculate_vector_differences(sentence, emotion, specific_emotion, language=language)
        # print(dissimilarities)
        norm_senti_score = self.normalize_sentiment_scores(dissimilarities)
        word_sentiment_scores = self.assign_word_sentiment_scores(sentence, norm_senti_score, language=language)
        return word_sentiment_scores

    def noun_emotions(self,
                      sentence: str,
                      noun_list: list,
                      emotion: str = None,
                      specific_emotion: str = None,
                      count: bool = False,
                      language: str = None):
        """
        Analyzes emotions associated with specific nouns within a sentence.

        Args:
            sentence (str): The sentence containing the nouns for emotion analysis.
            emotion (str) -> Optional: The general emotion of the sentence.
            specific_emotion (str) -> Optional: The specific emotion of the sentence.
            noun_list (list): A list of nouns to analyze within the sentence.
            count (bool) : True or False for switching on off counting the number of nouns in each segment.
        Returns:
            dict: A dictionary categorizing nouns into positive, neutral, and negative based on their associated emotions.
        """
        language = self.lang_detector(sentence)
        if emotion == None or specific_emotion == None:
            emotion, specific_emotion = self.get_emotion(sentence, language=language)
        word_emo_list = self.word_emotions(sentence, emotion, specific_emotion, language)

        pos = defaultdict(list)
        neut = defaultdict(list)
        neg = defaultdict(list)

        pos_count = defaultdict(int)
        neut_count = defaultdict(int)
        neg_count = defaultdict(int)

        for noun in noun_list:
            for dict_ in word_emo_list:
                for word, score in dict_.items():
                    if noun in word:
                        if score > self.noun_th:  # when positive
                            pos[noun].append(score)
                            pos_count[noun] += 1
                        elif score < -1 * self.noun_th:
                            neg[noun].append(score)
                            neg_count[noun] += 1
                        else:
                            neut[noun].append(score)
                            neut_count[noun] += 1

        positive_noun_score = {}
        neutral_noun_score = {}
        negative_noun_score = {}

        for word, score_list in dict(pos).items():
            avg_score = sum(score_list) / len(score_list)
            positive_noun_score[word] = avg_score
        for word, score_list in dict(neut).items():
            avg_score = sum(score_list) / len(score_list)
            neutral_noun_score[word] = avg_score
        for word, score_list in dict(neg).items():
            avg_score = sum(score_list) / len(score_list)
            negative_noun_score[word] = avg_score

        scores = [dict(positive_noun_score), dict(neutral_noun_score), dict(negative_noun_score)]
        counts = [dict(pos_count), dict(neut_count), dict(neg_count)]

        if count == True:
            return scores, counts
        else:
            return scores