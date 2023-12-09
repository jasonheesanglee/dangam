import re
import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from collections import defaultdict
from .config import EmotionSegmentatorConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

from konlpy.tag import Okt

okt = Okt()

cfg = {
    'model_name': 'hun3359/klue-bert-base-sentiment',
    'sub_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment',
    'word_senti_model_name': 'keonju/chat_bot',
    'text_col': 'text',
    'ori_emo_col': 'posneg',
    'default_emo_col': 'default_emotion',
    'normalized_emo_col': 'gpt_emotion',
    'truncation': True,
    'sent_emo_col': 'klue_emo',
    'sent_spec_emo_col': 'klue_specific_emo',
    'max_length': 512
}
cfg = EmotionSegmentatorConfig(cfg)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


emotion_labels = {
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
    40: "bewildered", 41: "beleaguered", 42: "self-conscious", 43: "lonely",
    44: "inferiority", 45: "guilty", 46: "ashamed", 47: "abominable",
    48: "pathetic", 49: "baffled", 50: "joyful", 51: "grateful",
    52: "trusting", 53: "comfortable", 54: "satisfying", 55: "thrilled",
    56: "relaxed", 57: "relieved", 58: "excited", 59: "confident"
}

neg_tag = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
           10, 11, 12, 13, 14, 15, 16, 18, 19,
           20, 21, 22, 23, 24, 25, 26, 27,
           30, 32, 34, 35, 36, 37, 38, 39,
           40, 41, 43, 44, 45, 46, 47, 48, 49]
pos_tag = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
neut_tag = [17, 28, 29, 31, 33, 42]


# noinspection PyTypeChecker
class EmotionSegmentator:
    VERSION = '0.1.0'
    CREATED_BY = 'jasonheesanglee\thttps://github.com/jasonheesanglee'

    def __init__(self):
        print('''
CAUTION
This logic performs the best with the models that are pretrained with
AI HUB Dataset https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86
or any Dataset that has 60 sentiment tags listed as described in
\thttps://huggingface.co/hun3359/klue-bert-base-sentiment/blob/main/config.json

You can also modify configuration by calling EmotionSegmentatorConfig()
        '''
              )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
        self.sub_tokenizer = AutoTokenizer.from_pretrained(cfg.sub_model_name)
        self.sub_model = AutoModelForSequenceClassification.from_pretrained(cfg.sub_model_name)
        self.word_senti_model = AutoModel.from_pretrained(cfg.sub_model_name)
        self.word_senti_tokenizer = AutoTokenizer.from_pretrained(cfg.sub_model_name)
        self.sep_token = ' >>>> '
        self.emotion_label = emotion_labels
        self.truncation = cfg.truncation
        self.pos_tag = pos_tag
        self.neut_tag = neut_tag
        self.neg_tag = neg_tag
        self.ori_emo_col = cfg.ori_emo_col
        self.text_col = cfg.text_col
        self.default_emo_col = cfg.default_emo_col
        self.normalized_emo_col = cfg.normalized_emo_col
        self.sent_emo_col = cfg.sent_emo_col
        self.sent_spec_emo_col = cfg.sent_spec_emo_col

    def cfg_info(self):
        print(f"""
'model_name' - The model that will run through the first loop of the sentence segmentation.
'sub_model_name' - The model that will run through the second loop of the sentence segmentation.
'text_col' - The name of the column that you want to segment the emotion.
'default_emo_col' - Pre-labeled emotion by user.
'ori_emo_col' - Pre-segmented emotions by user.
\tPerforms the best if this section is segmented into 'positive', 'negative', 'neutral'.
\tUsed for accuracy evaluation.
'normalized_emo_col' - Normalized pre-labeled emotion.
\tPerforms the best if this section is in English.
\tDirectly used from the second loop, since it will only segment positive, negative, neutral.
\tNot into 60 different emotions.
'truncation' - Bool : Turning on and off Truncation throughout the module. 
'sent_emo_col' - The column name of sentence emotion (pos/neg/neut) you want this module to set.
'sent_spec_emo_col' - The column name of sentence emotion (pos/neg/neut) you want this module to set.
'max_length' - Max length for chunk_text
        """)

    def check_default(self):
        print("""
'model_name': 'hun3359/klue-bert-base-sentiment',
\tBetter if you stick to this. This is one of the only options that segments the sentences into 60 sentiment labels.
'sub_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment'
'text_col': 'text'
\tYou need to modify this if your col name is not 'text'.
'ori_emo_col': 'posneg'
\tYou need to modify this if your col name is not 'posneg'.
'default_emo_col': 'default_emotion'
\tYou need to modify this if your col name is not 'default_emotion'.
'normalized_emo_col': 'gpt_emotion'
\tYou need to modify this if your col name is not 'gpt_emotion'.
'truncation': True,
'sent_emo_col': 'klue_emo'
\tYou can leave it as it is, change it, as you wish.
'sent_spec_emo_col': 'klue_specific_emo'
\tYou can leave it as it is, change it, as you wish.
'max_length': 512
        """)

    def chunk_text(self, text, max_length=512):
        tokens = self.tokenizer.tokenize(text)
        chunk_size = max_length - 2
        for i in range(0, len(tokens), chunk_size):
            yield ' '.join(tokens[i:i + chunk_size])

    def get_emotion(self,
                    original_emotion,
                    default_specific_emotion,
                    gpt_specific_emotion,
                    sentence
                    ):
        chunks = list(self.chunk_text(sentence))
        sum_prob = None
        num_chunks = 0

        for chunk in chunks:
            if gpt_specific_emotion != '-':
                inputs = self.tokenizer(gpt_specific_emotion + self.sep_token + chunk, return_tensors='pt',
                                        padding=True, truncation=self.truncation)
            else:
                inputs = self.tokenizer(default_specific_emotion + self.sep_token + chunk, return_tensors='pt',
                                        padding=True, truncation=self.truncation)

            with torch.no_grad():
                logits = self.model(**inputs).logits
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

        if sentiment_idx in self.neg_tag:
            emotion = 'negative'
            specific_emotion = self.emotion_label[sentiment_idx]
        elif sentiment_idx in self.pos_tag:
            emotion = 'positive'
            specific_emotion = self.emotion_label[sentiment_idx]
        else:
            emotion = 'neutral'
            specific_emotion = self.emotion_label[sentiment_idx]

        if original_emotion.strip() == emotion.strip():
            return emotion, specific_emotion
        else:
            emotion_counts = {'positive': 0, 'negative': 0}

            for chunk in chunks:
                if gpt_specific_emotion != '-':
                    inputs = self.sub_tokenizer(gpt_specific_emotion + self.sep_token + chunk, return_tensors='pt',
                                                padding=True, truncation=self.truncation)
                else:
                    inputs = self.sub_tokenizer(default_specific_emotion + self.sep_token + chunk, return_tensors='pt',
                                                padding=True, truncation=self.truncation)

                with torch.no_grad():
                    logits = self.sub_model(**inputs).logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                emotion_pred = probabilities.argmax().item()  ##### majority votes
                if emotion_pred == 0:
                    emotion_counts['negative'] += 1
                elif emotion_pred == 1:
                    emotion_counts['positive'] += 1

            if emotion_counts['negative'] > emotion_counts['positive']:
                emotion = 'negative'
            elif emotion_counts['negative'] < emotion_counts['positive']:
                emotion = 'positive'
            else:
                emotion = 'neutral'
            specific_emotion = gpt_specific_emotion
            if original_emotion.strip() == emotion.strip():
                return emotion, specific_emotion
            else:  ############ 여기에 아예 catboost를 써볼까....??
                sum_prob = None
                num_chunks = 0
                for chunk in chunks:
                    if gpt_specific_emotion != '-':
                        inputs = self.sub_tokenizer(gpt_specific_emotion + self.sep_token + chunk, return_tensors='pt',
                                                    padding=True, truncation=self.truncation)
                    else:
                        inputs = self.sub_tokenizer(default_specific_emotion + self.sep_token + chunk,
                                                    return_tensors='pt', padding=True, truncation=self.truncation)

                    with torch.no_grad():
                        logits = self.sub_model(**inputs).logits
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

    def match_rate_calc(self, df):
        mat = 0
        unmat = 0
        for row_num in tqdm(range(df.shape[0])):
            sentence = df.iloc[row_num][self.text_col]
            original_emotion = df.iloc[row_num][self.ori_emo_col]
            default_spec_emo = df.iloc[row_num][self.default_emo_col]
            gpt_spec_emo = df.iloc[row_num][self.normalized_emo_col]
            pred_emotion, specified_emotion = self.get_emotion(original_emotion,
                                                               default_spec_emo,
                                                               gpt_spec_emo,
                                                               sentence
                                                               )
            if pred_emotion == original_emotion:
                mat += 1
            else:
                unmat += 1
        match_rate = mat / (mat + unmat) * 100

        return match_rate

    def get_word_embeddings(self, sentence, max_length=512):
        inputs = self.word_senti_tokenizer(sentence, return_tensors="pt", max_length=max_length,
                                           truncation=self.truncation, padding='max_length',
                                           return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping')
        with torch.no_grad():
            outputs = self.word_senti_model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings, offset_mapping

    def get_sentence_embedding(self, sentence, max_length=512):
        inputs = self.word_senti_tokenizer(sentence, return_tensors="pt", max_length=max_length,
                                           truncation=self.truncation,
                                           padding="max_length")
        with torch.no_grad():
            outputs = self.word_senti_model(**inputs)
        # Average the token embeddings to get the sentence embedding
        sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding

    def get_emotion_embedding(self, emotion_label, emotion_spec_label):
        inputs = self.word_senti_tokenizer(emotion_label + ' ' + emotion_spec_label, return_tensors='pt', padding=True,
                                           truncation=self.truncation)
        with torch.no_grad():
            outputs = self.word_senti_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def calculate_vector_differences(self, sentence, emotion_label):
        word_embeddings, offset_mapping = self.get_word_embeddings(sentence)
        sentence_embedding = self.get_sentence_embedding(sentence)
        emotion_embedding = self.get_emotion_embedding(emotion_label)
        alignment_threshold = 0.7  # 0.75 ~ 0.7
        combined_embedding = (sentence_embedding - emotion_embedding) / 2

        dissimilarities = []

        for word_embedding in word_embeddings.squeeze():
            sim = F.cosine_similarity(combined_embedding, word_embedding.unsqueeze(0))
            if sim > alignment_threshold:
                adjust_sim = sim.item()
            else:
                adjust_sim = 1 - sim.item()

            dissimilarities.append(adjust_sim)

        return dissimilarities

    def normalize_sentiment_scores(self, dissimilarities):
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

    def assign_word_sentiment_scores(self, sentence, normalized_scores):
        encoded = self.word_senti_tokenizer.encode_plus(sentence, return_tensors='pt', truncation=self.truncation,
                                                        padding='max_length', max_length=512)
        tokenized_words = self.word_senti_tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        word_scores = {}
        current_word = ""
        current_score = 0
        num_tokens = 0

        for token, score in zip(tokenized_words, normalized_scores):
            if token.startswith("##"):
                current_word += token.lstrip("##")
                current_score += score
                num_tokens += 1
            else:
                if current_word and not current_word.startswith('['):
                    average_score = current_score / num_tokens if num_tokens > 0 else current_score
                    word_scores[current_word] = average_score
                current_word = token
                current_score = score
                num_tokens = 1

        # Add the last word
        if current_word and not current_word.startswith('['):
            average_score = current_score / num_tokens if num_tokens > 0 else current_score
            word_scores[current_word] = average_score

        return word_scores

    def word_segmentator(self, sentence):
        pattern = '[^ㄱ-ㅣ가-힣a-zA-Z0-9+]'
        sentence = re.sub(pattern, ' ', sentence)
        dissimilarities = self.calculate_vector_differences(sentence)
        norm_senti_score = self.normalize_sentiment_scores(dissimilarities)
        word_sentiment_scores = self.assign_word_sentiment_scores(sentence, norm_senti_score)
        return word_sentiment_scores

    def noun_emotions(self, sentence, noun_list):
        word_emo_list = self.word_segmentator(sentence)

        pos = defaultdict(list)
        neut = defaultdict(list)
        neg = defaultdict(list)
        for noun in list(set(noun_list)):
            for word, score in word_emo_list.items():
                if noun in word:
                    if score > 0.3:  # when positive
                        pos[noun].append(score)
                    elif score < -0.3:
                        neg[noun].append(score)
                    else:
                        neut[noun].append(score)

        positive_noun_score = {}
        neutral_noun_score = {}
        negative_noun_score = {}

        for word, score_list in pos:
            avg_score = sum(score_list) / len(score_list)
            positive_noun_score[word] = avg_score
        for word, score_list in neut:
            avg_score = sum(score_list) / len(score_list)
            neutral_noun_score[word] = avg_score
        for word, score_list in neg:
            avg_score = sum(score_list) / len(score_list)
            negative_noun_score[word] = avg_score

        return positive_noun_score, neutral_noun_score, negative_noun_score
