import re

import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from collections import defaultdict
from .config import DanGamConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel


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
    40: "bewildered", 41: "beleaguered", 42: "self conscious", 43: "lonely",
    44: "inferiority", 45: "guilty", 46: "ashamed", 47: "abominable",
    48: "pathetic", 49: "baffled", 50: "joyful", 51: "grateful",
    52: "trusting", 53: "comfortable", 54: "satisfying", 55: "thrilled",
    56: "relaxed", 57: "relieved", 58: "excited", 59: "confident"
}

emotion_labels_ko = {
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

neg_tag = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
           10, 11, 12, 13, 14, 15, 16, 18, 19,
           20, 21, 22, 23, 24, 25, 26, 27,
           30, 32, 34, 35, 36, 37, 38, 39,
           40, 41, 43, 44, 45, 46, 47, 48, 49]
pos_tag = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
neut_tag = [17, 28, 29, 31, 33, 42]


# noinspection PyTypeChecker
class DanGam:
    """
    The DanGam class is designed for advanced emotion analysis in textual data. It utilizes
    sophisticated NLP models to accurately identify and categorize emotions at both sentence
    and word levels, providing nuanced insights into the emotional tone of text.

    Attributes:
        cfg (DotDict): Configuration settings for the DanGam instance.
        model (AutoModelForSequenceClassification): Primary model for general emotion classification.
        sub_model (AutoModelForSequenceClassification): Secondary model for more detailed emotion classification.
        word_senti_model (AutoModel): Model dedicated to word-level sentiment analysis.
        original_emotion_column, text_col, default_emotion_column, normalized_emotion_column, sentence_emotion_column, sentence_specific_emotion_column (str): Column names for various data attributes, configurable as per user needs.

    Methods:
        cfg_info(): Displays the current configuration settings of the DanGam instance.
        check_default(): Outputs the default configuration values for reference.
        check_config(): Returns the current configuration of DanGam as a dictionary.
        chunk_text(text, max_length): Splits a given text into manageable chunks for model processing.
        get_emotion(...): Analyzes and returns the overall emotion of a sentence based on model predictions.
        match_rate_calc(df): Calculates and returns the accuracy of emotion predictions in a DataFrame.
        normalize_sentiment_scores(dissimilarities): Normalizes sentiment dissimilarity scores.
        word_emotions(sentence, emotion_label, specific_emotion_label): Segments words in a sentence and assigns them emotions.
        noun_emotions(sentence, noun_list): Analyzes and categorizes emotions associated with specific nouns in a sentence.

    Usage:
        - Initialize the class with default or custom configuration.
        - Use its methods to perform detailed emotion segmentation and analysis in textual content.
    """
    VERSION = '0.0.129'
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
        self.emotion_label = emotion_labels
        self.emotion_label_ko = emotion_labels
        self.pos_tag = pos_tag
        self.neut_tag = neut_tag
        self.neg_tag = neg_tag

    def config_info(self):
        """
        Prints the current configuration information of the DanGam.
        Includes details about the models used, text and emotion column names, and other settings.
        """
        print(f"""
'model_name' - The model that will run through the first loop of the sentence segmentation.\n
'sub_model_name' - The model that will run through the second loop of the sentence segmentation.\n
'word_senti_model_name' - The model that will run through the second loop of the sentence segmentation.\n
'text_col' - The name of the column that you want to segment the emotion.\n
'default_emo_col' - Pre-labeled emotion by user.\n
'ori_emo_col' - Pre-segmented emotions by user.
\tPerforms the best if this section is segmented into 'positive', 'negative', 'neutral'.
\tUsed for accuracy evaluation.\n
'normalized_emo_col' - Normalized pre-labeled emotion.
\tPerforms the best if this section is in English.
\tDirectly used from the second loop, since it will only segment positive, negative, neutral.
\tNot into 60 different emotions.\n
'truncation' - Bool : Turning on and off Truncation throughout the module.\n 
'sent_emo_col' - The column name of sentence emotion (pos/neg/neut) you want this module to set.\n
'sent_spec_emo_col' - The column name of sentence emotion (pos/neg/neut) you want this module to set.\n
'max_length' - Max length for chunk_text.
        """)

    def initialize_models(self):
        """
        Initialize models based on the current configuration.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_name)
        self.sub_tokenizer = AutoTokenizer.from_pretrained(self.cfg.sub_model_name)
        self.sub_model = AutoModelForSequenceClassification.from_pretrained(self.cfg.sub_model_name)
        self.word_senti_model = AutoModel.from_pretrained(self.cfg.word_senti_model_name)
        self.word_senti_tokenizer = AutoTokenizer.from_pretrained(self.cfg.word_senti_model_name)

        self.text_col = self.cfg.text_col
        self.ori_emo_col = self.cfg.original_emotion_column
        self.normalized_emo_col = self.cfg.normalized_emotion_column

        self.default_emo_col = self.cfg.default_emotion_column
        self.sent_emo_col = self.cfg.sentence_emotion_column
        self.sent_spec_emo_col = self.cfg.sentence_specific_emotion_column
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

    def chunk_text(self, text: str, max_length=512):
        """
        Splits a given text into chunks for processing.
        Ensures that each chunk is within the specified maximum length.

        Args:
            text (str): The text to be chunked.
            max_length (int): Maximum length of each text chunk.

        Yields:
            str: Chunks of the original text.
        """
        tokens = self.tokenizer.tokenize(text)
        chunk_size = max_length - 2
        for i in range(0, len(tokens), chunk_size):
            yield ' '.join(tokens[i:i + chunk_size])

    def get_emotion(self,
                    sentence: str,
                    original_emotion: str,
                    default_specific_emotion: str,
                    normalized_emotion: str
                    ):
        """
        Determines the overall emotion of a given sentence by analyzing it in chunks.
        Considers both the general and specific emotions to enhance accuracy.

        Args:
            sentence, original_emotion, default_specific_emotion, normalized_emotion : Parameters defining the sentence and its emotions.

        Returns:
            tuple: A tuple containing the general emotion and the specific emotion of the sentence.
        """
        chunks = list(self.chunk_text(sentence))
        sum_prob = None
        num_chunks = 0

        for chunk in chunks:
            if normalized_emotion != '-':
                inputs = self.tokenizer(normalized_emotion + self.sep_token + chunk, return_tensors='pt',
                                        padding=self.padding, truncation=self.truncation)
            else:
                inputs = self.tokenizer(default_specific_emotion + self.sep_token + chunk, return_tensors='pt',
                                        padding=self.padding, truncation=self.truncation)

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
                if normalized_emotion != '-':
                    inputs = self.sub_tokenizer(normalized_emotion + self.sep_token + chunk, return_tensors='pt',
                                                padding=self.padding, truncation=self.truncation)
                else:
                    inputs = self.sub_tokenizer(default_specific_emotion + self.sep_token + chunk, return_tensors='pt',
                                                padding=self.padding, truncation=self.truncation)

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
            specific_emotion = normalized_emotion
            if original_emotion.strip() == emotion.strip():
                return emotion, specific_emotion
            else:  ############ 여기에 아예 catboost를 써볼까....??
                sum_prob = None
                num_chunks = 0
                for chunk in chunks:
                    if normalized_emotion != '-':
                        inputs = self.sub_tokenizer(normalized_emotion + self.sep_token + chunk, return_tensors='pt',
                                                    padding=self.padding, truncation=self.truncation)
                    else:
                        inputs = self.sub_tokenizer(default_specific_emotion + self.sep_token + chunk,
                                                    return_tensors='pt', padding=self.padding, truncation=self.truncation)

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
        """
        Calculates the accuracy of emotion predictions in a dataframe by comparing predicted emotions
        with their original annotations.

        Args:
            df (DataFrame): A pandas DataFrame containing text data along with original and predicted emotion annotations.

        Returns:
            float: The match rate percentage indicating the accuracy of emotion predictions.
        """
        mat = 0
        unmat = 0
        for row_num in tqdm(range(df.shape[0])):
            sentence = df.iloc[row_num][self.text_col]
            original_emotion = df.iloc[row_num][self.ori_emo_col]
            default_spec_emo = df.iloc[row_num][self.default_emo_col]
            norm_spec_emo = df.iloc[row_num][self.normalized_emo_col]
            pred_emotion, specified_emotion = self.get_emotion(sentence,
                                                               original_emotion,
                                                               default_spec_emo,
                                                               norm_spec_emo
                                                               )
            if pred_emotion == original_emotion:
                mat += 1
            else:
                unmat += 1
        match_rate = mat / (mat + unmat) * 100

        return match_rate

    def get_word_embeddings(self, sentence):
        """
        Retrieves word embeddings for a given sentence using the specified tokenizer and model.

        Args:
            sentence (str): The sentence for which to get word embeddings.
            max_length (int): Maximum length for tokenization.

        Returns:
            tuple: A tuple containing word embeddings and offset mappings.
        """
        inputs = self.word_senti_tokenizer(sentence, return_tensors="pt", max_length=self.max_length,
                                           truncation=self.truncation, padding='max_length',
                                           return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping')
        with torch.no_grad():
            outputs = self.word_senti_model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings, offset_mapping

    def get_sentence_embedding(self, sentence):
        """
        Computes the embedding of a sentence by averaging the embeddings of its constituent tokens.

        Args:
            sentence (str): The sentence to compute embedding for.
            max_length (int): Maximum length for tokenization.

        Returns:
            torch.Tensor: The computed sentence embedding.
        """
        inputs = self.word_senti_tokenizer(sentence, return_tensors="pt", max_length=self.max_length,
                                           truncation=self.truncation,
                                           padding="max_length")
        with torch.no_grad():
            outputs = self.word_senti_model(**inputs)
        # Average the token embeddings to get the sentence embedding
        sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding

    def get_emotion_embedding(self, emotion):
        """
        Computes the embedding for a general emotion label.

        Args:
            emotion (str): The emotion label for which to compute the embedding.

        Returns:
            torch.Tensor: The embedding of the specified emotion.
        """
        inputs = self.word_senti_tokenizer(emotion, return_tensors='pt', padding=self.padding,
                                           truncation=self.truncation)
        with torch.no_grad():
            outputs = self.word_senti_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def get_specific_emotion_embedding(self, specific_emotion):
        """
        Computes the embedding for a specific emotion label.

        Args:
            specific_emotion (str): The specific emotion label for which to compute the embedding.

        Returns:
            torch.Tensor: The embedding of the specified specific emotion.
        """
        inputs = self.word_senti_tokenizer(specific_emotion, return_tensors='pt', padding=self.padding,
                                           truncation=self.truncation)
        with torch.no_grad():
            outputs = self.word_senti_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def calculate_vector_differences(self, sentence, emotion, specific_emotion):
        """
        Calculates vector differences (dissimilarities) between words in a sentence and a combined emotion embedding.

        Args:
            sentence (str): The sentence for analysis.
            emotion (str): General emotion label for the sentence.
            specific_emotion (str): Specific emotion label for the sentence.

        Returns:
            list: A list of dissimilarity scores for each word in the sentence.
        """
        word_embeddings, offset_mapping = self.get_word_embeddings(sentence)
        sentence_embedding = self.get_sentence_embedding(sentence)
        emotion_embedding = self.get_emotion_embedding(emotion)
        specific_emotion_embedding = self.get_emotion_embedding(specific_emotion)

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

    def assign_word_sentiment_scores(self, sentence, normalized_scores):
        """
        Assigns sentiment scores to each word in a sentence based on normalized dissimilarities.

        Args:
            sentence (str): The sentence to assign word sentiment scores.
            normalized_scores (list): Normalized sentiment scores for each token in the sentence.

        Returns:
            dict: A dictionary mapping each word in the sentence to its sentiment score.
        """
        encoded = self.word_senti_tokenizer.encode_plus(sentence, return_tensors='pt', truncation=self.truncation,
                                                        padding='max_length', max_length=512)
        tokenized_words = self.word_senti_tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        word_scores = []
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
                    word_scores.append({current_word:average_score})
                current_word = token
                current_score = score
                num_tokens = 1

        # Add the last word
        if current_word and not current_word.startswith('['):
            average_score = current_score / num_tokens if num_tokens > 0 else current_score
            word_scores.append({current_word:average_score})

        return word_scores

    def word_emotions(self, sentence: str, emotion: str, specific_emotion: str):
        """
        Segments a sentence and assigns emotions to each word based on the overall sentence emotion and specific emotion.

        Args:
            sentence (str): The sentence for segmentation.
            emotion (str): The general emotion of the sentence.
            specific_emotion (str): The specific emotion of the sentence.

        Returns:
            dict: A dictionary mapping each word in the sentence to its assigned emotion.
        """
        pattern = '[^ㄱ-ㅣ가-힣a-zA-Z0-9+]'
        sentence = re.sub(pattern, ' ', sentence)
        dissimilarities = self.calculate_vector_differences(sentence, emotion, specific_emotion)
        norm_senti_score = self.normalize_sentiment_scores(dissimilarities)
        word_sentiment_scores = self.assign_word_sentiment_scores(sentence, norm_senti_score)
        return word_sentiment_scores

    def noun_emotions(self, sentence: str, emotion: str, specific_emotion: str, noun_list: list, count: bool = False):
        """
        Analyzes emotions associated with specific nouns within a sentence.

        Args:
            sentence (str): The sentence containing the nouns for emotion analysis.
            noun_list (list): A list of nouns to analyze within the sentence.
            count (bool) : True or False for switching on off counting the number of nouns in each segment.
        Returns:
            dict: A dictionary categorizing nouns into positive, neutral, and negative based on their associated emotions.
        """
        word_emo_list = self.word_emotions(sentence, emotion, specific_emotion)

        pos = defaultdict(list)
        neut = defaultdict(list)
        neg = defaultdict(list)

        pos_count = defaultdict(int)
        neut_count = defaultdict(int)
        neg_count = defaultdict(int)

        for noun in list(set(noun_list)):
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
