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
    The DanGam class is designed for emotion analysis in textual data.
    It leverages advanced NLP models
    to segment general and specific emotions at both sentence and word levels.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for processing input text.
        model (AutoModelForSequenceClassification): Primary model for emotion classification.
        sub_tokenizer (AutoTokenizer): Secondary tokenizer for enhanced emotion analysis.
        sub_model (AutoModelForSequenceClassification): Secondary model for detailed emotion classification.
        word_senti_model (AutoModel): Model for word-level sentiment analysis.
        word_senti_tokenizer (AutoTokenizer): Tokenizer for word-level sentiment analysis.
        sep_token (str): Separator token used in text processing.
        emotion_label (dict): Dictionary mapping emotion indices to labels.
        truncation (bool): Flag to enable or disable truncation in text processing.
        pos_tag, neut_tag, neg_tag (list): Lists of tags for positive, neutral, and negative emotions.
        ori_emo_col, text_col, default_emo_col, normalized_emo_col, sent_emo_col, sent_spec_emo_col (str): Column names for various data attributes.

    Methods:
        cfg_info(): Displays configuration information.
        check_default(): Checks and prints default configuration settings.
        chunk_text(text, max_length): Breaks text into chunks for processing.
        get_emotion(...): Determines the overall emotion of a sentence.
        match_rate_calc(df): Calculates the match rate of predicted emotions against original emotions.
        get_word_embeddings(sentence, max_length): Retrieves word embeddings from a sentence.
        get_sentence_embedding(sentence, max_length): Computes the sentence embedding.
        get_emotion_embedding(emotion): Computes the embedding for a general emotion.
        get_specific_emotion_embedding(specific_emotion): Computes the embedding for a specific emotion.
        calculate_vector_differences(sentence, emotion, specific_emotion): Calculates the emotional dissimilarities in a sentence.
        normalize_sentiment_scores(dissimilarities): Normalizes sentiment scores to a standard range.
        assign_word_sentiment_scores(sentence, normalized_scores): Assigns sentiment scores to each word in a sentence.
        word_segmentator(sentence, emotion, specific_emotion): Segments and assigns emotions to words in a sentence.
        noun_emotions(sentence, noun_list): Analyzes and categorizes emotions associated with specific nouns.

    Usage:
        - Initialize the class.
        - Use its methods for emotion segmentation and analysis in text.
    """
    VERSION = '0.0.10'
    CREATED_BY = 'jasonheesanglee\thttps://github.com/jasonheesanglee'

    def __init__(self, cfg=None):
        """
        Initialize DanGam with default or custom configuration.

        Args:
            cfg (dict, optional): Custom configuration settings.
        """
        print('''
CAUTION
This logic performs the best with the models that are pretrained with
AI HUB Dataset https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86
or any Dataset that has 60 sentiment tags listed as described in https://huggingface.co/hun3359/klue-bert-base-sentiment/blob/main/config.json\n
You can also modify configuration by calling DanGamConfig()
        '''
              )
        if cfg is not None:
            self.cfg = DotDict(DanGamConfig(cfg))
        else:
            self.cfg = DotDict(DanGamConfig())
        self.initialize_models()
        self.sep_token = ' >>>> '
        self.emotion_label = emotion_labels
        self.emotion_label_ko = emotion_labels
        self.pos_tag = pos_tag
        self.neut_tag = neut_tag
        self.neg_tag = neg_tag

    def cfg_info(self):
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
        self.truncation = self.cfg.truncation
        self.ori_emo_col = self.cfg.original_emotion_column
        self.text_col = self.cfg.text_col
        self.default_emo_col = self.cfg.default_emotion_col
        self.normalized_emo_col = self.cfg.normalized_emotion_col
        self.sent_emo_col = self.cfg.sentence_emotion_col
        self.sent_spec_emo_col = self.cfg.sentence_specific_emotion_col
        self.max_length = self.cfg.max_length

    def update_config(self, new_config):
        """
        Update the configuration of DanGam and reinitialize components as necessary.

        Args:
            new_cfg (dict): A dictionary containing the new configuration settings.
        """
        self.cfg = DanGamConfig(new_config)
        self.initialize_models()

    def check_default(self):
        print("""
'model_name': 'hun3359/klue-bert-base-sentiment',
\tBetter if you stick to this. This is one of the only options that segments the sentences into 60 sentiment labels.\n
'sub_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment'\n
'text_col': 'text'
\tYou need to modify this if your col name is not 'text'.\n
'original_emotion_column': 'posneg'
\tYou need to modify this if your col name is not 'posneg'.\n
'default_emo_col': 'default_emotion'
\tYou need to modify this if your col name is not 'default_emotion'.\n
'normalized_emo_col': 'gpt_emotion'
\tYou need to modify this if your col name is not 'gpt_emotion'.\n
'truncation': True,\n
'sent_emo_col': 'klue_emo'\n
\tYou can leave it as it is, change it, as you wish.\n
'sent_spec_emo_col': 'klue_specific_emo'
\tYou can leave it as it is, change it, as you wish.\n
'max_length': 512
        """
              )

    def check_config(self):
        """
        Returns the current configuration of DanGam as a dictionary.
        """
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
                    original_emotion: str,
                    default_specific_emotion: str,
                    gpt_specific_emotion: str,
                    sentence
                    ):
        """
        Determines the overall emotion of a given sentence by analyzing it in chunks.
        Considers both the general and specific emotions to enhance accuracy.

        Args:
            original_emotion, default_specific_emotion, gpt_specific_emotion, sentence: Parameters defining the sentence and its emotions.

        Returns:
            tuple: A tuple containing the general emotion and the specific emotion of the sentence.
        """
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
        """
        Retrieves word embeddings for a given sentence using the specified tokenizer and model.

        Args:
            sentence (str): The sentence for which to get word embeddings.
            max_length (int): Maximum length for tokenization.

        Returns:
            tuple: A tuple containing word embeddings and offset mappings.
        """
        inputs = self.word_senti_tokenizer(sentence, return_tensors="pt", max_length=max_length,
                                           truncation=self.truncation, padding='max_length',
                                           return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping')
        with torch.no_grad():
            outputs = self.word_senti_model(**inputs)
        embeddings = outputs.last_hidden_state
        return embeddings, offset_mapping

    def get_sentence_embedding(self, sentence, max_length=512):
        """
        Computes the embedding of a sentence by averaging the embeddings of its constituent tokens.

        Args:
            sentence (str): The sentence to compute embedding for.
            max_length (int): Maximum length for tokenization.

        Returns:
            torch.Tensor: The computed sentence embedding.
        """
        inputs = self.word_senti_tokenizer(sentence, return_tensors="pt", max_length=max_length,
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
        inputs = self.word_senti_tokenizer(emotion, return_tensors='pt', padding=True,
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
        inputs = self.word_senti_tokenizer(specific_emotion, return_tensors='pt', padding=True,
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

        alignment_threshold = 0.7  # 0.75 ~ 0.7
        emotion_threshold = 0.3  # 0.75 ~ 0.7

        general_weight = 0.5 if general_alignment.item() > emotion_threshold else 0.75
        specific_weight = 0.1 if specific_alignment.item() > emotion_threshold else 0.23
        sentence_weight = 1 - (general_weight + specific_weight)

        combined_embedding = (emotion_embedding * general_weight -
                              sentence_embedding * sentence_weight +
                              specific_emotion_embedding * specific_weight) / 2

        # combined_embedding = (sentence_embedding - emotion_embedding) / 2

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

    def word_segmentator(self, sentence, emotion, specific_emotion):
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

    def noun_emotions(self, sentence, noun_list):
        """
        Analyzes emotions associated with specific nouns within a sentence.

        Args:
            sentence (str): The sentence containing the nouns for emotion analysis.
            noun_list (list): A list of nouns to analyze within the sentence.

        Returns:
            dict: A dictionary categorizing nouns into positive, neutral, and negative based on their associated emotions.
        """
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
