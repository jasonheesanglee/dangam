class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DanGamConfig:
    """
    Configuration class for DanGam. It handles the setup of model parameters and preferences.

    Methods:
        cfg_info(): Prints the detailed information about configuration options.
        check_default(): Prints the default configuration settings.
    """
    VERSION = '0.0.126'
    CREATED_BY = 'jasonheesanglee\thttps://github.com/jasonheesanglee'

    def cfg_info(self) -> None:
        print(f"""
'model_name' - The model that will run through the first loop of the sentence segmentation.
'sub_model_name' - The model that will run through the second loop of the sentence segmentation.
'word_senti_model_name' - The model that will through the loop of the word segmentation.
'text_col' - The name of the column that you want to segment the emotion.
'default_emotion_column' - Pre-labeled emotion by user.
'original_emotion_column' - Pre-segmented emotions by user.
\tPerforms the best if this section is segmented into 'positive', 'negative', 'neutral'.
\tUsed for accuracy evaluation.
'normalized_emotion_column' - Normalized pre-labeled emotion.
\tPerforms the best if this section is in English.
\tDirectly used from the second loop, since it will only segment positive, negative, neutral.
\tNot into 60 different emotions.
'sentence_emotion_column' - The column name of sentence emotion (pos/neg/neut) you want this module to set.
'sentence_specific_emotion_column' - The column name of sentence emotion (pos/neg/neut) you want this module to set.
'truncation' - Turning on and off Truncation throughout the module.
'max_length' - Max length for chunk_text
'emotion_threshold' - Threshold for emotion and specific emotion embeddings are adjusted accordingly to refine the combined embedding, ensuring a more nuanced sentiment analysis. 
'alignment_threshold' - Threshold for the cosine similarity between the combined sentence-emotion embedding and each individual word embedding.
'emotion_weight_reach_threshold' - The weight to be multiplied on emotion embedding when similarity exceeds threshold.
'emotion_weight_not_reach_threshold' - The weight to be multiplied on emotion embedding when similarity doesn't exceed threshold.
'specific_weight_reach_threshold' - The weight to be multiplied on specific emotion embedding when similarity exceeds threshold.
'specific_weight_not_reach_threshold' - The weight to be multiplied on specific emotion embedding when similarity doesn't exceed threshold.
'noun_threshold' - Threshold for deciding the emotion_segment of a word.
        """)

    def check_default(self):
        return """
'model_name': 'hun3359/klue-bert-base-sentiment',
\tBetter if you stick to this.
This is one of the only options that segments Korean sentences into 60 sentiment labels.
'sub_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment'
\tAny model that supports Positive/Negative segmentation will work.
'word_senti_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment'
\tAny model that supports Positive/Negative segmentation will work.
'text_col': 'text'
\tYou need to modify this if your col name is not 'text'.
'original_emotion_column': 'posneg'
\tYou need to modify this if your col name is not 'posneg'.
'default_emotion_column': 'default_emotion'
\tYou need to modify this if your col name is not 'default_emotion'.
'normalized_emotion_column': 'gpt_emotion'
\tYou need to modify this if your col name is not 'gpt_emotion'.
'sentence_emotion_column': 'klue_emo'
\tYou can leave it as it is, change it, as you wish.
'sentence_specific_emotion_column': 'klue_specific_emo'
\tYou can leave it as it is, change it, as you wish.
'truncation': True
'padding': True
'max_length': 512

### BELOW NEEDS CAUTION WHEN MODIFYING ###

'emotion_threshold' : 0.3
'alignment_threshold': 0.7
'emotion_weight_reach_threshold':0.5
'emotion_weight_not_reach_threshold': 0.75
'specific_weight_reach_threshold': 0.1
'specific_weight_not_reach_threshold': 0.23
'noun_threshold' : 0.3
"""


    def __init__(self, cfg=None):
        """
        Initializes the DanGamConfig with default or user-provided settings.

        Args:
            cfg (dict, optional): A dictionary containing configuration settings.
                                  If provided, it overrides the default settings.
        """
        config = {
            'model_name': 'hun3359/klue-bert-base-sentiment',
            'sub_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment',
            'word_senti_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment',
            'text_col': 'text',
            'original_emotion_column': 'posneg',
            'default_emotion_column': 'default_emotion',
            'normalized_emotion_column': 'gpt_emotion',
            'sentence_emotion_column': 'klue_emo',
            'sentence_specific_emotion_column': 'klue_specific_emo',
            'truncation': True,
            'padding': True,
            'max_length': 512,
            'alignment_threshold': 0.7,
            'emotion_threshold' : 0.3,
            'emotion_weight_reach_threshold':0.5,
            'emotion_weight_not_reach_threshold': 0.75,
            'specific_weight_reach_threshold': 0.1,
            'specific_weight_not_reach_threshold': 0.23,
            'noun_threshold' : 0.3,
        }

        if cfg is not None:
            if isinstance(cfg, dict):
                config.update(cfg)

        self.model_name = config['model_name']
        self.sub_model_name = config['sub_model_name']
        self.word_senti_model_name = config['word_senti_model_name']
        self.text_col = config['text_col']
        self.original_emotion_column = config['original_emotion_column']
        self.normalized_emotion_column = config['normalized_emotion_column']
        self.default_emotion_column = config['default_emotion_column']
        self.sentence_emotion_column = config['sentence_emotion_column']
        self.sentence_specific_emotion_column = config['sentence_specific_emotion_column']
        self.truncation = config['truncation']
        self.padding = config['padding']
        self.max_length = config['max_length']
        self.alignment_threshold = config['alignment_threshold']
        self.emotion_threshold = config['emotion_threshold']
        self.emotion_weight_reach_threshold = config['emotion_weight_reach_threshold']
        self.emotion_weight_not_reach_threshold = config['emotion_weight_not_reach_threshold']
        self.specific_weight_reach_threshold = config['specific_weight_reach_threshold']
        self.specific_weight_not_reach_threshold = config['specific_weight_not_reach_threshold']
        self.noun_threshold = config['noun_threshold']

    def get_config(self):
        """
        Returns the current configuration as a dictionary.
        """
        return {
            'model_name': self.model_name,
            'sub_model_name': self.sub_model_name,
            'word_senti_model': self.word_senti_model_name,
            'original_emotion_column' : self.original_emotion_column,
            'text_col' : self.text_col,
            'default_emotion_col' : self.default_emotion_column,
            'normalized_emotion_col' : self.normalized_emotion_column,
            'sentence_emotion_col' : self.sentence_emotion_column,
            'sentence_specific_emotion_col' : self.sentence_specific_emotion_column,
            'truncation' : self.truncation,
            'padding': self.padding,
            'max_length': self.max_length,
            'alignment_threshold': self.alignment_threshold,
            'emotion_threshold': self.emotion_threshold,
            'emotion_weight_reach_threshold': self.emotion_weight_reach_threshold,
            'emotion_weight_not_reach_threshold': self.emotion_weight_not_reach_threshold,
            'specific_weight_reach_threshold': self.specific_weight_reach_threshold,
            'specific_weight_not_reach_threshold': self.specific_weight_not_reach_threshold,
            'noun_threshold': self.noun_threshold,

        }