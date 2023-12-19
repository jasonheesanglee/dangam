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
    VERSION = '0.0.137'
    CREATED_BY = 'jasonheesanglee\thttps://github.com/jasonheesanglee'

    def cfg_info(self) -> None:
        print(f"""
'ko_model_name' - The model that will run through the first loop of the Korean sentence segmentation.
'ko_sub_model_name' - The model that will run through the second loop of the Korean sentence segmentation.
'ko_word_senti_model_name' - The model that will through the loop of the Korean word segmentation.
'en_model_name' - The model that will run through the first loop of the English sentence segmentation.
'en_sub_model_name' - The model that will run through the second loop of the English sentence segmentation.
'en_word_senti_model_name' - The model that will through the loop of the English word segmentation.
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
'ko_model_name': 'hun3359/klue-bert-base-sentiment',
\tBetter if you stick to this.
This is one of the only options that segments Korean sentences into 60 sentiment labels.
'ko_sub_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment'
\tAny model that supports Positive/Negative segmentation will work.
'ko_word_senti_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment'
\tAny model that supports Positive/Negative segmentation will work.
'en_model_name': 'SamLowe/roberta-base-go_emotions',
\tBetter if you stick to this.
This is one of the only options that segments English sentences into 28 sentiment labels.
'en_sub_model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
\tAny model that supports Positive/Negative segmentation will work. (May need some modification)
'en_word_senti_model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
\tAny model that supports Positive/Negative segmentation will work. (May need some modification)
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
                'ko_model_name': 'hun3359/klue-bert-base-sentiment',
                'ko_sub_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment',
                'ko_word_senti_model_name': 'WhitePeak/bert-base-cased-Korean-sentiment',
                'en_model_name': 'SamLowe/roberta-base-go_emotions',
                'en_sub_model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'en_word_senti_model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'truncation': True,
                'padding': True,
                'max_length': 512,
                'alignment_threshold': 0.4,
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

        self.ko_model_name = config['ko_model_name']
        self.ko_sub_model_name = config['ko_sub_model_name']
        self.ko_word_senti_model_name = config['ko_word_senti_model_name']
        self.en_model_name = config['en_model_name']
        self.en_sub_model_name = config['en_sub_model_name']
        self.en_word_senti_model_name = config['en_word_senti_model_name']
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
            'ko_model_name': self.ko_model_name,
            'ko_sub_model_name': self.ko_sub_model_name,
            'ko_word_senti_model_name': self.ko_word_senti_model_name,
            'en_model_name' : self.en_model_name,
            'en_sub_model_name' : self.en_sub_model_name,
            'en_word_senti_model_name' : self.en_word_senti_model_name,
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