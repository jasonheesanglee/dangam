class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DanGamConfig:
    """
    Configuration class for DanGam. It handles the setup of model parameters and preferences.

    Attributes:
        VERSION (str): Version of the DanGam.
        CREATED_BY (str): Creator information.

    Methods:
        cfg_info(): Prints the detailed information about configuration options.
        check_default(): Prints the default configuration settings.
    """
    VERSION = '0.0.10'
    CREATED_BY = 'jasonheesanglee\thttps://github.com/jasonheesanglee'

    def cfg_info(self) -> None:
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
            'default_emotion_col': 'default_emotion',
            'normalized_emotion_col': 'gpt_emotion',
            'sentence_emotion_col': 'klue_emo',
            'sentence_specific_emotion_col': 'klue_specific_emo',
            'truncation': True,
            'max_length': 512
        }

        if cfg is not None:
            if isinstance(cfg, dict):
                config.update(cfg)
        default_cfg = config

        self.model_name = default_cfg['model_name']
        self.sub_model_name = default_cfg['sub_model_name']
        self.word_senti_model_name = default_cfg['word_senti_model_name']
        self.text_col = default_cfg['text_col']
        self.original_emotion_column = default_cfg['original_emotion_column']
        self.normalized_emotion_col = default_cfg['normalized_emotion_col']
        self.default_emo_col = default_cfg['default_emotion_col']
        self.sentence_emotion_col = default_cfg['sentence_emotion_col']
        self.sentence_specific_emotion_col = default_cfg['sentence_specific_emotion_col']
        self.truncation = default_cfg['truncation']
        self.max_length = default_cfg['max_length']

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
            'default_emotion_col' : self.default_emo_col,
            'normalized_emotion_col' : self.normalized_emotion_col,
            'sentence_emotion_col' : self.sentence_emotion_col,
            'sentence_specific_emotion_col' : self.sentence_specific_emotion_col,
            'truncation' : self.truncation,
            'max_length' : self.max_length
        }