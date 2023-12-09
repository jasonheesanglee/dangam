class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class EmotionSegmentatorConfig:
    VERSION = '0.1.0'
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
        default_cfg = {
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

        if cfg is not None:
            if isinstance(cfg, dict):
                default_cfg.update(cfg)
        default_cfg = DotDict(default_cfg)

        self.model_name = default_cfg.model_name
        self.sub_model_name = default_cfg.sub_model_name
        self.word_senti_model_name = default_cfg.word_senti_model_name
        self.text_col = default_cfg.text_col
        self.ori_emo_col = default_cfg.ori_emo_col
        self.normalized_emo_col = default_cfg.normalized_emo_col
        self.default_emo_col = default_cfg.default_emo_col
        self.sent_emo_col = default_cfg.sent_emo_col
        self.sent_spec_emo_col = default_cfg.sent_spec_emo_col
        self.truncation = default_cfg.truncation
        self.max_length = default_cfg.max_length
