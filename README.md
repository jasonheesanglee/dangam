[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/jasonheesanglee/DanGam/blob/master/LICENSE)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/DanGam.svg)](https://pypi.python.org/pypi/DanGam/)
[![PyPI status](https://img.shields.io/pypi/status/DanGam.svg)](https://pypi.python.org/pypi/DanGam/)

# DanGam

DanGam is a Python package designed for advanced emotion analysis in text, mainly focused on the Korean language.<br>
DanGam provides insights into the emotional tone of texts, aiming for more accurate and context-aware sentiment analysis.<br>
The name DanGam came from the abbreviation of "Word-Emotion" in Korean (단어-감정).

> [!IMPORTANT]
> Latest Version of the model is 0.0.133<br>
> GPU Enabled from Version 0.0.130

## Installation
DanGam can be easily installed via pip.<br>Simply run the following command in your terminal:<br>
```shell

pip install DanGam
```

> [!TIP]<br>
> Just in case if you encounter issues with other packages, try run the following command in your terminal before installing DanGam.<br>
> If you still encounter any problems installing DanGam, please report it to me.
```shell
pip install torch numpy pandas tqdm transformers scipy regex
```



## Features
- **Sentence Emotion Segmentation**: DanGam segments sentences and identifies their overarching emotional tone (positive, negative, or neutral).
- **Word-Level Emotion Analysis**: It goes deeper into the emotional analysis by evaluating the sentiment of individual words within the context of their sentences.
- **Customizability**: Flexible configuration options allow users to tailor the analysis to specific requirements.
- **Support for Korean Language**: Specifically for Korean language texts, offering more reliable results than general-purpose sentiment analysis tools.

## Quick Start
```
from dangam import DanGam
```

```shell
# Initialize the DanGam
dangam = DanGam()
# add configuration dictionary if needed.
# details explained after this code cell.

# Example text
text = "나는 방금 먹은 마라탕이 너무 좋다. 적당한 양념에 알싸한 마라향이 미쳤다. 그런데 고수는 진짜 싫다!"
original_emotion = "positive"
default_emotion = "good food"
normalized_specific_emotion = "satisfied"

# Analyze the emotion of the sentence
emotion, specified_emotion = dangam.get_emotion(text, original_emotion, default_emotion, normalized_specific_emotion)

print("Sentence Emotion:", emotion)
print("Specified Emotion:", specified_emotion)
#### Sentence Emotion: positive
#### Specified Emotion: satisfied

# Analyze the emotion of each word

words_emotion = dangam.word_emotions(text, emotion, specified_emotion)
print(words_emotion)
# {'나는': 1.0,
# '방금': 0.8419228076866834,
# '먹은': 1.0,
# '마라탕이': 0.8522973110543406,
# '너무': 1.0,
# '좋다': 1.0,
# '적당한': 0.965806179144829,
# '양념에': 0.7151325862316465,
# '알싸한': 0.4678710873322536,
# '마라향이': 0.328179239525493,
# '미쳤다': 0.34263925379014165,
# '그런데': -0.07491504014905744,
# '고수는': -0.7992964009024587,
# '진짜': -0.9295882226863167,
# '싫다': -0.9120299268217638}
```

## Configuration
DanGam allows a wide range of degrees of customization. <sub>~~(at least trying)~~</sub> <br>
You can modify various settings like model names, column names, etc., to fit your specific needs.
- Initialization:
  - When initially calling `DanGam`, you can add configuration setting in a form of Dictionary.<br>
    ```
    dangam = DanGam(cfg:dict)
    ```
  - The dictionary should be in the format of<br>
    ```
    {"model_name":"hf/some_model", "sub_model_name":"hf/some_model", ...}
    ```
  - You can modify a part of the configuration; it will use the default configuration for not mentioned ones.<br><br>
`config_info()`:
  - Prints the current configuration information of the DanGam.
  - Includes details about the models used, text and emotion column names, and other settings.<br><br>
`check_default()`:
  - Outputs the default configuration values for reference.<br><br>
`check_config()`:
  - Returns the current configuration of DanGam as a dictionary.<br><br>
`update_config(config)`:
  - Update the configuration of DanGam and reinitialize components as necessary.<br><br>
  
  <details>
    <summary>List of modifiable configurations</summary>
      
      - model_name
        - The model that will run through the first loop of the sentence segmentation.
  
      - sub_model_name
        - The model that will run through the second loop of the sentence segmentation.
  
      - word_senti_model_name
        - The model that will through the loop of the word segmentation.
  
      - text_col
        - The name of the column that you want to segment the emotion.
  
      - default_emotion_column
        - Pre-labeled emotion by user.
  
      - original_emotion_column
        - Pre-segmented emotions by user.
        - Performs the best if this section is segmented into 'positive', 'negative', 'neutral'.
        - Used for accuracy evaluation.
  
      - normalized_emotion_column
        - Normalized pre-labeled emotion.
        - Performs the best if this section is in English.
        - Directly used from the second loop, since it will only segment positive, negative, neutral.
        - Not into 60 different emotions.
  
      - sentence_emotion_column
        - The column name of sentence emotion (pos/neg/neut) you want this module to set.
  
      - sentence_specific_emotion_column
        - The column name of sentence emotion (pos/neg/neut) you want this module to set.
  
      - truncation
        - Turning on and off Truncation throughout the module.
  
      - max_length
        - Max length for chunk_text
  
      - emotion_threshold
        - The threshold for emotion and specific emotion embeddings are adjusted accordingly to refine the combined embedding, ensuring a more nuanced sentiment analysis.

      - alignment_threshold
        - The threshold for the cosine similarity between the combined sentence-emotion embedding and each individual word embedding.
  
      - emotion_weight_reach_threshold
        - The weight to be multiplied on emotion embedding when similarity exceeds the threshold.
  
      - emotion_weight_not_reach_threshold
        - The weight to be multiplied on emotion embedding when similarity doesn't exceed the threshold.
  
      - specific_weight_reach_threshold
        - The weight to be multiplied on specific emotion embedding when similarity exceeds the threshold.
  
      - specific_weight_not_reach_threshold
        - The weight to be multiplied on specific emotion embedding when similarity doesn't exceed the threshold.
  
      - noun_threshold
        - The threshold for deciding the emotion segment of a word.
  </details>

## Core Functionality
The primary objective of `word_segmentator` is to assign sentiment scores to each word in a given sentence.<br>
These scores are not just arbitrary numbers; they represent how closely each word aligns with the overall emotional tone of the sentence.<br>This process involves several steps, starting from embedding extraction to sentiment score normalization.<br><br>

`get_emotion(sentence, origianl_emotion, default_specific_emotion, normalized_emotion)`:<br>
  Determines the overall emotion of a given sentence by analyzing it in chunks.<br>
  Considers both the general and specific emotions to enhance accuracy.<br>
  
  Arguments:
    - `sentence` : str - *The sentence to extract the emotions from.*
    - `original_emotion` (str) -> optional : *The pre-segmented emotion (positive, negative, neutral)*
    - `default_specific_emotion` (str) -> optional : *The pre-segmented specific emotion (love, thrilled, happy, sad, etc..)*
    - `normalized_emotion` (str) -> optional : *Normalized User input emotion (good food, bad person, lovely day, etc..)*
  Returns:
    - `emotion` (str) : *A string of overall emotion of the sentence. (positive, neutral, negative)*
    - `specific_emotion` (str) : *A string of specific emotion of the sentence. (one out of 60 emotions)*

> [!WARNING]<br>
> Depreciated from `Ver 0.0.133`<br>
>   ~~`match_rate_calc(df)`:<br>~~
>     ~~Calculates the accuracy of emotion predictions in a dataframe by comparing predicted emotions with their original annotations.<br><br>~~

    
`word_emotions(sentence, emotion, specific_emotion)`:<br>
  Segments a sentence and assigns emotions to each word based on the overall sentence emotion and specific emotion.<br>
  Args:<br>
    - `sentence` (str): The sentence for segmentation.
    -  `emotion` (str) -> Optional: The general emotion of the sentence.
    -  `specific_emotion` (str) -> Optional: The specific emotion of the sentence.

  Returns:
      `dict`: A dictionary mapping each word in the sentence to its assigned emotion.

`noun_emotions(sentence, noun_list, count)`:<br>
  Analyzes emotions associated with specific nouns within a sentence.
  Args:
      `sentence` (str): The sentence containing the nouns for emotion analysis.
      `emotion` (str) -> Optional: The general emotion of the sentence.
      `specific_emotion` (str) -> Optional: The specific emotion of the sentence.
      `noun_list` (list): A list of nouns to analyze within the sentence.
      `count` (bool) : True or False for switching on off counting the number of nouns in each segment.
  Returns:
      `dict`: A dictionary categorizing nouns into positive, neutral, and negative based on their associated emotions.
  

## Embedding Extraction and Analysis
The function begins by extracting embeddings for each word in the sentence, as well as for the sentence as a whole.<br>Embeddings are essentially numerical representations that capture the semantic essence of words and sentences.<br>For a more nuanced analysis, it also considers specific emotion embeddings, which are representations of predefined emotional states or tones.<br>By comparing word embeddings with the sentence and emotion embeddings, the function can gauge the degree of emotional congruence or divergence each word has with the overall sentence sentiment.

## Sentiment Score Calculation
The core of `word_segmentator` lies in calculating these sentiment scores.<br>It does so by measuring the cosine similarity between the combined sentence and emotion embeddings and individual word embeddings.<br>This similarity metric is then adjusted to account for dissimilarities.<br>The function implements a threshold-based mechanism to enhance the accuracy of these calculations, ensuring that the scores genuinely reflect whether each word shares or contrasts with the sentence's emotional tone.

## Normalization and Interpretation
Post-calculation, the sentiment scores undergo normalization, a crucial step to ensure that the scores are within a consistent range (typically -1 to 1).<br>This normalization helps in interpreting the scores uniformly across different sentences and contexts.<br>A score closer to 1 indicates a strong alignment with the sentence's emotion, whereas a score near -1 suggests a contrast.<br>Scores around 0 imply neutrality or a lack of strong emotional alignment.

## Diagram
<img src="https://github.com/jasonheesanglee/dangam/blob/2da81bac42fde4688590c5c6981d496ea5d10fa1/data/DanGam.png">

## Contributing
Contributions to DanGam are welcome!<br>
Whether it's feature requests, bug reports, or code contributions, please feel free to contribute.<br>
<sub>If you are interested in hiring me, please feel free to contact <a href="mailto:jason.heesang.lee96@gmail.com">jason.heesang.lee96@gmail.com</a>

## License
Dangam is released under MIT License, making it suitable for both personal and commercial use.

