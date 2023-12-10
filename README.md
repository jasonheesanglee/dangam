[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/jasonheesanglee/DanGam/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/DanGam.svg)](https://badge.fury.io/py/DanGam)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/DanGam.svg)](https://pypi.python.org/pypi/DanGam/)
[![PyPI status](https://img.shields.io/pypi/status/DanGam.svg)](https://pypi.python.org/pypi/DanGam/)

# DanGam

DanGam is a Python package designed for advanced emotion analysis in text, particularly focused on Korean language.<br>
Utilizing state-of-the-art NLP models, DanGam provides nuanced insights into the emotional tone of texts, enabling more accurate and context-aware sentiment analysis.<br>
The name DanGam came from the abbreviation of "Word-Emotion" in Korean (단어-감정).

## Installation
DanGam can be easily installed via pip. Simply run the following command in your terminal:<br>
```shell
pip install dangam
```

## Features
- **Sentence Emotion Segmentation**: DanGam segments sentences and identifies their overarching emotional tone (positive, negative, or neutral).
- **Word-Level Emotion Analysis**: It goes deeper into the emotional analysis by evaluating the sentiment of individual words within the context of their sentences.
- **Customizability**: Flexible configuration options allow users to tailor the analysis to specific requirements.
- **Support for Korean Language**: Specifically fine-tuned for Korean language texts, offering more reliable results than general-purpose sentiment analysis tools.

## Quick Start
```
from dangam import DanGam, DanGamConfig
```

```shell
# Initialize the DanGam
dangam = DanGam()

# Example text
text = "나는 방금 먹은 마라탕이 너무 좋다. 적당한 양념에 알싸한 마라향이 미쳤다. 그런데 고수는 진짜 싫다!"

# Analyze the emotion of the sentence
emotion, specified_emotion = dangam.get_emotion(text)

print("Sentence Emotion:", emotion)
print("Specified Emotion:", specified_emotion)
#### Sentence Emotion: positive
#### Specified Emotion: love

# Analyze the emotion of each word

words_emotion = dangam.word_segmentator(text, emotion, specified_emotion)
print(words_emotion)
# [{'나는': 1.0},
# {'방금': 0.7820801305188394},
# {'먹은': 1.0},
# {'마라탕이': 0.8607851484076837},
# {'너무': 1.0},
# {'좋다': 1.0},
# {'적당한': 0.9875665687342202},
# {'양념에': 0.7548636813630957},
# {'알싸한': 0.4555193623403373},
# {'마라향이': 0.3306043180344058},
# {'미쳤다': 0.3766061055294425},
# {'그런데': -0.0704177035812985},
# {'고수는': -0.7508980581598864},
# {'진짜': -0.9509202889420695},
# {'싫다': -0.9517512748457806}]
```

## Configuration
DanGam allows customization through the EmotionSegmentatorConfig class.<br>
You can modify various settings like model names, column names, etc., to fit your specific needs.

## Core Functionality
The primary objective of `word_segmentator` is to assign sentiment scores to each word in a given sentence.<br>These scores are not just arbitrary numbers; they represent how closely each word aligns with the overall emotional tone of the sentence.<br>This process involves several steps, starting from embedding extraction to sentiment score normalization.

## Embedding Extraction and Analysis
The function begins by extracting embeddings for each word in the sentence, as well as for the sentence as a whole.<br>Embeddings are essentially numerical representations that capture the semantic essence of words and sentences.<br>For a more nuanced analysis, it also considers specific emotion embeddings, which are representations of predefined emotional states or tones.<br>By comparing word embeddings with the sentence and emotion embeddings, the function can gauge the degree of emotional congruence or divergence each word has with the overall sentence sentiment.

## Sentiment Score Calculation
The core of `word_segmentator` lies in calculating these sentiment scores.<br>It does so by measuring the cosine similarity between the combined sentence and emotion embeddings and individual word embeddings.<br>This similarity metric is then adjusted to account for dissimilarities.<br>The function implements a threshold-based mechanism to enhance the accuracy of these calculations, ensuring that the scores genuinely reflect whether each word shares or contrasts with the sentence's emotional tone.

## Normalization and Interpretation
Post-calculation, the sentiment scores undergo normalization, a crucial step to ensure that the scores are within a consistent range (typically -1 to 1).<br>This normalization helps in interpreting the scores uniformly across different sentences and contexts.<br>A score closer to 1 indicates a strong alignment with the sentence's emotion, whereas a score near -1 suggests a contrast.<br>Scores around 0 imply neutrality or a lack of strong emotional alignment.

## Contributing
Contributions to DanGam are welcome!<br>
Whether it's feature requests, bug reports, or code contributions, please feel free to contribute.<br>
<small style="color: grey;">If you are interested in hiring me, please feel free to contact <a href="mailto:jason.heesang.lee96@gmail.com">jason.heesang.lee96@gmail.com</a></small>

## License
Dangam is released under MIT License, making it suitable for both personal and commercial use.

