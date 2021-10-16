GRP 22 TOXIC COMMENT CLASSIFIER COURSEWORK:

Objective:
Build a Toxic Comments Classifier using Natural Language Processing (NLP)
that is capable of detecting profanity and finding probabilities for different types of labels such as:

toxic
severe toxic
obscene
threat
insult
identity hate

Problem Statement:
Social media has become a powerful tool and unique place for people to express their views and opinions. But there are people who take advantage of this oppurtunity andmisuse this freedom to show their toxic mindset. Derogatory comments are made while expressing their toxic views. As a result, Cyberbullying has become a major societalproblem thesedays. The level of toxicity could vary from insults to threats to obscene statements and so on. Even though many would say this profanity is just human natureand it should be ignored, as members of civilised society this should not be allowed given the harmful impact it makes on individuals.
Hence it is the responsibility of the host to make sure that the discussions that happen on their online forum is identified and a prevention system is implemented. So a systemshould be developed, that identifies any negative online behaviour and should trigger the prevention unit to take action accordingly. Here we focus on using NLP model tocreate the former part of the system. I.e Identification unit.
It is important that these sort of behaviour that are prfone/vulgar are filtered and moderated. As mentioned above the level of toxicity can be categorized into different labels.Our NLP model can calculate the probability to classify comments into different categories/labels based on level of profanity. In other words we are dealing with a multi labelcomment classification model. Being a multi-label classification, data can belong to more than a label at same time.
That is, given a comment by user, it has to be categorised into one or moThis being a multi-label classification, data can belong to more than a label at same time.re of thefollowing: toxic, severe-toxic, obscene, threat, insult or identity-hate, with probabilties.

Dataset:
The dataset is available via closed kaggle competition. The dataset contains comments from Wikipedia's Wikipedia’s talk page edits.The Wikipedia Talk Page Dataset isprepared by Jigsaw and publicly available at Kaggle[reference]. Data source:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge


Techonlogies:

-->Random forest, NLP and Flask

Dependencies to be downloaded to the conda environment:

dependencies:
  - ipykernel
  - jupyter_client
  - jupyter_core
  - jupyter
  - line_profiler
  - ipython
  - ipython_genutils
  - matplotlib
  - memory_profiler
  - scikit-learn>=0.20
  - scipy
  - seaborn
  - setuptools
  - nltk
  - numpy>=1.16
  - pandas>=0.24.0
  - psutil
  - python>=3.6

Run the application:

  Copy all the files in the location to a folder named toxic classifier


-->Open Anaconda Prompt/terminal

-->Use cd command to change directory to 'Application'

-->Enter the command 'python Toxic_Comment.py' to run the Python file

-->The application will start Running on http://127.0.0.1:5000/ copy this to the browser

--->App is ready.

