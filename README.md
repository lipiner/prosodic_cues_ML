# prosodic_cues_ML
Using multiple machine learning models to predict emotions labels based on prosodic cues

# Abstract
Whenever we engage in social interaction, it is important for us to accurately identify what our partners feel in order to get along with them. Although we usually good at this, in some situation we can fail to correctly recognize emotions. Thus, having an automatic tool that recognizes emotions can have a big contribution for medical use and security matters, among other fields. Indeed, recent studies have shown that emotions can be classified using machine learning algorithms, based only on acoustic features. Yet, very little work has been done on real, natural and continuous stimuli.
In this research, I used audio stimuli that were collected in the lab and contain recordings of non-actor participants while discussing highly emotional autobiographical events, including the participants’ emotional valence while telling the story. Using three machine learning algorithms (random forests, support vector machines and polynomial regression), I will try to predict various emotional dimensions based on acoustic features. My hypothesis is that a computational model can identify some emotional characteristics based on prosodic cues alone. Although the accuracy level was usually not much above the chance level, the correlation between the predictions and the true ratings (‘EA score’) is significant when the model predicted valence and intensity dimensions. Furthermore, comparing these results to humans found better performances of the computational model when it predicted the intensity dimension, and didn’t find difference in the ability to identify valence dimension.

# Project files
### main files:
- features_extraction.py - extracta features using OpenSmile tool
- main.py - run several ML models (3 different algorithms) on the data with different types of labels
### helper files:
- preprocessing_ratings.py - preprocessing the ratings filea and creates a base labels file and some statistics on the ratings files.
- preprocessing_features.py - contains functions for splitting the data and getting the features and the labels
