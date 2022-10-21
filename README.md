<p align="center">
  <a href="" rel="noopener">
 <img src="https://storage.googleapis.com/kaggle-media/competitions/jigsaw/003-avatar.png" alt="Jigsaw"></a>
</p>
<h1 align="center">Toxic Comments Classification</h1>

<div align="center">
A multi-label classification problem in which we identify the toxicity types of a given comment text.
</div>

## Problem Definition
In an online environment, the comment section of any social media is very fragile these days. The threat of abuse and harassment online will negate the main purpose of social media, which is to exchange information and have healthy relationships with other people. Social media allows you to connect with friends and family, learn new things, develop interests and have fun. Many people stop expressing themselves and give up on seeking different opinions due to online abuse and harassment. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

## Motivation
The aim of this project is to understand how to solve a natural language processing problem and produce a model that is capable of classifying different types of toxicity like threat, obscenity, insult, and identity-based hate.

## Idea
The main idea is to train classification models such as logistic regression and random forest with the given comment texts. Other than that, neural networks are also expected to be the better model to solve this problem since the output of softmax activation function with a threshold will showcase each label prediction probability. In short, neural networks output may have better usability and probably better performance. But since the dataset is imbalanced, the main evaluation metrics that is used to measure the model's performance is the Area Under the Curve (AUC) score. Last but not least, this project also looks to deploy the model into a web application interface created using [Streamlit](https://streamlit.io) for reproducibility purposes.



## Acknowledgements
The data used in this experimentation is taken from [Kaggle's Jigsaw Toxic Comments Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
