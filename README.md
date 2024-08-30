# book-reviewer

This application will perform sentiment analysis on the reviews for books on bookmeter.com 

Bookmeter is a Japanese website which allows users to register books they have read and review them. Unfortunately, there is no rating system. This application will use sentiment analysis to determine rating for books. 

The training.txt is a dataset of reviews taken from bookmeter which I have rated as negative or positive. I then use a Japanese version of BERT to train the model. Once the model is trained, selenium is used to grab the html from the website and take the reviews from a specific book. The trained model then performs sentiment analysis on each review and averages them out to get the score for the book.

Potential Improvements:
    The current training dataset was all labeled by hand by me, so it is very small. Increasing the size of the training dataset would allow the model to be more consistent. Another issue, is the reviews are often mixed with summariziations of the book or other noise. This leads to the sentiment analysis being thrown off as positive or neagtive words might be used, but not in reference to the rating of the book. 

Necessary libraries:
    -selenium
    -transformers
    -torch
