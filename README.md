# Political-Inclination
This a machine learning project to classify and predict the political inclination of Twitter users according to their tweets 


Data was collected from Twitter API : https://developer.twitter.com/en/docs

We extracted the tweets posted by the following six Twitter accounts: realDonaldTrump, mike_pence, GOP, HillaryClinton, timkaine, TheDemocrats.

For every tweet, we collected two pieces of information:
- `screen_name`: the Twitter handle of the user tweeting and
- `text`: the content of the tweet.

The overarching goal of the problem is to "predict" the political inclination (Republican/Democratic) of the Twitter user from one of his/her tweets. The ground truth is determined from the `screen_name` of the tweet as follows
### Training Data
- `realDonaldTrump, mike_pence, GOP` are Republicans
- `HillaryClinton, timkaine, TheDemocrats` are Democrats

The tweets_test.csv dataset is used to classify and predict the political inclination of the tweets. 

**Packages:**

1. nltk
2. Counter
3. pandas
4. string
5. numpy 
6. sklearn
