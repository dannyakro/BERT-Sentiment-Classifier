#### What has been improved
1. Stopword removal in initial raw text 
2. Multiprocessing of BERT Outputs 
3. Understanding "ranking with the Google n-gram dataset", where do we get the rankings ? 

#### What else needs to be done 
1. Better documentation 
2. Tagging of the modularity classes
3. Asking some questions on the output from the sentiment classifier 

## Understood Processes: 

1. Mask the words that are interesting. We do this by a for loop through the sentence and base interesting keywords off the positions and relative positions of the keywords to words that potentially denote interesting information e.g. skip https:// etc

2. Contextualising the emotion-denoting words using [BERT](https://huggingface.co/bert-base-uncased.). What BERT does: Given a sentence "this is making me so \<mask\>, I am having a great day", Predicts the potential \<mask\> word. Output results can be "happy", "ecstatic". Here, we use a pre-trained model. 

3. Generate a synonym network. How do we do this?: 

    a. IMDB review data is scraped. Tokenize the words in the sentences, sum and rank them based on frequency in the dataset. Compare the rankings to [Google-n Gram] dataset, which allows us to select the top 1K "emotion denoting words" in this dataset. 
    
    b. From this 1K keywords, snowball it with the Thesauras API. Multiple root words can lead to the same synonym keywords and this will be our "connections" or edges. \
    
    c. Construct network map, clustering the data based on these edges. We get a network map with Authority Scores (Which opposes Hub scores, but is calculated as the sum of hub scores from nodes pointing to it nonetheless
    
    d. Mapping network scores - Self Score: The emotion denoting word and its own score from the Network Map. Pred Score: The predicted word outputs from BERT and their respective scores from the Network Map. 
        
    e. Get the average predicted metric scores from all of the corpus documents. i.e. Average Predicted Authority Scores, Average Degree Centrality, Average Betweeness Centrality.
    
    f. We use the network words to tag the words. 
    
    g. With all of these average metrics, split 80% 20% for training and testing set. Train with Random Forest Classifier, Decision Tree Classifier, Logistic Regression. Select the best model out of all of these. 


