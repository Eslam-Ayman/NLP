<h2>Abstract</h2>
This Sentiment Analysis project is about knowing if the review of the movie is positive or negative review , we used more than one model during the process to know which model is the best.

<h2>Data set</h2>
The Dataset we used is labeled dataset consist of 2000 rows of labeled POS and NEG data http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW3/

<h2>Pre-processing</h2>
The classification algorithm will need some sort of feature vector in order to perform the classification task. The simplest way to convert a corpus to a vector format is the bag-of-words approach, where each unique word in a text will be represented by one number. Firstly we use function to removes punctuation, stopwords, and returns a list of the remaining words, or tokens. Then we make vectorization to convert each review into a vector.

<h2>Methodology</h2>
<b>Experiment 1</b>
The first model we used is LinearSVR from SVM from the library sklearn,this is a linear classifier we tried it first with the default loss which is squared_hinge and then the other loss which is hinge and changed the iteration number.


<b>Experiment 2</b>
The second model we used is MultinomialNB from the library sklearn, this is the Naive bayes first we tried it with the all the default parameters then we did cross validation on the Dataset and changed the alpha parameter.


<b>Experiment 3</b>
The last model we used is the LogisticRegression from linear_model from the library sklearn, this model differ from the Naive Bayes that the features weight takes features dependence into account first we tried with the default parameters. then we changed the iteration number and the multiclass parameters

<h2>Results</h2>
<span>
<img src="http://fci.helwan.edu.eg/w/images/8/87/NaiveBayes.PNG">
<img src="http://fci.helwan.edu.eg/w/images/0/08/NaiveBayes_model.PNG">
</span>
