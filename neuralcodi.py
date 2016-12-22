__author__ = "Laura Martinez Sanchez"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "lmartisa@gmail.com"

rbm = BernoulliRBM(random_state=0, verbose=True)
logistic = linear_model.LogisticRegression()
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X, y)

# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X,y)

predicted = classifier.predict(imgOriginal)
 #pensar en binarizar los datos!!!http://stackoverflow.com/questions/23419165/restricted-boltzmann-machine-how-to-predict-class-labels
#mirar lda!!!!
print logistic_classifier
print"___________________________________________________________________"

###############################################################################
# Evaluation

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        y,
        classifier.predict(X))))

print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        y,
        logistic_classifier.predict(X))))

###############################################################################


