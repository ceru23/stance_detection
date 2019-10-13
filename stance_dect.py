import numpy as np
import csv
import pandas as pd
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

filename_train = 'SemEval2016-Task6-subtaskA-traindata-gold.csv'
filename_test = 'SemEval2016-Task6-subtaskA-testdata-gold.txt'
filename_second_task = 'stance.csv'

stopwords = stopwords.words("english")


def load_file(filename,quotechar=None,delimiter=None):
    '''
    Returns data frame of given csv file given its path.
    Can use delimiter or quotechar.

    Returns:    dataframe
    '''

    data = []

    with open(filename,'r',encoding='iso-8859-1',errors='ignore') as f:
        if(delimiter):
            reader = csv.reader(f, delimiter=delimiter)
        elif(quotechar):
            reader = csv.reader(f, quotechar=quotechar)
        else:
            raise Exception("Delimiter or quotechar must be provided.")
        columns = next(reader)
        for line in reader:
            data.append(line)


    return pd.DataFrame(data,columns=columns)


def clean_text(text):
    '''
    Remove non alphanumeric character, lower string,
    extract tokens

    Returns: list of tokens
    '''

    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = [w for w in text.split(" ") if w not in stopwords]

    return text



def create_word2vec(corpus,**kwargs):
    '''
    Wrapper function for Word2Vec, init a model
    and return trained models

    Returns: Word2Vec object
    '''

    model = Word2Vec(corpus,**kwargs)
    model.train(corpus,total_examples=len(corpus),epochs=50)

    return model

def visualize_word2vec(model,n_vectors=100):
    '''
    Visualize word2vec vectors in two dimension
    reduced using tSNE. n_vectors is the size of
    a random sample of vectors in vocabolary.
    If n_vectors is None, all the words vectors are considered.
    '''


    token = []
    labels = []

    for word in model.wv.vocab:
        token.append(model[word])
        labels.append(word)

    if(n_vectors is not None):
        token = random.choices(token,k=n_vectors)
        labels = random.choices(labels,k=n_vectors)


    tsne = TSNE(
        perplexity=40,
        n_components=2,
        n_iter=1000)
    red_token = tsne.fit_transform(token)

    X = red_token[:,0]
    Y = red_token[:,1]

    plt.scatter(X,Y)

    for label,x,y in zip(labels,X,Y):
        plt.annotate(
            label,
            xy=(x,y),
            xytext=(0,0),
            textcoords='offset points'
            )
    plt.xlim(X.min()+0.00005, X.max()+0.00005)
    plt.ylim(Y.min()+0.00005, Y.max()+0.00005)
    plt.show()

def embed_text(data_frame,text_column,model_wv):
    '''
    Extact text from the text_column of data_frame
    and embed vectors with the word2vec model_vw

    Returns: array of the average word vectors for row
    '''

    vocabolary = set(model_wv.wv.vocab)
    text = []
    if(isinstance(text_column,list)):
        for i in range(len(text_column)):
            if i==0:
                text = data_stance[text_column[i]]
            else:
                text = text+" "+data_stance[text_column[i]]
        text = text.apply(clean_text)
    else:
        text = data_frame[text_column].apply(clean_text)
    embed = []
    for t in text:
        embed.append(np.mean([model_wv.wv[w] for w in t if w in vocabolary],axis=0))

    return np.asarray(embed)


def extract_labels(lab,lab_names):
    '''
    Convert three labels with valueles
    [-1,0,1]

    Returns: list of labels
    '''

    labels = []
    for l in lab:
        if(l==lab_names[0]):
            labels.append(-1)
        elif(l==lab_names[1]):
            labels.append(0)
        else:
            labels.append(1)

    return labels

def train_single_svm(x_train,y_train,x_test,y_test,**kwargs):
    '''
    Given train and test sets with labels, train and return
    a Support Vector Machine with parametrs in **kwargs
    Prints the results of 5-folds cross validation,
    a classification report and the comparison with
    the classificaiotn report of the majority classifier

    Returns: trained SVM
    '''



    clf = sklearn.svm.SVC(**kwargs)
    clf.fit(x_train,y_train)
    cross_score = sklearn.model_selection.cross_val_score(clf,x_train,y_train,cv=5)
    print("5-folds cross validation: ",cross_score)

    y_hat = clf.predict(x_test)

    print("Single SVM on stance given text: ")
    print(sklearn.metrics.classification_report(y_test,y_hat))

    max_label = max(y_train,key=y_train.count)
    print("Majority classifier on stance given text: ")
    print(sklearn.metrics.classification_report(y_test,np.ones(len(y_hat))*max_label))

    return clf


def resample_data(dataframe,labels,support,n_resample,label_column='Stance'):
    '''
    Returns a resampled dataframe, having in input the
    unbalanced dataframe, labels, support, the label column
    and the number of resample, n_resample an integer (in that case
    all the classes will be resampled to have n_resample points) or a list
    one value for each class. If a class has less point than its resample
    value, selection with repetition is applied.

    Returns: resampled DataFrame
    '''



    if(not isinstance(n_resample,list)):
        n_resample = n_resample*np.ones(len(labels)).astype(np.int)

    resampled_data = []
    for i in range(len(labels)):
        t_data = dataframe.loc[dataframe[label_column]==labels[i]]
        t_data = sklearn.utils.resample(
                        t_data,
                        replace=(support[i]<n_resample[i]),
                        n_samples=n_resample[i]
                        )
        resampled_data.append(t_data)

    return pd.concat(resampled_data)


def train_knn(x,y,x_test,y_test,neigh=15):
    '''
    Trains a k-nearest neighbour classifier, having in
    input trainset and testset and the number
    of neighbors (default = 15).
    Prints classification report, and classification
    report of the random classifier for comparison.

    Returns: trained K-Nearest Neighbor classifier
    '''

    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neigh)
    knn.fit(x,y)
    y_hat = knn.predict(x_test)

    y_random = random.choices(list(set(y_test)),k=len(y_test))

    print("knn on target given text: ")
    print(sklearn.metrics.classification_report(y_test,y_hat))
    print("Random classifier on target given text: ")
    print(sklearn.metrics.classification_report(y_test,y_random))

    return knn



def knn_plus_svm(data_train,data_test,x_test,y_test,svm_target,predicted_classes,knn):
    '''
    Given a train and a test dataframe, and a
    predicted targets or a trained knn classifier
    to infer them, prints the classification scores
    of the SVMs given in input (if svm_target is none
    the SVMs are trained on each classes).
    The prediction is performed using knn for
    predicting the classes, and the using the
    relative SVM for each of them.

    Returns: dictionary of trained SVM for each class
    '''


    target = data_train['Target'].unique()

    if(svm_target is None):
        svm_target = {}

        for t in target:
            class_data_train =  data_train.loc[data_train['Target']==t]
            x_class_train = embed_text(class_data_train,'Tweet',model)
            y_class_train = extract_labels(class_data_train['Stance'].tolist(),['AGAINST','NONE','FAVOR'])
            class_data_test = data_test.loc[data_test['Target']==t]
            x_class_test = embed_text(class_data_test,'Tweet',model)
            y_class_test = extract_labels(class_data_test['Stance'].tolist(),['AGAINST','NONE','FAVOR'])

            print("SVM for class ",t)
            svm_class = train_single_svm(
                            x_class_train,
                            y_class_train,
                            x_class_test,
                            y_class_test,
                            kernel='rbf',
                            gamma='scale',
                            decision_function_shape='ovr')
            svm_target[t] = svm_class



    if(predicted_classes is None):
        predicted_classes = knn.predict(x_test)

    y_final = []
    y_hat_final = []
    for t in target:

        x_class = []
        y_class = []
        for j in range(len(predicted_classes)):
            if(predicted_classes[j]==t):
                x_class.append(x_test[j,:])
                y_class.append(y_test[j])

        y_hat_class = svm_target[t].predict(x_class)
        y_final.extend(y_class)
        y_hat_final.extend(y_hat_class)


    print("OVERALL: ")
    print(sklearn.metrics.classification_report(y_final,y_hat_final))

    return svm_target


if __name__=='__main__':

    # loading data
    data_train = load_file(filename_train,quotechar='"')
    data_test = load_file(filename_test,delimiter='\t')
    data_stance = load_file(filename_second_task,delimiter=',')

    # extracting raw text
    corpus = pd.concat([data_train.Tweet,data_test.Tweet]).to_list()

    # prepocess text and return tokens
    clean_corpus = list(map(clean_text,corpus))

    # train a Word2Vec model on the corpus
    model = create_word2vec(clean_corpus,
                        size=150,
                        window=12,
                        min_count=2,
                        workers=16)


    # model.save("tweet2vec_corpus")
    #model = Word2Vec.load("tweet2vec_corpus")

    # visualize the obtained word embedding
    visualize_word2vec(model,n_vectors=100)  ###check random img migliore

    # create test and train set
    x_train = embed_text(data_train,'Tweet',model)
    y_train = extract_labels(data_train['Stance'].tolist(),['AGAINST','NONE','FAVOR'])

    x_test = embed_text(data_test,'Tweet',model)
    y_test = extract_labels(data_test['Stance'].tolist(),['AGAINST','NONE','FAVOR'])


    # train a single SVM for all the 5 targets
    print("---------------------------")
    print("RESULTS OF A SINGLE SVM : ")
    print("---------------------------")

    svm_1 = train_single_svm(
                x_train,
                y_train,
                x_test,
                y_test,
                kernel='rbf',
                gamma='scale',
                decision_function_shape='ovr')

    print("---------------------------")

    support = data_train['Stance'].value_counts().to_list()
    labels = data_train['Stance'].unique()


    # resample the train dataset
    data_resampled = resample_data(
                data_train,
                support=support,
                labels=labels,
                n_resample=[1000,800,800]
                )

    x_train_res = embed_text(data_resampled,'Tweet',model)
    y_train_res = extract_labels(data_resampled['Stance'].tolist(),['AGAINST','NONE','FAVOR'])


    print("---------------------------")
    print("RESULTS OF A SINGLE SVM ON RESMAPLED DATA : ")
    print("---------------------------")

    # train a SVM for all the 5 targets on the resampled dataset
    svm_1_res = train_single_svm(
                x_train_res,
                y_train_res,
                x_test,
                y_test,
                kernel='rbf',
                gamma='scale',
                decision_function_shape='ovr')

    print("---------------------------")

    y_train_target = data_train['Target']
    y_test_target = data_test['Target']

    # train a KNN on the 5 targets
    print("---------------------------")
    print("RESULTS OF KNN CLASSIFIER : ")
    print("---------------------------")


    knn = train_knn(
                x_train,
                y_train_target,
                x_test,
                y_test_target
                )

    print("---------------------------")

    # train one svm for each target class with knn for class inference

    print("---------------------------")
    print("RESULTS OF 5 SVMs WITH KNN LABELS : ")
    print("---------------------------")

    svm_ensemble_knn = knn_plus_svm(
                        data_train,
                        data_test,
                        x_test,
                        y_test,
                        predicted_classes=None,
                        svm_target=None,
                        knn=knn)

    # train one svm for each target class with class labels
    print("---------------------------")

    print("---------------------------")
    print("RESULTS OF 5 SVMs WITH TEST LABELS : ")
    print("---------------------------")

    svm_ensemble_labels = knn_plus_svm(
                        data_train,
                        data_test,
                        x_test,
                        y_test,
                        predicted_classes=y_test_target,
                        svm_target=None,
                        knn=None)


    map_label = {'hillary':'Hillary Clinton',
            'abortion':'Legalization of Abortion',
            'climate':'Climate Change is a Real Concern',
            'feminism':'Feminist Movement'
            }

    # test model on stance.csv data file

    x_stance_title = embed_text(data_stance,'title',model)
    x_stance_text = embed_text(data_stance,'text',model)
    x_stance_combined = embed_text(data_stance,['title','text'],model)
    y_stance = [map_label[y] for y in data_stance['controversial trending issue']]


    y_hat_stance_title = knn.predict(x_stance_title)
    y_hat_stance_text = knn.predict(x_stance_text)
    y_hat_stance_combined = knn.predict(x_stance_combined)


    print("---------------------------")
    print("KNN CLASSIFIER ON STANCE TASK : ")
    print("---------------------------")

    print("KNN CLASSIFIER WITH TITLE ONLY : ")

    print(sklearn.metrics.classification_report(y_stance,y_hat_stance_title))

    print("KNN CLASSIFIER WITH TEXT ONLY : ")

    print(sklearn.metrics.classification_report(y_stance,y_hat_stance_text))

    print("KNN CLASSIFIER COMBINED : ")

    print(sklearn.metrics.classification_report(y_stance,y_hat_stance_combined))


    title = data_stance['title'].to_list()
    text = data_stance['text'].to_list()


    #using svm trained on the whole data to predict stance

    y_hat_class  = svm_1_res.predict(x_stance_title)

    data_stance['stance'] = pd.Series(y_hat_class, index=data_stance.index)

    print("---------------------------")
    print("RESULTS OF 1 SVM ON STANCE : ")
    print("---------------------------")

    for t in data_stance['controversial trending issue'].unique():
        print(t)
        print(data_stance[data_stance['controversial trending issue']==t].stance.value_counts())

    print("---------------------------")

    #using one svm trained per class and dataset labels
    print("---------------------------")
    print("RESULTS OF CLASS SVMs ON STANCE  USING LABELS : ")
    print("---------------------------")


    target = list(set(y_stance))

    for t in target:
        x_class = []
        title_class = []
        text_class = []
        for i in range(len(y_stance)):
            if(y_stance[i]==t):
                x_class.append(x_stance_title[i,:])
                title_class.append(title[i])
                text_class.append(text[i])


        y_hat_class = svm_ensemble_labels[t].predict(x_class)
        temp_data_frame = pd.DataFrame(data={'title':title_class,
                                            'text':text_class,
                                            'target':t,
                                             'stance':y_hat_class})
        print(t,"   ",temp_data_frame['stance'].value_counts())

    print("---------------------------")

    #using one svm trained per class and knn inferred labels
    #if label is "Atheism" use the svm trained to the whole data
    print("---------------------------")
    print("RESULTS OF CLASS SVMs ON STANCE  USING KNN INFERRED LABELS : ")
    print("---------------------------")

    target = list(set(y_hat_stance_title))

    for t in target:
        x_class = []
        title_class = []
        text_class = []
        for i in range(len(y_hat_stance_title)):
            if(y_hat_stance_title[i]==t):
                x_class.append(x_stance_title[i,:])
                title_class.append(title[i])
                text_class.append(text[i])

        if(t is not "Atheism"):
            y_hat_class = svm_ensemble_labels[t].predict(x_class)
        else:
            y_hat_class = svm_1_res.predict(x_class)
        temp_data_frame = pd.DataFrame(data={'title':title_class,
                                        'text':text_class,
                                        'target':t,
                                         'stance':y_hat_class})
        print(t,"   ",temp_data_frame['stance'].value_counts())


    print("---------------------------")
