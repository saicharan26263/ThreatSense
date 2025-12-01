from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
import keras


le = preprocessing.LabelEncoder()
filename = ""
feature_extraction = None
X = Y = doc = label_names = None
X_train = X_test = y_train = y_test = None

lstm_acc = cnn_acc = svm_acc = knn_acc = dt_acc = random_acc = nb_acc = 0
lstm_precision = cnn_precision = svm_precision = knn_precision = dt_precision = random_precision = nb_precision = 0
lstm_recall = cnn_recall = svm_recall = knn_recall = dt_recall = random_recall = nb_recall = 0
lstm_fm = cnn_fm = svm_fm = knn_fm = dt_fm = random_fm = nb_fm = 0


main = Tk()
main.title("ThreatSense")
main.geometry("1350x750")
main.config(bg='#e0f7fa')


title_font = ('Helvetica', 18, 'bold')
title = Label(main, text='ThreatSense: Intelligent Threat Detection', bg='#006064', fg='white', font=title_font)
title.pack(fill=X, pady=10)


text_frame = Frame(main)
text_frame.pack(pady=10)
text = Text(text_frame, height=15, width=150, font=('Helvetica', 12))
scroll = Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)
text.pack(side=LEFT)
scroll.pack(side=RIGHT, fill=Y)

def upload():
    global filename, X, Y, doc, label_names
    filename = filedialog.askopenfilename(initialdir="datasets", title="Select CSV File", filetypes=[("CSV Files","*.csv")])
    if not filename:
        return
    dataset = pd.read_csv(filename)
    label_names = dataset.labels.unique()
    dataset['labels'] = le.fit_transform(dataset['labels'])
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values.astype(int)
    doc = [' '.join(map(str, X[i])) for i in range(len(X))]
    text.delete('1.0', END)
    text.insert(END, f"Loaded dataset: {filename}\nTotal samples: {len(dataset)}\n")

def tfidf():
    global X, feature_extraction
    feature_extraction = TfidfVectorizer()
    X = feature_extraction.fit_transform(doc).toarray()
    text.delete('1.0', END)
    text.insert(END, "TF-IDF processing completed.\n")

def eventVector():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    text.delete('1.0', END)
    text.insert(END, f"Unique labels: {label_names}\n")
    text.insert(END, f"Training samples: {len(X_train)}\nTesting samples: {len(X_test)}\n")
def neuralNetwork():
    global lstm_acc, lstm_precision, lstm_recall, lstm_fm
    global cnn_acc, cnn_precision, cnn_recall, cnn_fm
    
    text.delete('1.0', END)
    Y1 = Y.reshape(-1,1)
    X_train1, X_test1, y_trains1, y_tests1 = train_test_split(X, Y1, test_size=0.2, random_state=42)

    enc = OneHotEncoder(sparse_output=False)
    y_train1 = enc.fit_transform(y_trains1)
    y_test1 = enc.transform(y_tests1)

    X_train2 = X_train1.reshape((X_train1.shape[0], X_train1.shape[1], 1))
    X_test2 = X_test1.reshape((X_test1.shape[0], X_test1.shape[1], 1))
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train1.shape[1],1)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_train1.shape[1], activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_train2, y_train1, epochs=1, batch_size=64, verbose=0)
    pred = np.argmax(model.predict(X_test2), axis=1)
    y_test_labels = np.argmax(y_test1, axis=1)
    lstm_acc = accuracy_score(y_test_labels, pred) * 100
    lstm_precision = precision_score(y_test_labels, pred, average='macro') * 100
    lstm_recall = recall_score(y_test_labels, pred, average='macro') * 100
    lstm_fm = f1_score(y_test_labels, pred, average='macro') * 100

    #/text.insert(END, f"LSTM Accuracy: {lstm_acc:.2f}\nPrecision: {lstm_precision:.2f}\nRecall: {lstm_recall:.2f}\nF-measure: {lstm_fm:.2f}\n")


    cnn_model = Sequential()
    cnn_model.add(Dense(512, input_shape=(X_train1.shape[1],), activation='relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(512, activation='relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(y_train1.shape[1], activation='softmax'))
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist1 = cnn_model.fit(X_train1, y_train1, epochs=10, batch_size=128, validation_split=0.2, verbose=0)
    pred_cnn = np.argmax(cnn_model.predict(X_test1), axis=1)
    cnn_acc = accuracy_score(y_test_labels, pred_cnn) * 100
    cnn_precision = precision_score(y_test_labels, pred_cnn, average='macro') * 100
    cnn_recall = recall_score(y_test_labels, pred_cnn, average='macro') * 100
    cnn_fm = f1_score(y_test_labels, pred_cnn, average='macro') * 100

    text.insert(END, f"\nCNN Accuracy: {cnn_acc:.2f}\nPrecision: {cnn_precision:.2f}\nRecall: {cnn_recall:.2f}\nF-measure: {cnn_fm:.2f}\n")

def svmClassifier():
    global svm_acc, svm_precision, svm_recall, svm_fm
    text.delete('1.0', END)
    cls = svm.SVC(C=2.0, kernel='linear', gamma='scale', random_state=0)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    svm_acc = accuracy_score(y_test, pred) * 100
    svm_precision = precision_score(y_test, pred, average='macro') * 100
    svm_recall = recall_score(y_test, pred, average='macro') * 100
    svm_fm = f1_score(y_test, pred, average='macro') * 100
    text.insert(END, f"SVM Accuracy: {svm_acc:.2f}\nPrecision: {svm_precision:.2f}\nRecall: {svm_recall:.2f}\nF-measure: {svm_fm:.2f}\n")

def knnClassifier():
    global knn_acc, knn_precision, knn_recall, knn_fm
    text.delete('1.0', END)
    cls = KNeighborsClassifier(n_neighbors=10)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    knn_acc = accuracy_score(y_test, pred) * 100
    knn_precision = precision_score(y_test, pred, average='macro') * 100
    knn_recall = recall_score(y_test, pred, average='macro') * 100
    knn_fm = f1_score(y_test, pred, average='macro') * 100
    text.insert(END, f"KNN Accuracy: {knn_acc:.2f}\nPrecision: {knn_precision:.2f}\nRecall: {knn_recall:.2f}\nF-measure: {knn_fm:.2f}\n")

def randomForestClassifier():
    global random_acc, random_precision, random_recall, random_fm
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=5, random_state=0)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    random_acc = accuracy_score(y_test, pred) * 100
    random_precision = precision_score(y_test, pred, average='macro') * 100
    random_recall = recall_score(y_test, pred, average='macro') * 100
    random_fm = f1_score(y_test, pred, average='macro') * 100
    text.insert(END, f"Random Forest Accuracy: {random_acc:.2f}\nPrecision: {random_precision:.2f}\nRecall: {random_recall:.2f}\nF-measure: {random_fm:.2f}\n")

def naiveBayesClassifier():
    global nb_acc, nb_precision, nb_recall, nb_fm
    text.delete('1.0', END)
    cls = BernoulliNB()
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    nb_acc = accuracy_score(y_test, pred) * 100
    nb_precision = precision_score(y_test, pred, average='macro') * 100
    nb_recall = recall_score(y_test, pred, average='macro') * 100
    nb_fm = f1_score(y_test, pred, average='macro') * 100
    text.insert(END, f"Naive Bayes Accuracy: {nb_acc:.2f}\nPrecision: {nb_precision:.2f}\nRecall: {nb_recall:.2f}\nF-measure: {nb_fm:.2f}\n")

def decisionTreeClassifier():
    global dt_acc, dt_precision, dt_recall, dt_fm
    text.delete('1.0', END)
    cls = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    dt_acc = accuracy_score(y_test, pred) * 100
    dt_precision = precision_score(y_test, pred, average='macro') * 100
    dt_recall = recall_score(y_test, pred, average='macro') * 100
    dt_fm = f1_score(y_test, pred, average='macro') * 100
    text.insert(END, f"Decision Tree Accuracy: {dt_acc:.2f}\nPrecision: {dt_precision:.2f}\nRecall: {dt_recall:.2f}\nF-measure: {dt_fm:.2f}\n")

def plotGraph(metric, title_name, ylabel_name):
    values = [knn_acc, nb_acc, dt_acc, svm_acc, random_acc, cnn_acc] if metric=='accuracy' else \
             [knn_precision, nb_precision, dt_precision, svm_precision, random_precision, cnn_precision] if metric=='precision' else \
             [knn_recall, nb_recall, dt_recall, svm_recall, random_recall, cnn_recall] if metric=='recall' else \
             [knn_fm, nb_fm, dt_fm, svm_fm, random_fm, cnn_fm]
    labels = ['KNN','NB','DT','SVM','RF','CNN']
    plt.figure(figsize=(10,6))
    plt.bar(labels, values, color='teal')
    plt.title(title_name)
    plt.ylabel(ylabel_name)
    plt.ylim(0, 100)
    plt.show()

def graphAccuracy(): plotGraph('accuracy', 'Accuracy Comparison', 'Accuracy (%)')
def graphPrecision(): plotGraph('precision', 'Precision Comparison', 'Precision (%)')
def graphRecall(): plotGraph('recall', 'Recall Comparison', 'Recall (%)')
def graphFmeasure(): plotGraph('fmeasure', 'F-measure Comparison', 'F-measure (%)')

# ---------------------- BUTTONS ----------------------
button_frame = Frame(main, bg='#e0f7fa')
button_frame.pack(pady=10)

buttons = [
    ("Upload Dataset", upload),
    ("Run TF-IDF", tfidf),
    ("Generate Event Vector", eventVector),
    ("Neural Network", neuralNetwork),
    ("SVM", svmClassifier),
    ("KNN", knnClassifier),
    ("Random Forest", randomForestClassifier),
    ("Naive Bayes", naiveBayesClassifier),
    ("Decision Tree", decisionTreeClassifier),
    ("Accuracy Graph", graphAccuracy),
    ("Precision Graph", graphPrecision),
    ("Recall Graph", graphRecall),
    ("F-measure Graph", graphFmeasure)
]

for i, (text_btn, cmd) in enumerate(buttons):
    b = Button(button_frame, text=text_btn, command=cmd, width=20, bg='#006064', fg='white', font=('Helvetica', 12, 'bold'))
    b.grid(row=i//3, column=i%3, padx=5, pady=5)

main.mainloop()
