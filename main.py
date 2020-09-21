import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    csv = np.loadtxt('megabaseParsed;0;1;2.txt', delimiter=',', dtype='int')
    # print(np.isnan(np.sum(csv)))
    array = np.array(csv)
    elos = array[:, [1, 2]]
    results = array[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(elos, results, test_size=0.15, random_state=2)
    # model = LogisticRegression(solver='lbfgs', C=0.001, multi_class='auto', random_state=0, n_jobs=-1)
    model = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
    model.fit(x_train, y_train)
    # print("Coefficients (BlackWin, Draw, WhiteWin): ")
    # print(model.coef_)
    # print(model.intercept_)
    # print(model.predict_proba(x_test))
    print("Score training: " + model.score(x_train, y_train).__str__())
    print("Score test: " + model.score(x_test, y_test).__str__())
    y_pred = model.predict(x_test)
    print("Confusion Matrix (black/draw/white):")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
