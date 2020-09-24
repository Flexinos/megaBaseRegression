import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.extmath import cartesian


def my_logistic_regression():
    logistic_model = LogisticRegression(solver='lbfgs', C=0.001, multi_class='auto', random_state=0, n_jobs=5)
    logistic_model.fit(x_train, y_train)
    print("Coefficients (BlackWin, Draw, WhiteWin): ")
    print(logistic_model.coef_)
    print(logistic_model.intercept_)
    print(logistic_model.predict_proba(x_test))
    return logistic_model


def my_kneighbors_classifier(neighbors):
    # 10 neighbors -> score test: 0.50
    # 100 neighbors -> score test 0.544
    # 512 neighbors -> score test 0.5478
    print("Fitting kneighbors_model with " + neighbors.__str__() + " neighbors...")
    kneighbors_model = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=-1)
    kneighbors_model.fit(x_train, y_train)
    return kneighbors_model


def check_csv():
    # computationally extremely intensive
    # sum is NaN if at least one element is Nan
    if np.isnan(np.sum(csv)):
        raise Exception("Csv contains Nan!")


if __name__ == '__main__':
    csv = np.loadtxt('megabaseParsed;0;1;2.txt', delimiter=',', dtype='int')
    # check_csv()
    array = np.array(csv)
    elos = array[:, [1, 2]]
    results = array[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(elos, results, test_size=0.05, random_state=0)
    # model = my_logistic_regression()
    model = my_kneighbors_classifier(512)
    y_pred = model.predict(x_test)
    print("Score training: " + model.score(x_train, y_train).__str__())
    print("Score test: " + model.score(x_test, y_test).__str__())
    print("Confusion Matrix (black/draw/white):")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    table_elo_range = np.arange(800, 3000, dtype='int')
    combinations = cartesian([table_elo_range, table_elo_range])
    look_up_table = model.predict_proba(combinations) * 100
    look_up_table = np.c_[combinations, look_up_table]
    np.savetxt('lookuptable.txt', look_up_table, fmt='%.5g')
