from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
_K=3
def listToString(s):
    str1 = ""
    return (str1.join(s))

class classification:
    def __init__(self,x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
    def naiveBayes(self):
        gnb = BernoulliNB()
        y_pred_nb = gnb.fit(self.x_train, self.y_train.ravel()).predict(self.x_test)
        accuracy = '{:.2f}'.format(gnb.score(self.x_test,y_pred_nb))
        return listToString(y_pred_nb), accuracy

    def supportVectorMachine(self):
        clf = SVC()
        y_pred_svm = clf.fit(self.x_train, self.y_train.ravel()).predict(self.x_test)
        accuracy = '{:.2f}'.format(clf.score(self.x_test,y_pred_svm))
        return listToString(y_pred_svm), accuracy

    def KNearestNeighbors(self):
        neigh = KNeighborsClassifier(n_neighbors=_K)
        y_pred_knn = neigh.fit(self.x_train, self.y_train.ravel()).predict(self.x_test)
        accuracy = '{:.2f}'.format(neigh.score(self.x_test,y_pred_knn))
        return listToString(y_pred_knn), accuracy

#  ประกาศ class ที่ชื่อ classification
#  บรรทัดที่ 5-6 convert list เป็น string 
#  บรรทัดที่ 10-13 เมื่อมีการเรียก class classification จะทำการ initial ค่าก่อนจะมีการเรียกใช้ method
#  บรรทัดที่ 14-18 เป็นการclassify ด้วยวิธี naive bayes โดย import BernoulliNB ของ sklearn เพื่อทำการclassify data  หลังจากนั้นเอา data ที่อยู่ใน x_train, y_train มาเช้า function ที่ชื่อ gnb.fit() คือ เอา data set ( x_train) กับ target (y_train) มา training ก่อนทำการ classify ซึ่งการ classify ใช้ .predict() ในการทำส่วนผลลัพธ์ที่ได้ จะเป็น array string เช่น [''Positive] ,['Nagative'] จึ่งทำการ convert ผลลัพธ์ที่ได้ให้ได้string เมื่อได้ผลลัพธ์ออกมาแล้วทำการหา accuracy ด้วย function .score() เพื่อบอกว่า classify ที่ทำมานั้นมีความเที่ยงตรงแค่ไหน
