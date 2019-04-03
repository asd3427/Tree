# -*- coding:utf-8 -*-
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
import csv

myData = open(r'data.csv')
reader = csv.reader(myData)
headers=next(reader)
print (headers)
featuelist=[]
labeList=[]

for row in reader:
    labeList.append(row[len(row)-1])
    rowDict={}
    for i in range(0,len(row)-1):
        rowDict[headers[i]]=row[i]

    #Make sure the below line is not inside the second loop
    featuelist.append(rowDict)  #<--This was the typo. 

#print(featuelist)    
vec=DictVectorizer(sparse=False)
dummyX=vec.fit_transform(featuelist)
print('dummyX:'+str(dummyX))
print(vec.get_feature_names())
print('labeList:'+str(labeList) )

lb=preprocessing.LabelBinarizer()
dummyY=lb.fit_transform(labeList)
print('dummyY:'+str(dummyY))

X_train, X_test, y_train, y_test = train_test_split( dummyX, dummyY, test_size =0.3, random_state = 100)

print("ter",X_train)



clf_entropy =DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=5, min_samples_leaf=10)
clf_entropy.fit(X_train, y_train)
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=100, splitter='best')


with open('entropy.txt','w') as f:
    f=tree.export_graphviz(clf_entropy,feature_names=vec.get_feature_names(),out_file=f)


with open ('TTT.csv','w') as f:
    w = csv.writer(f)
    w.writerows(X_test)
    f.close()

y_pred_en = clf_entropy.predict(X_train)
y_pred_en

print("Accuracy for train", accuracy_score(y_train,y_pred_en))

y_pred_en = clf_entropy.predict(X_test)
y_pred_en

print("Accuracy is ", accuracy_score(y_test,y_pred_en))


from sklearn import metrics
## 使用classification_report查看每个类别的正确率，召回率
print "report:"
print metrics.classification_report(y_test, y_pred_en)

## 输出到excel，一种方法是你先输出到csv文件中，然后直接用excel打开即可。
## 按照要求，是要将测试集的预测结果和真实结果输入到文件中，你的输入到TTT.csv的这段代码并没有实现。 
## 应该是这样的结构
## 真实值, 预测值
## acc,  acc
## unacc,  acc
## .....
l_true = lb.inverse_transform(y_test)
l_pred = lb.inverse_transform(y_pred_en)
with open("result.csv", "w") as f:
    cw = csv.writer(f, dialect="excel")
    cw.writerow(["true", "predict"])
    cw.writerows(zip(l_true, l_pred))

## 如果不想使用csv，可以使用python的另外两个库xlrd和xlwt，前者读excel表 ，后者是写excel表
import xlwt
# 创建一个excel表
workbook = xlwt.Workbook(encoding = 'utf-8')
# 创建一个sheet
worksheet = workbook.add_sheet('true_and_pred')
# 往单元格写数据
# header
worksheet.write(0, 0, "true")
worksheet.write(0, 1, "pred")

for i in range(len(l_true)):
    worksheet.write(i + 1, 0, l_true[i])
    worksheet.write(i + 1, 1, l_pred[i])

# 保存
workbook.save("excel.xls")



#### 关于剪枝
##sklearn目前没有具体实现后剪枝的功能。
## 现在能做的是预剪枝，就是设置Classifier或者Regression里的参数max_depth, min_samples_split, min_samples_leaf。
## 后剪枝的确是在sklearn中做不到的。
# 预剪枝就是在构建决策树的时候进行剪枝。通常决策树会不断生长，直到该树枝无法变得更纯（对于ID3决策树来说，就是无法使得熵更小）。我们可以通过设定一个阈值使得决策树提前终止生长。比如设定最小分叉样本数(sklearn RandomForest中的min_samples_split)，当该树杈上的样本小于所设定的min_sample_splt，树就不再继续生长。
# 后剪枝就是在决策树在完全生长完之后，再剪去一些树枝、枝叶。方法也是类似的，我们先设定一个阈值。将某节点与其父节点合并，熵必然会增加，如果这个增加值小于这个阈值，我们最终就保留这个合并。这个过程由树叶逐渐向根部进行。