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
## ʹ��classification_report�鿴ÿ��������ȷ�ʣ��ٻ���
print "report:"
print metrics.classification_report(y_test, y_pred_en)

## �����excel��һ�ַ��������������csv�ļ��У�Ȼ��ֱ����excel�򿪼��ɡ�
## ����Ҫ����Ҫ�����Լ���Ԥ��������ʵ������뵽�ļ��У�������뵽TTT.csv����δ��벢û��ʵ�֡� 
## Ӧ���������Ľṹ
## ��ʵֵ, Ԥ��ֵ
## acc,  acc
## unacc,  acc
## .....
l_true = lb.inverse_transform(y_test)
l_pred = lb.inverse_transform(y_pred_en)
with open("result.csv", "w") as f:
    cw = csv.writer(f, dialect="excel")
    cw.writerow(["true", "predict"])
    cw.writerows(zip(l_true, l_pred))

## �������ʹ��csv������ʹ��python������������xlrd��xlwt��ǰ�߶�excel�� ��������дexcel��
import xlwt
# ����һ��excel��
workbook = xlwt.Workbook(encoding = 'utf-8')
# ����һ��sheet
worksheet = workbook.add_sheet('true_and_pred')
# ����Ԫ��д����
# header
worksheet.write(0, 0, "true")
worksheet.write(0, 1, "pred")

for i in range(len(l_true)):
    worksheet.write(i + 1, 0, l_true[i])
    worksheet.write(i + 1, 1, l_pred[i])

# ����
workbook.save("excel.xls")



#### ���ڼ�֦
##sklearnĿǰû�о���ʵ�ֺ��֦�Ĺ��ܡ�
## ������������Ԥ��֦����������Classifier����Regression��Ĳ���max_depth, min_samples_split, min_samples_leaf��
## ���֦��ȷ����sklearn���������ġ�
# Ԥ��֦�����ڹ�����������ʱ����м�֦��ͨ���������᲻��������ֱ������֦�޷���ø���������ID3��������˵�������޷�ʹ���ظ�С�������ǿ���ͨ���趨һ����ֵʹ�þ�������ǰ��ֹ�����������趨��С�ֲ�������(sklearn RandomForest�е�min_samples_split)����������ϵ�����С�����趨��min_sample_splt�����Ͳ��ټ���������
# ���֦�����ھ���������ȫ������֮���ټ�ȥһЩ��֦��֦Ҷ������Ҳ�����Ƶģ��������趨һ����ֵ����ĳ�ڵ����丸�ڵ�ϲ����ر�Ȼ�����ӣ�����������ֵС�������ֵ���������վͱ�������ϲ��������������Ҷ����������С�