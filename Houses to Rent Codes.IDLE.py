import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\python.samira\Projects\houses_to_rent_fanology\houses_to_rent.csv")
df


 df.describe()

 
 df.groupby(['city'])['rent amount (R$)'].mean()
 #in dastoor miad sotoone city ro bar asase sotoone rent amount mide va miangine rent ro dar har city mide

sns.catplot(x = 'animal', y = 'rent amount (R$)',col = 'city',data = df,kind = 'bar')


plt.figure(figsize = (10,10))
sns.heatmap(df.corr(),annot = True)#in dastoor miad ye nemoodare garmaee estelahan mide ke mizane
#hambastegie ficherhaye mokhtalef be ham ro neshoon mide,agar teyfe rang be samte siah pish
#bere mizane hambastegi be samte sefr meil mikone va agar teyfe rang be samte sefid pish
#bere mizane hambastegi be samte 1 ya 100% pish mire,yani kamelan hambaste be ham hastan

df.info()# in dastoor ye etelaate koli az dadehaye ma mide,masalan mige dadehaye null darim
#ya na,ya masalan noe har ficher ro moshakas mikone ke masalan intiger ya float,ya noe digari
#hast,noktei ke peyda mikonim az in tarigh,motevajeh mishim fichere "floor"noe dadehayash string hast
#,object hamoon string hast ke nabayad intori bashe va bayad dadehaye in ficher az noe intiger bashe,
#be dalile "-" ke  meghdare ye seri dadehaye fichere floor hast,string hesab shode,bayad in irad ro raf konim
#ebteda az tarighe dastoore zir hesab mikonim chand ta az dadehaye in ficher in meghdar yani - hastan :

df[df['floor'] == '-']#pas in dastoor kole radifhayee ro az jadvale asli mide ke fichere "floor" dar anha ba
#meghdare "-" por shode ast,2461 radif mide.

df['floor'].unique()#in dastoor dadehaye yeke mojood dar fichere floor ro mide,masalan ye araye mide ke dar
#halate koli in ficher az che dadehayee por shode

df['floor'] = df['floor'].replace('-',0)#in dastoor hame "-" ha ra ba meghdare 0 jaygozin mikonad
df

df[df['floor'] = '-']#yek bare dige az in dastoor estefade mikonim bebinim aya satri vojood dare ke meghdare
#floor barash hamchenan "-" bashe ya dorost amal kardim

df['floor'] = df['floor'].astype(int)#in dastoor noe dade ra baraye ma da sotoone floor ha tabdil be intiger mikone

df.info()#dobare migirim dastoore info ro ta bebinim typha taghir karde ya na

from sklearn.preprocessing import LabelEncoder#KabelEncoder baraye tabdile dadehaye gheire adadi be adadi be kar miravad:
le = LabelEncoder()

df['animal'] = le.fit_transform(df['animal'])#in 3 khat code hameye dadehaye gheir adadi ra tandil be adad mikonad dar jadval
df['city'] = le.fit_transform(df['city'])
df['furniture'] = le.fit_transform(df['furniture'])

#raveshe dige baraye tabdile dadehaye gheire adadi be adadi One Hot Encoding ke khodam test kardam inja:
#in ravesh string ha ro be soorate adade binary mide
#codhaye zir baraye tabdile maghadire string fiture "animal" be adade binary hast:

from sklearn.preprocessing import LabelBinarizer
oneHot = LabelBinarizer()
df['animal'] = oneHot.fit_transform(df['animal'])
df['animal'].unique()# out : array([0, 1]) ,az in tarigh test mikonim ke dastoor dorost amal karde ya kheir,bale dorost
#amal karde va dadeha adadi shodand



#hala dadeha kamelan amade train kardan hastan,az inja be bad dadeha ro taghsim mikonim be test va train,
#va sepas be model yad midim,sepas test mikonim model ro az tarighe dadehaye test:

from sklearn.model_selection import train_test_split

target = df['rent amount (R$)']#in dastoor moshakhas mikone ma mikhaym label hadafemoon chi bashe,be ebarati gharare kodam ficher ra pishbini konim
data = df.drop(columns = ['rent amount (R$)','total (R$)']#dar inja bayad 2 sotoone marboot be ficherhaye rent amount va total
#ke dar vaghe javabe ma hastan ro hazf shvand az df.
data
               
x_train,x_test,y_train,y_test = train_test_split()#dastoore kolie joda sazie dadehaye train va test ke bayad meghdardehi beshe mesle paeen:

# hala dadeha vaghan amade hastan baraye train kardan va model sakhtan
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size = 0.3)
#ba in dastoor darim dadehaye train(70% dadehaye kol)ro + dadehaye target midim be algorythm
#jahate modelsazi

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)#inja darim be model yad midim            
               

predicted = lr.predict(x_test)#in khat miad dadehaye x_test ro migire va predict mikone baramoon
plt.scatter(y_test,predicted)#inja miad baramoon nemoodare moghayesei y_test ha va predict shodeha ro mide

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,predicted))               
#in dastoor miad oonayee ke model predict karde(predicted) va dadehaye vaghei
#ke dashtim khodemoon az ghabl(y_train) ro baramoon moghayese mikone ke cheghad model dorost amak mikone
#dar vaghe deghat model ro be ma mide in dastoor

