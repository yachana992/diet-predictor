import pandas as pd
import numpy as np
import re
import Recommenders as Recommenders
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import sklearn.feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
import scikitplot as skplt
import matplotlib.pyplot as plt



# We're going to need regular expressions now;
# import the re module
df = pd.read_excel('C:/Users/User/Desktop/webapp/data/khana.xlsx')
c=0
# Convert prime-notation height to inches
def inches(prime_str):
    ## Wrap the function in try/except
    try: 
        result = re.match(r"([0-9]*)('|' |\.| feet | ft | Feet |feet |ft |Feet )([0-9]{0,2}?)([\"]| inches | inch|inch|inches)?\Z", prime_str)
         # Access the extract values using the .group() method
        feet = int(result.group(1))
        if result.group(3) == "":
            inches = 0
        else:
            inches = int(result.group(3))
    except:
        # This code will only be called if the code within the try: block
        # throws an exception
        return None

    # We won't get to this point unless the code in the try block was successful
        
    return (feet * 12 + inches)*0.0254


df["What is your height? (in feet) "] = df["What is your height? (in feet) "].apply(inches)
mean_h=df["What is your height? (in feet) "].mean(axis=0, skipna=True)
df["What is your height? (in feet) "].fillna(value=mean_h, inplace=True)

mean_w=df['How much do you weigh? (in kgs) '].mean(axis=0, skipna=True)
df['How much do you weigh? (in kgs) '].fillna(value=mean_w, inplace=True)
#print(df['How much do you weigh? (in kgs) '])

bmi = df['How much do you weigh? (in kgs) ']/(df["What is your height? (in feet) "]**2)
#print(bmi)
mean_a=df['What is your age? '].mean(axis=0, skipna=True)
df['What is your age? '].fillna(value=mean_a, inplace=True)
age = df['What is your age? ']
#print(age)
fat_per =(1.20 * bmi) + (0.23 * age) 

fat_per = np.where(df['What is your gender']=='Female', (fat_per-5.4)/100, (fat_per-16.2)/100)
#print(fat_per)

bmr = 370 + 21.6*(1 - fat_per)*df['How much do you weigh? (in kgs) ']
#print(bmr)

#Calculating the calorie counts of breakfast, lunch, khaja and dinner

def breakfast(foods):
    c=0
    x=(str(foods)).split(',')
    for food in x:
        khane = food.lstrip()
        def switch(khane):
            s=0
            switcher={
            'Cereals':348,
            'Bread':265,
            'Tea/coffee':16,
            'Roti':200,
            'Milk':146,
            'Juice':120,
            'Eggs':143,
            }
            return int(switcher.get(khane,0))
        c=c+switch(khane)
    return c


def lunch(khana):
    switcher={
    'Veg Khana(dal, bhaat, tarkari)':392,
    'Non-Veg Khana( Khana + meat/fish/egg)':585,
    'Roti Tarkari':230,
    }
    return int(switcher.get(khana,0))

def khaja(bhok):
    switcher={
    'Chuira/Bhuja + Curry/Curd':100,
    'Fast Food(Chowmein, momo, burger, pizza, sandwiches )':500,
    'Roti/Haluwa':250,
    'Baked Items':217,
    }
    return int(switcher.get(bhok,0))

def water(ltre):
    switcher={
    'more than 2 litres':100,
    '700 ml-1200 ml':45,
    '200 ml-600ml':20,
    '1300 ml-2000 ml':83,
    }
    return int(switcher.get(ltre,0))

def exercise(activity):
    switcher={
    'Hard exercise (6-7 days/week)':500,
    'Light exercise (1-3 days/week)':250,
    'Little or no exercise':100,
    'Moderate exercise (3-5 days/week)':400,
    }
    return int(switcher.get(activity,0))

df['cal_con'] = df['What do you usually have for breakfast?'].apply(breakfast)+df['What do you usually have for lunch?'].apply(lunch)+df['What do you usually have for Khaja?'].apply(khaja)+df['What do you usually take for Dinner?'].apply(lunch)-df['How often do you involve yourself in physical activities ?'].apply(exercise)-df['How much is your daily water intake?'].apply(water)
df['status']=np.where(df['cal_con']>bmr, 'unhealthy', 'healthy')
df['status'] = [0 if x=='unhealthy' else 1 for x in df['status']]

x=df.drop(['status','Name','Timestamp'],1)
np.std(x, axis=0) == 0
y=df['status']
x = pd.get_dummies(x,columns=['What is your gender', 'What do you prefer?', 'How much is your daily water intake?', 'Do you take your foods on regular intervals?', 'How many hours do you sleep?','How often do you eat vegetables?','How often do you eat fruits? ', 'What do you usually have for breakfast?','What do you usually have for lunch?','What do you usually have for Khaja?', 'What do you usually take for Dinner?','Do you smoke? ','How often do you consume beverages? ','Do you eat eggs?','How often do you eat meat/fish?','How often do you involve yourself in physical activities ?','Do you have any of these?','What is your weekly food intake frequency on sweet foods?','What is your weekly food intake frequency on salty foods?','What is your weekly food intake frequency on fresh fruits?', 'What is your goal? '])
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(x)
x=pd.DataFrame(data=imp.transform(x), columns=x.columns)

#deleting outliers for weight
subset_outlier=x.iloc[:,:]
q75, q25 =np.percentile(subset_outlier["How much do you weigh? (in kgs) "],[75,25])
lower =q25-1.5*(q75-q25)
upper=q75 +1.5*(q75-q25)
subset_outlier["How much do you weigh? (in kgs) "][subset_outlier["How much do you weigh? (in kgs) "] > upper]=upper
subset_outlier["How much do you weigh? (in kgs) "][subset_outlier["How much do you weigh? (in kgs) "] < lower]=lower
x['How much do you weigh? (in kgs) ']=subset_outlier['How much do you weigh? (in kgs) ']
#print(x['How much do you weigh? (in kgs) '])

#deleting outliers for height
dff = x['What is your height? (in feet) ']
y2=pd.to_numeric(dff, errors='coerce')
#If ‘coerce’, then invalid parsing will be set as NaN
subset_outlier["What is your height? (in feet) "]=y2
q3, q1 = np.percentile(subset_outlier["What is your height? (in feet) "],[75,25])
lower1 =q1-1.5*(q3-q1)
upper1=q3 +1.5*(q3-q1)
subset_outlier["What is your height? (in feet) "][subset_outlier["What is your height? (in feet) "] > upper1]=upper1
subset_outlier["What is your height? (in feet) "][subset_outlier["What is your height? (in feet) "] < lower1]=lower1
x['What is your height? (in feet) ']= subset_outlier['What is your height? (in feet) ']
#print(x['What is your height? (in feet) '])

dfff = x['What is your age? ']
y3=pd.to_numeric(dfff, errors='coerce')
#If ‘coerce’, then invalid parsing will be set as NaN
subset_outlier["What is your age? "]=y3
q33, q11 = np.percentile(subset_outlier["What is your age? "],[75,25])
lower11 =q11-1.5*(q33-q11)
upper11=q33 +1.5*(q33-q11)
subset_outlier["What is your age? "][subset_outlier["What is your age? "] > upper11]=upper11
subset_outlier["What is your age? "][subset_outlier["What is your age? "] < lower11]=lower11
x['What is your age? ']= subset_outlier['What is your age? ']

df1 = x['How much do you weigh? (in kgs) ']
xx=np.array(df1)
df2 = x['What is your height? (in feet) ']
yy=np.array(df2)
df3 = x['What is your age? ']
z=np.array(df3)

p=np.nan_to_num(xx)
#converting nan to a number and storing it as an array p
q=np.nan_to_num(yy)
r=np.nan_to_num(z)
#replacing the old column with an array
subset_outlier["How much do you weigh? (in kgs) "]=p
subset_outlier["What is your height? (in feet) "]=r
subset_outlier["What is your age? "]=q

data_scale=scale(x[['How much do you weigh? (in kgs) ','What is your height? (in feet) ','What is your age? ']])
subset_outlier[['How much do you weigh? (in kgs) ','What is your height? (in feet) ','What is your age? ']]=data_scale

#Splitting training data and testing data using sklearn 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.70, random_state=1)
#print(df.shape)
#print(x.shape)

#using feature selection to select the most important features
#select = sklearn.feature_selection.SelectKBest(k=20)
#selected_features = select.fit(x_train, y_train)
#indices_selected = selected_features.get_support(indices=True)
#colnames_selected = [x.columns[i] for i in indices_selected]
#print(y_test)
#x_train_selected = x_train[colnames_selected]
#x_test_selected = x_test[colnames_selected]
#model = LogisticRegression()
#lab_enc = preprocessing.LabelEncoder()
#y_train = lab_enc.fit_transform(y_train)
#model.fit(x_train_selected,y_train)
#print(model.intercept_)
#print(model.coef_)


#taking model as a parameter to calculate score for each of the models
#Using KFold cross validation
"""kf = KFold(n_splits = 3)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

def get_score(model, x_train,x_test,y_train, y_test):
    model.fit(x_train,y_train)
    return model.score(x_test, y_test)

get_score(LogisticRegression(),x_train,x_test, y_train, y_test)
folds = StratifiedKFold(n_splits=3)

scores_l = []
scores_svm = []
scores_rf = []

for train_index, test_index in kf.split(x):
    x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]"""




"""model = LogisticRegression()
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
model.fit(x_train,y_train)
y_result = model.predict(x_train_selected)
y_hat = [x[1] for x in model.predict_proba(x_test_selected)]
auc = roc_auc_score(y_test, y_hat)

print(metrics.mean_absolute_error(y_test,y_hat))
print(metrics.mean_squared_error(y_test,y_hat))
print(np.sqrt(metrics.mean_squared_error(y_test,y_hat))) 
#folds = StratifiedKFold(n_splits)"""

folds = StratifiedKFold(n_splits=3)
scoring = ['f1','roc_auc' , 'mean_squared_error']
scoringg = ['neg_log_loss']
print(cross_val_score(LogisticRegression(),x,y))
print(cross_validate(LogisticRegression(),x,y,scoring = scoring))
print(cross_validate(LogisticRegression(),x,y,scoring = scoringg))
y_probas = cross_val_predict(LogisticRegression(), x, y,cv=folds, method='predict_proba')

#F1 = (2*(pres*recc)/(pres+recc))
#auc = roc_auc_score()
#print(cross_val_score(SVC(),x,y))
#print(cross_val_score(RandomForestClassifier(),x,y))


#User's input:
print("Enter your weight in kgs")
weight = float(input())
print("Enter your age")
age = float(input())
print("Enter your height in feet")
height = float(input())
print("Enter your sex(Male/Female")
sex = input()
print("How often do you eat vegetables(Often/Sometimes/Never")
vegetable = input()
print("How often do you eat fruits?(Often/Sometimes/Seldom/Never)")
fruits = input()
print("How often do you consume beverages?(Often/Sometimes/Seldom/Never")
beverages = input()
print("Do you have any of these?(1.Gastritis 2.High blood pressure 3.Low blood pressure 4.Cholesterol 5.None)")
dis = input()

switcher={
    '1': 'Gastritis',
    '2': 'High blood pressure',
    '3': 'Low blood pressure',
    '4': 'Cholesterol',
    '5': 'None',
}
disease = switcher.get(dis,None)
print("What do you usually have for breakfast?(1.Bread 2.Tea/Coffee 3.Milk 4.Juice 5.Fruits 6.Cereals 7.Roti)")
breakss = input()
switcher={
    '1': 'Bread',
    '2': 'Tea/Coffee',
    '3': 'Milk',
    '4': 'Juice',
    '5': 'Fruits',
    '6': 'Cereals',
    '7': 'Roti',
}
breaks = switcher.get(breakss,None)
print("How many hours do you sleep?(1. More than 9 hrs 2. 7-9 hrs 3. less than 7 hrs) ")
sleeps = input()
switcher = {
    '1': 'More than 9 hrs ',
    '2': '7-9 hrs',
    '3': 'less than 7 hrs',
}
sleeep = switcher.get(sleeps,None)
print("What is your daily intake on salty foods?(1.Once a day 2.Several times a day 3.Less often)")
saltt = input()
switcher = {
    '1': 'Once a day',
    '2': 'Several times a day',
    '3': 'Less often',
}
sallt = switcher.get(saltt, None)

print("Do you take your foods on regular intervals?(Yes/No")
interval= input()
print("What do you usually have for lunch?(1.Veg Khana(dal, bhaat, tarkari) 2.Non-Veg Khana( Khana + meat/fish/egg) 3.Roti Tarkari 4.Dhido)")
lun = input()
switcher={
    '1': 'Veg Khana(dal, bhaat, tarkari)',
    '2': 'Non-Veg Khana( Khana + meat/fish/egg)',
    '3': 'Roti Tarkari',
    '4': 'Dhido',
}
lunchh = switcher.get(lun)
print("What do you usually have for Khaja?(1.Fast Food(Chowmein, momo, burger, pizza, sandwiches ) 2.Chuira/Bhuja + Curry/Curd 3.Roti/Haluwa 4.Baked Item")
khaj = input()
switcher={
    '1.': 'Fast Food(Chowmein, momo, burger, pizza, sandwiches )',
    '2' : 'Chuira/Bhuja + Curry/Curd ',
    '3' : 'Roti/Haluwa',
    '4' : 'Baked Item',
}
khajja = switcher.get(khaj)
print("What do you usually take for Dinner?(1.Veg Khana(dal, bhat, tarkari, achar) 2.Non-Veg Khana(Khana + meat/fish/egg) 3.Roti tarkari 4.Dhido")
din = input()
switcher={
    '1': 'Veg Khana(dal, bhaat, tarkari)',
    '2': 'Non-Veg Khana( Khana + meat/fish/egg)',
    '3': 'Roti Tarkari',
    '4': 'Dhido',
}
dinnerr = switcher.get(din)
print("How often do you involve yourself in physical activities?(1.Light exercise (1-3 days/week) 2.Moderate exercise (3-5 days/week) 3.Hard exercise (6-7 days/week) 4.Little or no exercise")
exer = input()
switcher={
    '1': 'Light exercise (1-3 days/week)',
    '2': 'Moderate exercise (3-5 days/week)',
    '3': 'Hard exercise (6-7 days/week)',
    '4': 'Little or no exercise',
}
exercisee = switcher.get(exer)
print("How much is your daily water intake?(1.200 ml-600ml 2.700 ml-1200 ml 3.1300 ml-2000 ml 4.more than 2 litres )")
wat = input()
switcher={
    '1.': '200 ml-600ml',
    '2.': '700 ml-1200 ml',
    '3' : '1300 ml-2000',
    '4' : 'more than 2 litres',
}
waterr = switcher.get(wat)
print("How often do you eat meat/fish?(Often/Seldom/Never")
meat = input()


def breakfastt(foods):
    switcher = {
        'Cereals': 348,
        'Bread': 265,
        'Tea/coffee': 16,
        'Roti': 200,
        'Milk': 146,
        'Juice': 120,
        'Eggs': 143,
        }
    return float(switcher.get(foods, 0))



breakk = breakfastt(breaks)
luunch = lunch(lunchh)
khajaa = khaja(khajja)
dineer = lunch(dinnerr)
exercisse = exercise(exercisee)
wateer = water(waterr)
cal = breakk + luunch + khajaa + dineer + exercisse + wateer

if (sex.lower() == 'female'):
    f = 1
else:
    f = 0
if (sex.lower() == 'male'):
    m = 1
else:
    m = 0
if (water == '700 ml-1200 ml'):
    w = 1
else:
    w = 0
if (interval == 'No'):
    i = 1
else:
    i = 0
if (vegetable.lower() == 'Often'):
     v = 1
else:
    v = 0
if (breaks == 'Eggs'):
    b = 1
else:
    b = 0
if (breaks == 'Tea/coffee'):
    br = 1
else:
    br = 0
if (breaks == 'Fruits'):
    bbr = 1
else:
    bbr = 0
if (beverages.lower() == 'Never'):
    be = 1
else:
    be = 0
if (beverages.lower() == 'Sometimes'):
    bee = 1
else:
    bee = 0
if (meat.lower() == 'Seldom'):
    me = 1
else:
    me = 0
if (disease == 'Gastritis'):
    di = 1
else:
    di = 0
if (fruits.lower() == 'Once a day'):
    fr = 1
else:
    fr = 0
if (fruits.lower() == 'Several times a day'):
    fru = 1
else:
    fru = 0
if (breaks == 'Milk'):
    bf = 1
else:
    bf = 0
if (din == '2'):
    dd = 1
else:
    dd = 0
if(sleeep == '1'):
    sl = 1
else:
    sl = 0
if(meat.lower() == 'often'):
    fish = 1
else:
    fish = 0
if(dis == '2'):
    diss = 1
else:
    diss = 0
if(saltt == '1'):
    sal = 1
else:
    sal = 0


#context = {'result': users[1]}
#y = 1.16271392 + (2.35968961e-01) * weight + (4.41876254e-01) * height + (2.30573554e-01) * age + (5.30783461e-06) * cal + (5.54542353e-02) * f + (1.10725968e+00) * m + (-2.15605974e-01) * w + (7.57931996e-01) * i + (4.18164744e-01) * v + (-5.54265256e-01) * b + (-3.68757112e-01) * br + (-5.31546265e-01) * bbr +(-4.59515845e-01) * bf + (9.98961948e-01) * br + (5.75191395e-01) * be + (-2.93491173e-01) * bee +(-2.37550243e-01) * me + (-2.00067220e-01) * di + (7.71229106e-01) * fr + (-4.10816054e-01) * fru

y=0.7
if(y>0.5):
    bmii = weight/height**2
    fat_pert =(1.20 * bmii) + (0.23 * age) 
    fat_pert = np.where(sex.lower()=='female', (fat_pert-5.4)/100, (fat_pert-16.2)/100)
    bmrr = 370 + 21.6*(1 - fat_pert)*weight


    x1 = df.drop(['status','Name','Timestamp','What do you usually have for lunch?'],1)
    np.std(x1, axis=0) == 0
    x1 = pd.get_dummies(x1,columns=['What is your gender', 'What do you prefer?', 'How much is your daily water intake?', 'Do you take your foods on regular intervals?', 'How many hours do you sleep?', 'How often do you eat vegetables?', 'How often do you eat fruits? ', 'What do you usually have for breakfast?', 'What do you usually have for Khaja?', 'What do you usually take for Dinner?', 'Do you smoke? ', 'How often do you consume beverages? ', 'Do you eat eggs?', 'How often do you eat meat/fish?', 'How often do you involve yourself in physical activities ?', 'Do you have any of these?','What is your weekly food intake frequency on sweet foods?', 'What is your weekly food intake frequency on salty foods?', 'What is your weekly food intake frequency on fresh fruits?', 'What is your goal? '])
    imp1 = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp1.fit(x1)
    x1=pd.DataFrame(data=imp1.transform(x1), columns=x1.columns)

    df['What do you usually have for lunch?'] = [0 if x=='Veg Khana(dal, bhaat, tarkari)' else 1 for x in df['What do you usually have for lunch?']]
    y1 = df['What do you usually have for lunch?']

    x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, train_size=0.70, random_state=1)

    select1 = sklearn.feature_selection.SelectKBest(k=10)
    selected_features1 = select1.fit(x1_train, y1_train)
    indices_selected1 = selected_features1.get_support(indices=True)
    colnames_selected1 = [x1.columns[i] for i in indices_selected1]
    #print(y1_test)
    x1_train_selected = x1_train[colnames_selected1]
    x1_test_selected = x1_test[colnames_selected1]
    model1 = LogisticRegression()
    lab_enc1 = preprocessing.LabelEncoder()
    y1_train = lab_enc1.fit_transform(y1_train)
    model1.fit(x1_train_selected,y1_train)
    #print(model.intercept_)
    #print(model.coef_)
    y1_result = model1.predict(x1_train_selected)
    print(y1_result)
    print(colnames_selected1)
    print(model1.intercept_)
    print(model1.coef_)


    y1_eq = -0.90415237 + -0.03240596*weight + 0.0026198*cal + -0.9832399*f + 0.07908753*m + 0.87149338*sl  + -1.18706681*v + 0.86601081*dd +  0.73805572*fish +  0.4560349*diss +  0.75412865*sal 
    if (y1_eq < 1):
        print('Lunch for you is:')
        print("Veg Khana(dal, bhaat, tarkari)")
    else:
        print('Lunch for you is:')
        print("Non-Veg( Khana + meat/fish/egg)")





    
else:
    print("Thank god you're healthy")

#print(y_probas)
#skplt.metrics.plot_roc_curve(y, y_probas)
#plt.show()





