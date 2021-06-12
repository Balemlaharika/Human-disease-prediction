#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd
import os
from decimal import Decimal


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[3]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[4]:



from tkinter import *
from PIL import ImageTk, Image 


# In[5]:


def register():
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("1000x750")
    register_screen.configure(bg="lightblue")
   
    

    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()
    email=StringVar()
    phone=StringVar()
    
    Label(register_screen, text="PLEASE ENTER DETAILS BELOW",font=("Times", 24),fg="blue",bg="lightblue").pack()
    
    username_lable = Label(register_screen, text="USER NAME ",width="300",bg="lightblue", height="3", font=("Times", 14))
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username,width="25")
    username_entry.pack()
    Label(register_screen, text="",bg="lightblue").pack()
    
    password_lable = Label(register_screen, text="PASSWORD ",bg="lightblue",width="300", height="3", font=("Times", 14))
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password,show= '*',width="25")
    password_entry.pack()
    Label(register_screen, text="",bg="lightblue").pack()
    
    
    email_lable = Label(register_screen, text="EMAIL",width="300",bg="lightblue", height="3", font=("Times", 14))
    email_lable.pack()
    email_entry = Entry(register_screen, textvariable=email,width="25")
    email_entry.pack()
    Label(register_screen, text="",bg="lightblue").pack()
    
    phone_lable = Label(register_screen, text="PHONE NUMBER ",width="300", bg="lightblue",height="3", font=("Times", 14))
    phone_lable.pack()
    phone_entry = Entry(register_screen, textvariable=phone,width="25")
    phone_entry.pack()
    Label(register_screen, text="",bg="lightblue").pack()
    
    

    Button(register_screen, text="REGISTER", width=20, height=2, bg="blue",fg="white",font=("Times", 15), command = register_user).pack()


def login():
    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("Login")
    login_screen.geometry("1000x750")
    
    
    
   
    login_screen.configure(bg="lightblue")
    Label(login_screen, text="PLEASE DETAILS ENTER BELOW",fg="blue", bg="lightblue",font=("Times",15)).pack()
    Label(login_screen, text="",bg="lightblue").pack()

    global username_verify
    global password_verify

    username_verify = StringVar()
    password_verify = StringVar()

    global username_login_entry
    global password_login_entry

    Label(login_screen, text="USER NAME  ",width="300", height="2",bg="lightblue", font=("Times", 13)).pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify,width="25")
    username_login_entry.pack()
    Label(login_screen, text="",bg="lightblue").pack()
    Label(login_screen, text="PASSWORD ",width="300", height="2", bg="lightblue",font=("Times", 13)).pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show= '*',width="25")
    password_login_entry.pack()
    Label(login_screen, text="",bg="lightblue").pack()
    Button(login_screen, text="LOGIN", width=20, height=2, fg="white",bg="blue",font=("Times", 13), command = login_verify).pack()



def register_user():

    username_info = username.get()
    password_info = password.get()

    file = open(username_info, "w")
    file.write(username_info + "\n")
    file.write(password_info)
    file.close()

    username_entry.delete(0, END)
    password_entry.delete(0, END)

    Label(register_screen, text="Registration Success", fg="green", font=("Times", 14)).pack()

def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)

    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            login_sucess()

        else:
            password_not_recognised()

    else:
        user_not_found()



def login_sucess():
    global login_success_screen
    login_success_screen = Toplevel(login_screen)
    login_success_screen.title("Success")
    login_success_screen.geometry("1000x750")
    Label(login_success_screen, text="Login Success",width="30", height="2", font=("Times", 12)).pack()
    Button(login_success_screen, text="OK", command=delete_login_success).pack()
    Label(login_success_screen, text="Exit",width="30", height="2", font=("Times", 12)).pack()
    Button(login_success_screen, text="OK", command=main_screen.destroy).pack()



def password_not_recognised():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("Success")
    password_not_recog_screen.geometry("300x300")
    Label(password_not_recog_screen, text="Invalid Password ").pack()
    Button(password_not_recog_screen, text="OK", command=delete_password_not_recognised).pack()
def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Success")
    user_not_found_screen.geometry("300x300")
    Label(user_not_found_screen, text="User Not Found").pack()
    Button(user_not_found_screen, text="OK", command=delete_user_not_found_screen).pack()

def delete_login_success():
    login_success_screen.destroy()


def delete_password_not_recognised():
    password_not_recog_screen.destroy()


def delete_user_not_found_screen():
    user_not_found_screen.destroy() 

def main_account_screen():
    global main_screen
    main_screen = Tk()
    main_screen.geometry("1000x800")
    
    image=Image.open("bg2.jpg")
    photo=ImageTk.PhotoImage(image)
    lbl=Label(main_screen,image=photo)
    lbl.place(x=0,y=0,relwidth=1,relheight=1)
    
    
    main_screen.title("Account Login")
   
   
    
    
     
    l1= Label(text="HUMAN DISEASE PREDICTION SYSTEM",bg="#8acddd" ,font=("Times", 18)).pack(pady=80,side=TOP,anchor="e",padx=200)
    
    Label(text="SELECT YOUR CHOICE",bg="#8acddd", font=("Times", 17)).pack(pady=15,side=TOP,anchor="se",padx=20)
    Label(text="",bg="#8acddd").pack()
    Button(text="LOGIN", height="3", width="30", font=("Times",13),command = login).pack(side=TOP,anchor="e",padx=15,pady=20)
    Label(text="",bg="#8acddd").pack()
    Button(text="REGISTER", height="3", width="30", font=("Times",13),command=register).pack(side=TOP,anchor="e",padx=15,pady=20)
    


    main_screen.mainloop()


    
main_account_screen()






# In[6]:


l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']


# In[7]:


disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
    'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
    ' Migraine','Cervical spondylosis',
    'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
    'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
    'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
    'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
    'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
    'Impetigo']


# In[8]:


l2=[]
for i in range(0,len(l1)):
    l2.append(0)
print(l2)


# In[9]:


df=pd.read_csv("training.csv")


# In[10]:


df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)


# In[11]:


df.head()


# In[12]:


def plotPerColumnDistribution(df1, nGraphShown, nGraphPerRow):
    nunique = df1.nunique()
    df1 = df1[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df1.shape
    columnNames = list(df1)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()
    


# In[13]:


def plotScatterMatrix(df1, plotSize, textSize):
    df1 = df1.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df1 = df1.dropna('columns')
    df1 = df1[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df1 = df1[columnNames]
    ax = pd.plotting.scatter_matrix(df1, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df1.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
plotPerColumnDistribution(df, 10, 5)
plotScatterMatrix(df, 20, 10)


# In[14]:


X= df[l1]
y = df[["prognosis"]]
np.ravel(y)
print(X)
print(y)


# In[15]:


tr=pd.read_csv("testing.csv")


# In[16]:


tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)


# In[17]:


tr.head()


# In[18]:


plotPerColumnDistribution(tr, 10, 5)


# In[19]:


plotScatterMatrix(tr, 20, 10)


# In[20]:


X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
print(X_test)
print(y_test)


# In[ ]:





# In[21]:


accuracy_list = []
cross_accuracy_list = []
model_list = []


# In[ ]:





# In[22]:


root = Tk()
pred1=StringVar()
def DecisionTree():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif((Symptom1.get()=="Select Here") or (Symptom2.get()=="Select Here")):
        pred1.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        dt= DecisionTreeClassifier()
        dt = dt.fit(X, y)
# prediction of labels for the test data
        scores_dt = cross_val_score(dt, X, y, cv=5)
# mean of cross val score (accuracy)
        score= round(Decimal(scores_dt.mean() * 100), 2)
        cross_accuracy_list.append(score)
        t=print(f"Cross Validation Accuracy (DT): {score}%")
        
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = dt.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        if (h=='yes'):
            pred1.set(" ")
            pred1.set(disease[a])
        else:
            pred1.set(" ")
            pred1.set("Not Found")
        import sqlite3 
        conn = sqlite3.connect('database.db') 
        c = conn.cursor() 
        c.execute("CREATE TABLE IF NOT EXISTS DecisionTree(Name StringVar,Age StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        c.execute("INSERT INTO DecisionTree(Name,Age,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?,?)",(NameEn.get(),AgeEn.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),pred1.get()))
        conn.commit()  
        c.close() 
        conn.close()
       
        
       


# In[23]:


pred2=StringVar() 
def randomforest():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif((Symptom1.get()=="Select Here") or (Symptom2.get()=="Select Here")):
        pred1.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        rf = RandomForestClassifier(n_estimators=10, criterion='entropy')
        rf = rf.fit(X, y)
        scores_rf = cross_val_score(rf, X, y, cv=5)
# mean of cross val score (accuracy)
        score= round(Decimal(scores_rf.mean() * 100), 2)
        cross_accuracy_list.append(score)
        t1=print(f"Cross Validation Accuracy (RF): {score}%")
        

        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = rf.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        if (h=='yes'):
            pred2.set(" ")
            pred2.set(disease[a])
        else:
            pred2.set(" ")
            pred2.set("Not Found")
        import sqlite3 
        conn = sqlite3.connect('database.db') 
        c = conn.cursor() 
        c.execute("CREATE TABLE IF NOT EXISTS RandomForest(Name StringVar,Age StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        c.execute("INSERT INTO RandomForest(Name,Age,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?,?)",(NameEn.get(),AgeEn.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),pred2.get()))
        conn.commit()  
        c.close() 
        conn.close()
        
        


# In[24]:


pred4=StringVar()
def KNN():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif((Symptom1.get()=="Select Here") or (Symptom2.get()=="Select Here")):
        pred1.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=4)
        knn = knn.fit(X, y)
        scores_knn = cross_val_score(knn, X, y, cv=5)
# prediction of labels for the test data
        score = round(Decimal(scores_knn.mean() * 100), 2)
        cross_accuracy_list.append(score)
        print(f"Cross Validation Accuracy (KNN): {score}%")
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = knn.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break


        if (h=='yes'):
            pred4.set(" ")
            pred4.set(disease[a])
        else:
            pred4.set(" ")
            pred4.set("Not Found")
        import sqlite3 
        conn = sqlite3.connect('database.db') 
        c = conn.cursor() 
        c.execute("CREATE TABLE IF NOT EXISTS KNearestNeighbour(Name StringVar,Age StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        c.execute("INSERT INTO KNearestNeighbour(Name,Age,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?,?)",(NameEn.get(),AgeEn.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),pred4.get()))
        conn.commit()  
        c.close() 
        conn.close()
        


# In[25]:


pred3=StringVar()
def NaiveBayes():
    if len(NameEn.get()) == 0:
        pred1.set(" ")
        comp=messagebox.askokcancel("System","Kindly Fill the Name")
        if comp:
            root.mainloop()
    elif((Symptom1.get()=="Select Here") or (Symptom2.get()=="Select Here")):
        pred1.set(" ")
        sym=messagebox.askokcancel("System","Kindly Fill atleast first two Symptoms")
        if sym:
            root.mainloop()
    else:
        mnb = MultinomialNB()
        mnb = mnb.fit(X, y)
# Cross Validation Accuracy MNB
# performing cross validation with 5 different splits
        scores_mnb = cross_val_score(mnb, X, y, cv=5)
# mean of cross val score (accuracy)
        score = round(Decimal(scores_mnb.mean() * 100), 2)
        cross_accuracy_list.append(score)
        print(f"Cross Validation Accuracy (MNB): {score}%")
        psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
        for k in range(0,len(l1)):
            for z in psymptoms:
                if(z==l1[k]):
                    l2[k]=1

        inputtest = [l2]
        predict = mnb.predict(inputtest)
        predicted=predict[0]

        h='no'
        for a in range(0,len(disease)):
            if(predicted == a):
                h='yes'
                break
        if (h=='yes'):
            pred3.set(" ")
            pred3.set(disease[a])
        else:
            pred3.set(" ")
            pred3.set("Not Found")
        import sqlite3 
        conn = sqlite3.connect('database.db') 
        c = conn.cursor() 
        c.execute("CREATE TABLE IF NOT EXISTS NaiveBayes(Name StringVar,Age StringVar,Symtom1 StringVar,Symtom2 StringVar,Symtom3 StringVar,Symtom4 TEXT,Symtom5 TEXT,Disease StringVar)")
        c.execute("INSERT INTO NaiveBayes(Name,Age,Symtom1,Symtom2,Symtom3,Symtom4,Symtom5,Disease) VALUES(?,?,?,?,?,?,?,?)",(NameEn.get(),AgeEn.get(),Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get(),pred3.get()))
        conn.commit()  
        c.close() 
        conn.close()
       


# In[ ]:



    


# In[26]:


root.configure(background='lightblue')
root.title('Smart Disease Predictor System')


# In[27]:


Symptom1 = StringVar()
Symptom1.set("Select Here")
Symptom2 = StringVar()
Symptom2.set("Select Here")
Symptom3 = StringVar()
Symptom3.set("Select Here")
Symptom4 = StringVar()
Symptom4.set("Select Here")
Symptom5 = StringVar()
Symptom5.set("Select Here")
Name = StringVar()
Age=StringVar()


# In[28]:


prev_win=None
def Reset():
    global prev_win

    Symptom1.set("Select Here")
    Symptom2.set("Select Here")
    Symptom3.set("Select Here")
    Symptom4.set("Select Here")
    Symptom5.set("Select Here")
    
    NameEn.delete(first=0,last=100)
    
    pred1.set(" ")
    pred2.set(" ")
    pred3.set(" ")
    pred4.set(" ")
    try:
        prev_win.destroy()
        prev_win=None
    except AttributeError:
        pass


# In[29]:


from tkinter import messagebox
def Exit():
    qExit=messagebox.askyesno("System","Do you want to exit the system")
    if qExit:
        root.destroy()
        exit()


# In[30]:


w2 = Label(root, justify=LEFT, text="HUMAN DISEASE PREDICTION", fg="blue", bg="lightblue")
w2.config(font=("Times",30,"bold italic"))
w2.grid(row=1, column=0, columnspan=2, padx=100)


# In[31]:


NameLb = Label(root, text="Name of the Patient", fg="black", bg="lightblue")
NameLb.config(font=("Times",15,"bold italic"))
NameLb.grid(row=3, column=0, pady=15, sticky=W)
AgeLb = Label(root, text="Age", fg="black", bg="lightblue")
AgeLb.config(font=("Times",15,"bold italic"))
AgeLb.grid(row=4, column=0, pady=15, sticky=W)


# In[32]:


S1Lb = Label(root, text="Symptom 1", fg="Black", bg="lightblue")
S1Lb.config(font=("Times",15,"bold italic"))
S1Lb.grid(row=5, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="Black", bg="lightblue")
S2Lb.config(font=("Times",15,"bold italic"))
S2Lb.grid(row=6, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="Black",bg="lightblue")
S3Lb.config(font=("Times",15,"bold italic"))
S3Lb.grid(row=7, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="Black", bg="lightblue")
S4Lb.config(font=("Times",15,"bold italic"))
S4Lb.grid(row=8, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="Black", bg="lightblue")
S5Lb.config(font=("Times",15,"bold italic"))
S5Lb.grid(row=9, column=0, pady=10, sticky=W)


# In[33]:


NameEn = Entry(root,textvariable=Name)
NameEn.grid(row=3, column=1)
AgeEn = Entry(root,textvariable=Age)
AgeEn.grid(row=4, column=1)


# In[34]:


OPTIONS = sorted(l1)
S1 = OptionMenu(root, Symptom1,*OPTIONS)
S1.grid(row=5, column=1)

S2 = OptionMenu(root, Symptom2,*OPTIONS)
S2.grid(row=6, column=1)

S3 = OptionMenu(root, Symptom3,*OPTIONS)
S3.grid(row=7, column=1)

S4 = OptionMenu(root, Symptom4,*OPTIONS)
S4.grid(row=8, column=1)

S5 = OptionMenu(root, Symptom5,*OPTIONS)
S5.grid(row=9, column=1)


# In[35]:


dst = Button(root, text="Prediction",command=lambda:[DecisionTree(),randomforest(),NaiveBayes(),KNN()], bg="blue",fg="lightblue")
dst.config(font=("Times",15,"bold italic"))
dst.grid(row=5, column=3,padx=10)





ex = Button(root,text="Exit System", command=Exit,bg="blue",fg="lightblue",width=15)
ex.config(font=("Times",15,"bold italic"))
ex.grid(row=7,column=3,padx=10)


# In[ ]:





# In[36]:


t2=Label(root,font=("Times",15,"bold italic"),text="Result",height=1,bg="Purple"
         ,width=40,fg="white",textvariable=pred2,relief="sunken").grid(row=13, column=1, padx=10)


# In[37]:



root.mainloop()


# In[ ]:




