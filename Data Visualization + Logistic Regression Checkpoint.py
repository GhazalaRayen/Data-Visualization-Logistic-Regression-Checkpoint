#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv (r'C:\Users\ghaza\OneDrive\Bureau\gmc\titanic-passengers.csv',sep=";")


# In[21]:


df.head()


# In[22]:


df.head().isnull().sum()


# In[23]:


number_of_elements = len(df["Cabin"])
print("Number of elements:",number_of_elements)
print(df["Cabin"].value_counts())


# In[24]:


df["Cabin"].fillna('G6',inplace=True)


# In[25]:


df.head()


# In[26]:


g=sns.FacetGrid(df,col='Sex')
g.map(plt.hist,'Age')


# In[ ]:




Ces deux histogrammes nous montrent la relation entre le sexe d'une personne et son âge.Nous pouvons remarquer que le nombre de males est largement supérieur au nombre de femmes dans la tranche d'âge entre 20 et 40 ans.
# In[27]:


df[["Survived", "Pclass"]].groupby(["Survived"], as_index=True).mean()


# In[28]:


df['Name'].head()


# In[29]:


df['Name_Title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
df['Name_Title'].value_counts()


# In[30]:


sns.countplot(df['Pclass'], hue=df['Survived'])

On remarque que le plus grands taux de mortalité est chez les personnes qui on un"Pclass" égal à 3. En effet le nombre de morts dans le "Pclass 3" est presque le triple de personnes de la meme class et qui ont survécu.
# In[4]:


import seaborn as sns

Var_Corr = df.corr()
# plot the heatmap and annotation on it
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)

On remarque que la meilleure corrélatoin est entre le "Sibsp" et le "Parch". Même si la valeur de 0.41 reste faible et elle risuqe de nous donner des résultats qui ne sont pas très exacts.
# In[2]:


import pandas as pd

def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

        )


# In[31]:


df = pd.read_csv (r'C:\Users\ghaza\OneDrive\Bureau\gmc\titanic-passengers.csv',sep=";")
df.tail()


# In[32]:


number_of_elements = len(df["SibSp"])
print("Number of elements:",number_of_elements)
print(df["SibSp"].value_counts())


# In[15]:


number_of_elements = len(df["Parch"])
print("Number of elements:",number_of_elements)
print(df["Parch"].value_counts())


# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


g=sns.FacetGrid(df,col='SibSp')
g.map(plt.hist,'Parch')


# In[ ]:


sns.countplot(df['SibSp'], hue=df['Parch'])

 Cet histogramme nous donne tout ce qu'on a besoin de savoir sur la nombre de"Sibsp" dans chaque "Parch".
# In[37]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv (r'C:\Users\ghaza\OneDrive\Bureau\gmc\titanic-passengers.csv',sep=";")


# In[47]:


df = pd.read_csv (r'C:\Users\ghaza\OneDrive\Bureau\gmc\titanic-passengers.csv',sep=";")
df.head()


# In[74]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df = pd.read_csv (r'C:\Users\ghaza\OneDrive\Bureau\gmc\titanic-passengers.csv',sep=";")
x = df[['Age', 'Parch','SibSp']]
y = df['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0) 
logreg = LogisticRegression()   
logreg.fit(x_train, y_train)  
y_pred  = logreg.predict(x_test)   
print("Accuracy={:.2f}".format(logreg.score(x_test, y_test)))


# In[75]:


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)


# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[64]:


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
data_X, class_label = make_classification(n_samples=1000, n_classes=2, weights=[1,1], random_state=1)
    
trainX, testX, trainy, testy = train_test_split(data_X, class_label, test_size=0.3, random_state=1)


# In[70]:



probs = model.predict_proba(testX)
probs = probs[:, 1]


# In[71]:


auc = roc_auc_score(testy, probs)
print('AUC: %.2f' % auc)


# In[72]:


fpr, tpr, thresholds = roc_curve(testy, probs)


# In[73]:


plot_roc_curve(fpr, tpr)


# In[ ]:




