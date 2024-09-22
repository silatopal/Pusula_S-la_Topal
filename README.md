# Pusula_Sila_Topal

Sıla Topal silatopall7@gmail.com 

##Projenin içeriği 

import pandas as pd
df = pd.read_csv('side_effect_data1.csv')

print(df.head())

import matplotlib.pyplot as plt
import piplite
await piplite.install('seaborn')
import seaborn as sns
import numpy as np

print(df.shape)

print(df.dtypes)

print(df.info())

print(df.isnull().sum())

print(df.describe())

df.hist(figsize=(10, 8))
plt.show()

print(df.Ilac_Adi )

print(df.Yan_Etki)

unique_ilac_adlari = df['Ilac_Adi'].drop_duplicates()

print(unique_ilac_adlari)

unique_yan_etki_adlari = df['Yan_Etki'].drop_duplicates()

print(unique_yan_etki_adlari)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Ilac_Adi', y='Yan_Etki', data=df)
plt.title('ilaç ve yan etki  Dağılım Grafiği')
plt.xlabel('Ilac_Adi')
plt.ylabel('Yan_Etki')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x='Alerjilerim', y='Yan_Etki', data=df)
plt.title('Alerji ve Yan Etki  Dağılım Grafiği')
plt.xlabel('Alerjilerim')
plt.ylabel('Yan_Etki')
plt.show()

en_cok_kullanilan_ilac = df['Ilac_Adi'].value_counts()

print(en_cok_kullanilan_ilac)

yan_etki_sikliklari = df['Yan_Etki'].value_counts()

print(yan_etki_sikliklari)

plt.figure(figsize=(12, 8))
sns.barplot(x=yan_etki_sikliklari.index, y=yan_etki_sikliklari.values)
plt.xticks(rotation=90) 
plt.title('Yan Etki Sıklıkları')
plt.xlabel('Yan Etki Adları')
plt.ylabel('Sıklık')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(x=en_cok_kullanilan_ilac.index[:10], y=en_cok_kullanilan_ilac.values[:10])  # İlk 10 ilacı göstermek
plt.xticks(rotation=90)
plt.title('En Çok Kullanılan İlaçlar')
plt.xlabel('İlaç Adları')
plt.ylabel('Kullanılma Sıklığı')
plt.show()

missing_values = df.isnull().sum()
print("Eksik değerlerin sayısı:\n", missing_values)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df['Kilo'] = imputer.fit_transform(df[['Kilo']])

imputer = SimpleImputer(strategy='most_frequent')
df.to_csv('side_effect_data1_imputed.csv', index=False)

print(df.head()) 

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df['Ilac_Adi_encoded'] = label_encoder.fit_transform(df['Ilac_Adi'])
df['Yan_Etki_encoded'] = label_encoder.fit_transform(df['Yan_Etki'])
df['Alerjilerim_encoded'] = label_encoder.fit_transform(df['Alerjilerim'])

print(df)

