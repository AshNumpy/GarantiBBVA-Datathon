### Projenin Amacı
Bu projede GarantiBBVA 'nın çalışanlarına ilişkin veriler kullanılarak. İlgili çalışanların 2019 'da işten ayrılıp ayrılmayacağının tahminini yapmak amaçlanmıştır. 

### Süreç İle İlgili Görüşlerim
Proje süresince en çok vaktimi alan kısım ön işleme kısmı oldu. Zira verilerin hepsi birbirine girmiş durumdayı. Ön işleme kısmında ise beni en çok uğraştıran ayırca modelimin doğruluğunu düşürdüğünü düşündüğüm bir diğer durum ise kategorik sütunların yüzlerce kategorilerden oluşup iyi derecede bir ön işleme gerektirmesiydi.  

bkz: `istanbul, ıstanbul, İSTANBUL, ISTANBUL, İST ANBUL, kadıköy, KADIKOY, Marmara İst. , stnbul, uskudar/ist, etc...`

Ön işleme kısmını bitirdikten sonra veri analizi yapıp makine öğrenmesi ve derin öğrenme yöntemleri ile incelemeye devam ettim.  

Modelleri ile ilgili raporları ise aşağıda belirttim.

### Temel Model Raporu  
Sınıflandırma süresince kullanılan modeller aşağıdaki gibidir:  
* Lojistik Regreson  
* Random Forest Classifier
* Gradient Boosting Trees
* Gaussian Naive Bayes
* Bernoulli Naive Bayes
* Support Vector Machines

Kullanılan tüm bu sınıflandırma modellerinin yanında yapay sinir ağları ile çeşitli mimarilerde derin sinir ağları (DNN) kuruldu. Sonuç olarak Boost süreci başlatılmadan önce aşağıdaki doğruluklar elde edildi:

| Model | Doğruluk Oranı | AOC |
| :-------- | :-------- | :-------- |
| Lojistik Regresyon  | 0.718446 | 0.71 |
| Random Forest  | 0.795624  | 0.84  |
| Gradient Boosting Trees  | 0.795624  | 0.80  |
| Gaussian Naive Bayes  | 0.617201  | 0.62  |
| Bernoulli Naive Bayes  | 0.605583  | 0.62  |
| Support Vector Machines  | 0.744323  | 0.74  |
| Deep Neural Networks  | 0.800151  | 0.90  |

Yukarıdaki tabloda elde edilen sonuçlar modellerin default parametreleri ve hedef değişkenindeki sınıf ağırlıklarının belirtilmesi kullanılarak elde edilmiş sonuçlardır.

Derin sinir ağları kısmında ise pek çok çeşitli mimari denemeleri yapıldı.

<hr>

### Boosting Raporu

Modeller default ayarlarda denendikten sonra makine öğrenmesi modellerinden `scikitlearn.ensemble.RandomForestClassifier()` modeli seçildi.  

Derin sinir ağlarında seçilen modelin mimarisi ise aşağıdaki gibidir:  

```python
model = Sequential()

model.add(Dense(232, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(116, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(58, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(29, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Bu seçilen modeller hyperparameter tuning işlemine tabi tutuldu, ve sonrasında adaboost ve xgboost gibi boost yöntemleri ile boost edildi, sonuçlar:  

| Model | Boost Modeli | Doğruluk |
| :-------- | :-------- | :-------- |
| Random Forest  | AdaBoost | 0.864127 |
| MLPClassifier & Random Forest  | VotingClassifier | 0.82897 |
| Random Forest & XGBoost  | Stacked | 0.853866 |

Sonuç olarak AdaBoost ile boost edilmiş rassal ormanlar algoritmasını kullandım.
