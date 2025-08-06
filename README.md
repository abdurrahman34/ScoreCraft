# Home Credit Default Risk Projesi

## Proje Hakkında
Bu proje, Home Credit Default Risk veri seti üzerinde kredi geri ödeme olasılığını tahmin etmek için çeşitli makine öğrenimi modelleri geliştirmeyi amaçlamaktadır. Proje, veri ön işleme, özellik mühendisliği, sınıf dengesizliği düzeltme ve model eğitimi/değerlendirme adımlarını içermektedir.

## Veri Kaynağı
Projede kullanılan veri seti, Kaggle'da bulunan [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) yarışmasından alınmıştır. Bu veri seti, kredi geçmişi olmayan müşterilerin kredi geri ödeme yeteneklerini tahmin etmek için kullanılmaktadır. Veri seti, müşterilerin demografik bilgileri, kredi başvuru bilgileri, kredi bürosu verileri ve ödeme geçmişi gibi bilgileri içermektedir.

## Dosya Açıklamaları

### 1. Create_New_File.py
Bu dosya, veri setinden ilgili sütunları seçme ve yeni özellikler oluşturma işlemlerini gerçekleştirir. Özellik mühendisliği adımları burada uygulanır ve işlenmiş veri `application_train_with_features.csv` olarak kaydedilir.

### 2. preprocessing.py
Veri temizleme, eksik değer işleme ve aykırı değer tespiti gibi veri ön işleme adımlarını içerir. Ayrıca, veri setinin keşifsel veri analizi (EDA) bu dosyada gerçekleştirilir. İşlem sonunda, PCA tabanlı 2D görselleştirme oluşturulur ve `pca_visualization.png` olarak kaydedilir.

### 3. check_class_balance.py
Hedef değişkenin sınıf dağılımını kontrol eder ve görselleştirir. Sınıf dengesizliği tespit edilirse, SMOTE uygulaması için bir sonraki adıma geçilir.

### 4. apply_smote.py
SMOTE (Synthetic Minority Over-sampling Technique) yöntemi kullanarak veri setindeki sınıf dengesizliğini giderir. Azınlık sınıfı için sentetik örnekler oluşturarak veri setini dengeler. Ayrıca, veri sızıntısını önlemek için ID sütunlarını ("Unnamed: 0", "SK_ID_CURR", "ID" gibi) kaldırır ve dengelenmiş veri setini kaydeder.

### 5. train_logistic_regression.py
Lojistik Regresyon modelinin eğitilmesi, değerlendirilmesi ve görselleştirilmesi işlemlerini gerçekleştirir. Model performans metrikleri (doğruluk, kesinlik, duyarlılık, F1 skoru, ROC AUC) hesaplanır ve özellik önemleri belirlenir. Sonuçlar görsel olarak kaydedilir.

### 6. train_xgboost.py
XGBoost modelinin eğitilmesi, değerlendirilmesi ve görselleştirilmesi işlemlerini gerçekleştirir. Model performans metrikleri hesaplanır ve özellik önemleri belirlenir. Sonuçlar görsel olarak kaydedilir.

### 7. train_random_forest.py
Random Forest modelinin eğitilmesi, değerlendirilmesi ve görselleştirilmesi işlemlerini gerçekleştirir. Model performans metrikleri hesaplanır ve özellik önemleri belirlenir. Sonuçlar görsel olarak kaydedilir.

### 8. model_summary.py
Tüm modellerin performans özetini ve en önemli özelliklerini CSV dosyalarına kaydeder. Her model için doğruluk, kesinlik, duyarlılık, F1 skoru ve ROC AUC değerleri ile en önemli 3 özellik listelenir.

## Kurulum ve Çalıştırma

### Gereksinimler
Projeyi çalıştırmak için aşağıdaki Python paketleri gereklidir:
```
numpy==1.24.3
pandas
scikit-learn
matplotlib
seaborn
xgboost
imbalanced-learn
```

### Çalıştırma Sırası
1. `Create_New_File.py` - Özellik mühendisliği
2. `preprocessing.py` - Veri ön işleme ve EDA
3. `check_class_balance.py` - Sınıf dengesizliği kontrolü
4. `apply_smote.py` - SMOTE uygulaması ile sınıf dengeleme
5. `train_logistic_regression.py` - Lojistik Regresyon modeli eğitimi
6. `train_xgboost.py` - XGBoost modeli eğitimi
7. `train_random_forest.py` - Random Forest modeli eğitimi
8. `model_summary.py` - Model özeti ve özellik önem analizi

## Sonuçlar
Proje sonucunda, kredi geri ödeme olasılığını tahmin etmek için en iyi performansı gösteren model belirlenmiştir. Model performans metrikleri ve özellik önemleri, `model_performance_summary.csv` ve `top_features_summary.csv` dosyalarında kaydedilmiştir. Ayrıca, her model için karmaşıklık matrisi, ROC eğrisi ve özellik önem grafikleri görsel olarak kaydedilmiştir.
