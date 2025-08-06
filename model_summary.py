import pandas as pd
import numpy as np
import os
import sys
import importlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Model performans sonuçlarını ve özellik önemlerini yükle
def load_model_results(model_name):
    # Model performans sonuçlarını yükle
    results = {}
    
    try:
        # Her model için performans metriklerini manuel olarak hesapla
        if model_name == "Logistic Regression":
            # Logistic Regression modelini import et
            sys.path.append(os.getcwd())
            lr_module = importlib.import_module('train_logistic_regression')
            
            # Performans metriklerini al
            results = {
                "Model": model_name,
                "Accuracy": lr_module.accuracy_score(lr_module.y_test, lr_module.y_pred),
                "Precision": lr_module.precision_score(lr_module.y_test, lr_module.y_pred),
                "Recall": lr_module.recall_score(lr_module.y_test, lr_module.y_pred),
                "F1 Score": lr_module.f1_score(lr_module.y_test, lr_module.y_pred),
                "ROC AUC": lr_module.roc_auc_score(lr_module.y_test, lr_module.y_pred_proba)
            }
            
            # Özellik önemlerini al
            feature_importance = pd.DataFrame({
                'Feature': lr_module.X.columns,
                'Importance': np.abs(lr_module.best_model.coef_[0])  # Mutlak değeri al
            })
            
            # Özellik önemlerini normalize et (0-1 arası)
            if len(feature_importance) > 0:
                max_importance = feature_importance['Importance'].max()
                if max_importance > 0:  # Sıfıra bölmeyi önle
                    feature_importance['Importance'] = feature_importance['Importance'] / max_importance
                    
            # 3 ondalık basamağa yuvarla
            feature_importance['Importance'] = feature_importance['Importance'].round(3)
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
        elif model_name == "XGBoost":
            # XGBoost modelini import et
            sys.path.append(os.getcwd())
            xgb_module = importlib.import_module('train_xgboost')
            
            # Performans metriklerini al
            results = {
                "Model": model_name,
                "Accuracy": xgb_module.accuracy_score(xgb_module.y_test, xgb_module.y_pred),
                "Precision": xgb_module.precision_score(xgb_module.y_test, xgb_module.y_pred),
                "Recall": xgb_module.recall_score(xgb_module.y_test, xgb_module.y_pred),
                "F1 Score": xgb_module.f1_score(xgb_module.y_test, xgb_module.y_pred),
                "ROC AUC": xgb_module.roc_auc_score(xgb_module.y_test, xgb_module.y_pred_proba)
            }
            
            # Özellik önemlerini al
            feature_importance = pd.DataFrame({
                'Feature': xgb_module.X.columns,
                'Importance': xgb_module.best_model.feature_importances_
            })
            # 3 ondalık basamağa yuvarla
            feature_importance['Importance'] = feature_importance['Importance'].round(3)
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
        elif model_name == "Random Forest":
            # Random Forest modelini import et
            sys.path.append(os.getcwd())
            rf_module = importlib.import_module('train_random_forest')
            
            # Performans metriklerini al
            results = {
                "Model": model_name,
                "Accuracy": rf_module.accuracy_score(rf_module.y_test, rf_module.y_pred),
                "Precision": rf_module.precision_score(rf_module.y_test, rf_module.y_pred),
                "Recall": rf_module.recall_score(rf_module.y_test, rf_module.y_pred),
                "F1 Score": rf_module.f1_score(rf_module.y_test, rf_module.y_pred),
                "ROC AUC": rf_module.roc_auc_score(rf_module.y_test, rf_module.y_pred_proba)
            }
            
            # Özellik önemlerini al
            feature_importance = pd.DataFrame({
                'Feature': rf_module.X.columns,
                'Importance': rf_module.best_model.feature_importances_
            })
            # 3 ondalık basamağa yuvarla
            feature_importance['Importance'] = feature_importance['Importance'].round(3)
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        return results, feature_importance
    
    except Exception as e:
        print(f"{model_name} modeli için sonuçlar alınırken hata oluştu: {str(e)}")
        print(f"Lütfen önce {model_name.lower().replace(' ', '_')}.py scriptini çalıştırın.")
        return None, None

def main():
    # Tüm modellerin listesi
    models = ["Logistic Regression", "XGBoost", "Random Forest"]
    
    # Performans sonuçlarını ve özellik önemlerini topla
    all_results = []
    top_features = []
    
    for model_name in models:
        print(f"\n{model_name} modeli için sonuçlar alınıyor...")
        results, feature_importance = load_model_results(model_name)
        
        if results is not None and feature_importance is not None:
            all_results.append(results)
            
            # En önemli 3 özelliği al
            top_3_features = feature_importance.head(3)
            
            # Her bir özellik için bir satır oluştur
            for i, (_, row) in enumerate(top_3_features.iterrows()):
                top_features.append({
                    "Model": model_name,
                    "Rank": i + 1,
                    "Feature": row["Feature"],
                    "Importance": row["Importance"]
                })
    
    # Sonuçları DataFrame'e dönüştür
    if all_results:
        results_df = pd.DataFrame(all_results)
        features_df = pd.DataFrame(top_features)
        
        # CSV dosyalarına kaydet
        results_df.to_csv("model_performance_summary.csv", index=False)
        features_df.to_csv("top_features_summary.csv", index=False)
        
        print("\nModel performans özeti ve en önemli özellikler CSV dosyalarına kaydedildi:")
        print("- model_performance_summary.csv")
        print("- top_features_summary.csv")
        
        # Sonuçları ekrana yazdır
        print("\nModel Performans Özeti:")
        print(results_df)
        
        print("\nEn Önemli 3 Özellik:")
        for model_name in models:
            model_features = features_df[features_df["Model"] == model_name]
            print(f"\n{model_name}:")
            for _, row in model_features.iterrows():
                print(f"  {row['Rank']}. {row['Feature']}: {row['Importance']:.6f}")
    else:
        print("Hiçbir model sonucu bulunamadı. Lütfen önce model eğitim scriptlerini çalıştırın.")

if __name__ == "__main__":
    main()
