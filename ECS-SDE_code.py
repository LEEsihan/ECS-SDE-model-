import os
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from evaluation import evaluation_measures
from ECS.costcla.metrics import savings_score 
from sklearn.metrics import f1_score, confusion_matrix
from Gmdhpy.plot_model import PlotModel
from pytorch_tabnet.tab_model import TabNetClassifier
from Gmdhpy.gmdh import  Classifier_
from joblib import load

folder_name = "Graph_results"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
file_path = os.path.join(folder_name, 'feature_importance.xlsx')
file_path2 = os.path.join(folder_name, 'network_weights.xlsx')
 
file_path3 = os.path.join(folder_name, 'X_test_predicted_probas.npy')  
gmdh_model_path = os.path.join(folder_name, 'gmdh_model.pkl')  
gmdh_pred_path = os.path.join(folder_name, 'gmdh_pred.npy')  

def visualize_feature_importance(feature_importances, feature_names, model_identifier, folder_name):
    custom_dpi = 600
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(6, 3))
    fig.set_dpi(custom_dpi)
    plt.bar(range(len(feature_importances)), feature_importances,
                    color=[cmap(importance) for importance in feature_importances])
    plt.xlabel('Feature Index', fontsize=6)
    plt.ylabel('Feature Importance', fontsize=6)
    short_feature_names = [name[:10] for name in feature_names] 
    plt.xticks(range(len(feature_importances)), short_feature_names, rotation=30, ha='right')
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Importance Color', fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    plt.tick_params(axis='both', labelsize=4)
    save_path = os.path.join(folder_name, f'feature_importance_plot_{model_identifier}.png')
    plt.savefig(save_path, dpi=custom_dpi)  
    plt.close()

def train_tabnet_model( model_index, Train_x,Train_y):   
    X_train, X_val, y_train, y_val = train_test_split(Train_x,Train_y, test_size=0.1, random_state=42+ model_index)
    tabnet_params = {
        'n_d': 24,
        'n_a': 8,
        'n_steps': 3,
        'gamma': 1.0,
        'momentum': 0.8,
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=0.01)
    }
 
    clf = TabNetClassifier(**tabnet_params)
    clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=40, patience=5, 
            virtual_batch_size=32,weights=1,
            batch_size=128)  
    
    return clf

#--------------------------------------------------Training
X, y, cost_mat, feature_names = load('credit_data_1.joblib')
scaler = MinMaxScaler()  
scaler.fit(X)
X = scaler.transform(X)
sets = train_test_split(X, y,cost_mat,test_size=0.4,random_state=42)
X_train, X_test, Train_y, y_test, Train_cost_mat,test_cost_mat = sets

num_models = 20
tabnet_models = Parallel(n_jobs=5)(
        delayed(train_tabnet_model)(i, X_train,Train_y) for i in range(num_models)
        )

feature_importance_dfs = []
pred_test_list1 = []
pred_test_list2 = []
for model_index, model in enumerate(tabnet_models):
    print(f"Evaluating TabNet Model {model_index + 1}/{num_models}")
    pred_test = model.predict_proba(X_test)[:, 1]
    tabnet_pred_test = np.where(pred_test > 0.5, 1, 0)
    f1_clf = f1_score(y_test, tabnet_pred_test)
    print('F1 Score %0.6f:'%f1_clf )
    auc_roc2,auc_pr2,bs_min2,bs_maj2,bs2 = evaluation_measures(y_test, tabnet_pred_test)
    print('roc_auc %0.6f:'%auc_roc2)
    print("AUC_PR %0.6f:"%auc_pr2)
    print('bs_min %0.6f:'%bs_min2)
    print('bs_maj %0.6f:'%bs_maj2)
    print('bs2 %0.6f:'%bs2)

    feature_importance_df = pd.DataFrame(index=feature_names)    
    feature_importances = model.feature_importances_
    feature_importance_df[f'TabNet_{model_index + 1}'] = feature_importances
    model_identifier = f'TabNet_{model_index + 1}'
    visualize_feature_importance(feature_importances, feature_names, model_identifier, folder_name)
    feature_importance_dfs.append(feature_importance_df)  
    combined_feature_importance_df = pd.concat(feature_importance_dfs, axis=1)
    combined_feature_importance_df.to_excel(file_path)
    pred_test_list1.append(tabnet_pred_test)
    pred_test_list2.append(pred_test)

tabnet_pred_train = np.column_stack([model.predict_proba(X_train)[:, 1] for model in tabnet_models])
tabnet_pred_test2 = np.column_stack(pred_test_list2)

np.save(file_path3, tabnet_pred_test2)   

#--------------------------------------------------GMDH
GMDH = Classifier_(seq_type='random', 
                   ref_functions=('linear_cov'), 
                   max_layer_count = 20,
                   admix_features = True,
                   criterion_type='custom')    
GMDH.fit(tabnet_pred_train, Train_y,cost_mat=Train_cost_mat)
gmdh_pred, all_layer_weights = GMDH.predict_proba(tabnet_pred_test2)

with open(gmdh_model_path, 'wb') as f:
    pickle.dump(GMDH, f)
 
np.save(gmdh_pred_path, gmdh_pred)

df_weights = pd.DataFrame(all_layer_weights)
df_weights.to_excel(file_path2)
print("gmdh_pred",gmdh_pred)
auc_roc,auc_pr,bs_min,bs_maj,_ = evaluation_measures(y_test,gmdh_pred)
print('1_roc_auc: %0.6f;'%auc_roc)
print("2_AUC_PR: %0.6f;"%auc_pr)
print('3_bs_min: %0.6f;'%bs_min)
print('4_bs_maj: %0.6f;'%bs_maj)
 
gmdh_pred_ = np.where(gmdh_pred > 0.5, 1, 0)
tn, fp, fn, tp = confusion_matrix(y_test, gmdh_pred_).ravel()
recall = tp / (tp + fn)
tnr = tn / (tn + fp)
g_mean = np.sqrt(recall * tnr)
print('5_g_mean: %0.6f'%g_mean)
print("6_save %0.6f" %savings_score(y_test,gmdh_pred_, test_cost_mat))
print("Selected features: {}".format(GMDH.get_selected_features()))
PlotModel(GMDH, filename=os.path.join(folder_name, 'ECS_GMDH'),
          plot_neuron_name=True, view=True).plot()

