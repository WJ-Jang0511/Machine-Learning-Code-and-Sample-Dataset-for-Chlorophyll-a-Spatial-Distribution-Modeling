import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from collections import Counter
import matplotlib.pyplot as plt


def Run_PCA(File, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(os.getcwd(), File)
    df = pd.read_csv(file_path)

    df_data = df.drop(columns=['Date', 'CHL-A'])
    df_data = df_data.dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_data)

    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)

    components = pd.DataFrame(pca.components_, columns=df_data.columns,
                              index=[f'PC{i + 1}' for i in range(pca.n_components_)])

    num_components_to_check = min(5, components.shape[1])
    important_variables = set()

    for i in range(num_components_to_check):
        sorted_components = components.iloc[i, :].abs().sort_values(ascending=False)
        top_variables = sorted_components.head(5 - len(important_variables)).index
        important_variables.update(top_variables)
        if len(important_variables) >= 5:
            break

    important_variables = list(important_variables)[:5]
    print(f"5 important variables: {important_variables}")

    selected_df = df[important_variables]
    selected_df.to_csv('selected_variables.csv', index=False)

    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', color='k')
    plt.xlabel('Number of Principal Components', fontsize=16, fontweight='bold')
    plt.ylabel('Explained Variance Ratio', fontsize=16, fontweight='bold')
    plt.grid(axis='x', linestyle='--', linewidth=0.7, color='gray')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, color='gray')
    plt.xticks(range(0, len(explained_variance) + 1, 5), fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.axvline(x=5, color='red', linestyle='--', linewidth=1.0)

    plt.savefig("PCA_result.jpg")
    plt.show()

    principal_df = pd.DataFrame(data=pca_data, columns=[f'PC{i + 1}' for i in range(pca_data.shape[1])])
    components = pd.DataFrame(pca.components_, columns=df_data.columns,
                              index=[f'PC{i + 1}' for i in range(pca.n_components_)])
    components.to_csv('PCA_components_relationship.csv')

    return important_variables

def DataSampling(File, TargetBand, classnum, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(File)

    if classnum == 2:
        df['Class'] = df['CHL-A'].apply(lambda x: 1 if x < 87.1 else 2)
        X = df[TargetBand]
        y = df['Class']
        X_train_class1, X_test_class1, y_train_class1, y_test_class1 = train_test_split(X[y == 1], y[y == 1],
                                                                                        test_size=0.2)
        X_train_class2, X_test_class2, y_train_class2, y_test_class2 = train_test_split(X[y == 2], y[y == 2],
                                                                                        test_size=0.2)
        X_train = pd.concat([X_train_class1, X_train_class2])
        y_train = pd.concat([y_train_class1, y_train_class2])
        X_test = pd.concat([X_test_class1, X_test_class2])
        y_test = pd.concat([y_test_class1, y_test_class2])

    elif classnum == 3:
        df['Class'] = df['CHL-A'].apply(lambda x: 1 if x <= 57.31 else (2 if x <= 110.93 else 3))
        X = df[TargetBand]
        y = df['Class']
        X_train_class1, X_test_class1, y_train_class1, y_test_class1 = train_test_split(X[y == 1], y[y == 1],
                                                                                        test_size=0.2)
        X_train_class2, X_test_class2, y_train_class2, y_test_class2 = train_test_split(X[y == 2], y[y == 2],
                                                                                        test_size=0.2)
        X_train_class3, X_test_class3, y_train_class3, y_test_class3 = train_test_split(X[y == 3], y[y == 3],
                                                                                        test_size=0.2)
        X_train = pd.concat([X_train_class1, X_train_class2, X_train_class3])
        y_train = pd.concat([y_train_class1, y_train_class2, y_train_class3])
        X_test = pd.concat([X_test_class1, X_test_class2, X_test_class3])
        y_test = pd.concat([y_test_class1, y_test_class2, y_test_class3])
    else:
        print('Number of class more than 3, please select 2 or 3')
        return

    smote = SMOTE(k_neighbors=2)

    X_res_smote, y_res_smote = smote.fit_resample(X_train, y_train)
    pd.DataFrame(X_res_smote, columns=X_train.columns).to_csv(os.path.join(output_dir, 'SMOTE.csv'), index=False)

    smoteenn = SMOTEENN(smote=smote)
    X_res_smoteenn, y_res_smoteenn = smoteenn.fit_resample(X_train, y_train)
    pd.DataFrame(X_res_smoteenn, columns=X_train.columns).to_csv(os.path.join(output_dir, 'SMOTEENN.csv'), index=False)

    smotetomek = SMOTETomek(smote=smote)
    X_res_smotetomek, y_res_smotetomek = smotetomek.fit_resample(X_train, y_train)
    pd.DataFrame(X_res_smotetomek, columns=X_train.columns).to_csv(os.path.join(output_dir, 'SMOTETomek.csv'),
                                                                   index=False)

    pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).to_csv(
        os.path.join(output_dir, 'Orgin_train.csv'), index=False)
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
        os.path.join(output_dir, 'test.csv'), index=False)

    print(f"모든 결과 파일이 '{output_dir}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    output_folder = "ML_DB"

    important_variables = Run_PCA(r"Sample_Input.csv", output_folder)

    TargetBandRatio = important_variables + ["CHL-A"]
    classnum = 3

    DataSampling(r"Sample_Input.csv", TargetBandRatio, classnum, output_folder)