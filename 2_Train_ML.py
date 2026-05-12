import numpy as np
import pandas as pd
import os, math, glob
import pickle
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

def results_write(csv, list):
    for k in list:
        csv.write("{0},".format(k))
    csv.write("\n")

def MinMaxScaler2(array):
    maxmin = []
    Max = 500
    Min = 0
    array = (array - Min) / (Max - Min)
    maxmin.append(Max)
    maxmin.append(Min)
    return maxmin, array

def filter_nan(s, o):
    s = np.array(s).tolist()
    o = np.array(o).tolist()
    x = []
    y = []
    for length in range(0, len(s)):
        if float(o[length]) != 0 and not np.isnan(o[length]):
            x.append(s[length])
            y.append(float(o[length]))
    return x, y

def NSE(ObservedStreamFlow, SimulatedStreamFlow):
    x = ObservedStreamFlow
    y = SimulatedStreamFlow
    x, y = filter_nan(x, y)
    try:
        if len(x) == 0: return 0
        mean_observed = sum(x) / len(x)
        numerator = sum([(obs - sim) ** 2 for obs, sim in zip(x, y)])
        denominator = sum([(obs - mean_observed) ** 2 for obs in x])
        if denominator == 0: return 0
        E = 1 - (numerator / denominator)
    except:
        E = 0
    return E

def RMSE(s, o):
    try:
        A = 0.0
        s, o = filter_nan(s, o)
        if len(o) == 0: return 9999
        for i in range(len(o)):
            A += (s[i] - o[i])**2
        rmse = math.sqrt(A/len(o))
    except:
        rmse = 9999
    return rmse

def PB(SimulatedStreamFlow, ObservedStreamFlow):
    x = SimulatedStreamFlow
    y = ObservedStreamFlow
    x, y = filter_nan(x, y)
    A = 0.0
    B = 0.0
    for i in range(0, len(y)):
        A += (y[i] - x[i])
        B += y[i]
    if B == 0: return 0
    PB = 100.0 * (A / B)
    return PB


############## Data Load #################################################################################################

folderlist = ["ML_DB"]

for folder in folderlist:
    DBset = glob.glob(os.path.join(folder, "*.csv"))

    for DB in DBset:
        DBname = os.path.basename(DB)

        if DBname.lower() == "test.csv":
            continue

        prefix = DBname.split(".")[0].split("_")[0]
        pkl_folder = os.path.join(folder, f"{prefix}_pkl")
        os.makedirs(pkl_folder, exist_ok=True)

        Result_csv = open(os.path.join(folder, f"{prefix}_Result.csv"), 'a')

        data = pd.read_csv(DB)
        data_te = pd.read_csv(os.path.join(folder, "test.csv"))

        X = data[['B3/B6', 'B2/B6', 'B4/B6', 'B3/B7', 'B3/B8']].values
        Y = data["CHL-A"].values
        X_te = data_te[['B3/B6', 'B2/B6', 'B4/B6', 'B3/B7', 'B3/B8']].values
        Y_te = data_te["CHL-A"].values

        Scaler = MinMaxScaler(feature_range=(0, 1))
        X_n = Scaler.fit_transform(X)
        ymaxmin, Y_n = MinMaxScaler2(Y)
        X_te_n = Scaler.transform(X_te)
        ymaxmin2, Y_te_n = MinMaxScaler2(Y_te)

        xTrain, xVal, yTrain, yVal = train_test_split(X_n, Y_n, test_size=0.001, random_state=42)
        xTest = X_te_n
        yTest = Y_te_n

        OTy = yTrain * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]
        OVy = Y_te_n * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]

        ############## PLS #########################################################################################
        n_components = range(1, 5)
        tR2C_PLS = -999
        tR2V_PLS = -999
        Best_PLS = None

        for nc in n_components:
            PLSR = PLSRegression(n_components=nc)
            PLSR.fit(xTrain, yTrain)

            y_pred_tr = PLSR.predict(xTrain).flatten()
            y_pred_vl = PLSR.predict(xTest).flatten()

            R2C_PLS = np.corrcoef(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
            R2V_PLS = np.corrcoef(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2

            if R2V_PLS + R2C_PLS / 2 > tR2C_PLS + tR2V_PLS / 2:
                tR2C_PLS = R2C_PLS
                tR2V_PLS = R2V_PLS
                best_nc = nc
                by_pred_tr = y_pred_tr
                by_pred_vl = y_pred_vl
                Best_NSEC = NSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
                Best_NSEV = NSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
                Best_RMSEC = RMSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
                Best_RMSEV = RMSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
                Best_PBC = PB(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
                Best_PBV = PB(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
                Best_PLS = PLSR

        pls_model_path = os.path.join(pkl_folder, f"{DBname.split('.')[0]}_PLS.pkl")
        with open(pls_model_path, 'wb') as file:
            pickle.dump(Best_PLS, file)

        # Result_csv DBnum -> DBname으로 변경
        Result_csv.write("PLS,{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
            tR2C_PLS, tR2V_PLS, Best_NSEC, Best_NSEV, Best_RMSEC, Best_RMSEV, Best_PBC, Best_PBV, DBname))
        Result_csv.write("Num_est:{0}\n".format(best_nc))

        best_Ty = by_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]
        best_Vy = by_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]

        # 데이터 기록
        Result_csv.write("Observed Tr :," + ",".join(map(str, OTy)) + ",\n")
        Result_csv.write("predicted Tr :," + ",".join(map(str, best_Ty)) + ",\n")
        Result_csv.write("Observed Vl :," + ",".join(map(str, OVy)) + ",\n")
        Result_csv.write("predicted Vl :," + ",".join(map(str, best_Vy)) + ",\n")

        ############## Random_Forest #####################################################################################
        param_list = {"n_estimators": list(range(5, 200, 5)),
                      "max_depth": list(range(1, 20, 2)),
                      "min_samples_leaf": list(range(1, 20, 1))}

        rf_model = RandomForestRegressor(random_state=42)
        rf_random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_list, cv=5, n_jobs=4,
                                              n_iter=50)
        rf_random_search.fit(xTrain, yTrain)

        rf_model_path = os.path.join(pkl_folder, f"{DBname.split('.')[0]}_RF.pkl")
        with open(rf_model_path, 'wb') as file:
            pickle.dump(rf_random_search, file)

        y_pred_tr = rf_random_search.predict(xTrain).flatten()
        y_pred_vl = rf_random_search.predict(xTest).flatten()

        R2C_PLS = np.corrcoef(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        R2V_PLS = np.corrcoef(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        Best_NSEC = NSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_NSEV = NSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEC = RMSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEV = RMSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBC = PB(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBV = PB(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])

        Result_csv.write("RF,{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
            R2C_PLS, R2V_PLS, Best_NSEC, Best_NSEV, Best_RMSEC, Best_RMSEV, Best_PBC, Best_PBV, DBname))
        Result_csv.write("{0}\n".format(rf_random_search.best_params_))

        best_Ty = y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]
        best_Vy = y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]

        Result_csv.write("Observed Tr :," + ",".join(map(str, OTy)) + ",\n")
        Result_csv.write("predicted Tr :," + ",".join(map(str, best_Ty)) + ",\n")
        Result_csv.write("Observed Vl :," + ",".join(map(str, OVy)) + ",\n")
        Result_csv.write("predicted Vl :," + ",".join(map(str, best_Vy)) + ",\n")

        ############## Gradient Boosting #################################################################################
        param_list = {"n_estimators": list(range(5, 200, 5)),
                      "learning_rate": [0.1, 0.5, 1, 0.2],
                      "max_depth": list(range(1, 20, 2)),
                      "min_samples_leaf": list(range(1, 20, 1)),
                      'subsample': [0.5, 0.7, 0.9, 1.0],
                      'min_samples_split': list(range(2, 10, 1))}

        GB_model = GradientBoostingRegressor(random_state=42)
        GB_random_search = RandomizedSearchCV(estimator=GB_model, param_distributions=param_list, cv=5, n_jobs=4,
                                              n_iter=50)
        GB_random_search.fit(xTrain, yTrain)

        gb_model_path = os.path.join(pkl_folder, f"{DBname.split('.')[0]}_GB.pkl")
        with open(gb_model_path, 'wb') as file:
            pickle.dump(GB_random_search, file)

        y_pred_tr = GB_random_search.predict(xTrain).flatten()
        y_pred_vl = GB_random_search.predict(xTest).flatten()

        R2C_PLS = np.corrcoef(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        R2V_PLS = np.corrcoef(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        Best_NSEC = NSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_NSEV = NSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEC = RMSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEV = RMSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBC = PB(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBV = PB(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])

        Result_csv.write("GB,{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
            R2C_PLS, R2V_PLS, Best_NSEC, Best_NSEV, Best_RMSEC, Best_RMSEV, Best_PBC, Best_PBV, DBname))
        Result_csv.write("{0}\n".format(GB_random_search.best_params_))

        best_Ty = y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]
        best_Vy = y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]

        Result_csv.write("Observed Tr :," + ",".join(map(str, OTy)) + ",\n")
        Result_csv.write("predicted Tr :," + ",".join(map(str, best_Ty)) + ",\n")
        Result_csv.write("Observed Vl :," + ",".join(map(str, OVy)) + ",\n")
        Result_csv.write("predicted Vl :," + ",".join(map(str, best_Vy)) + ",\n")

        ############## XGBoost #################################################################################
        param_list = {"n_estimators": list(range(5, 200, 5)),
                      "learning_rate": [0.1, 0.5, 1],
                      "max_depth": list(range(1, 20, 2)),
                      "min_child_weight": list(range(1, 10, 1)),
                      'subsample': [0.8, 0.9, 1.0],
                      'colsample_bytree': [0.8, 0.9, 1.0],
                      'reg_alpha': [0, 0.1, 1],
                      'reg_lambda': [1, 1.5, 2]}

        XGB_model = XGBRegressor(random_state=42)
        XGB_random_search = RandomizedSearchCV(estimator=XGB_model, param_distributions=param_list, cv=5, n_jobs=4,
                                               n_iter=50)
        XGB_random_search.fit(xTrain, yTrain)

        xgb_model_path = os.path.join(pkl_folder, f"{DBname.split('.')[0]}_XGB.pkl")
        with open(xgb_model_path, 'wb') as file:
            pickle.dump(XGB_random_search, file)

        y_pred_tr = XGB_random_search.predict(xTrain).flatten()
        y_pred_vl = XGB_random_search.predict(xTest).flatten()

        R2C_PLS = np.corrcoef(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        R2V_PLS = np.corrcoef(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        Best_NSEC = NSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_NSEV = NSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEC = RMSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEV = RMSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBC = PB(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBV = PB(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])

        Result_csv.write("XGB,{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
            R2C_PLS, R2V_PLS, Best_NSEC, Best_NSEV, Best_RMSEC, Best_RMSEV, Best_PBC, Best_PBV, DBname))
        Result_csv.write("{0}\n".format(XGB_random_search.best_params_))

        best_Ty = y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]
        best_Vy = y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]

        Result_csv.write("Observed Tr :," + ",".join(map(str, OTy)) + ",\n")
        Result_csv.write("predicted Tr :," + ",".join(map(str, best_Ty)) + ",\n")
        Result_csv.write("Observed Vl :," + ",".join(map(str, OVy)) + ",\n")
        Result_csv.write("predicted Vl :," + ",".join(map(str, best_Vy)) + ",\n")

        ############## SVR #####################################################################################
        param_list = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                      "epsilon": [0.1, 0.2, 0.3, 0.4, 0.5],
                      "C": list(range(1, 100, 10)),
                      "gamma": [0.0001, 0.001, 0.01, 0.1, 1]}

        SVR_model = SVR()
        SVR_random_search = RandomizedSearchCV(estimator=SVR_model, param_distributions=param_list, cv=5, n_jobs=4,
                                               n_iter=50)
        SVR_random_search.fit(xTrain, yTrain)

        svr_model_path = os.path.join(pkl_folder, f"{DBname.split('.')[0]}_SVR.pkl")
        with open(svr_model_path, 'wb') as file:
            pickle.dump(SVR_random_search, file)

        y_pred_tr = SVR_random_search.predict(xTrain).flatten()
        y_pred_vl = SVR_random_search.predict(xTest).flatten()

        R2C_PLS = np.corrcoef(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        R2V_PLS = np.corrcoef(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        Best_NSEC = NSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_NSEV = NSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEC = RMSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEV = RMSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBC = PB(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBV = PB(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])

        Result_csv.write("SVM,{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
            R2C_PLS, R2V_PLS, Best_NSEC, Best_NSEV, Best_RMSEC, Best_RMSEV, Best_PBC, Best_PBV, DBname))
        Result_csv.write("{0}\n".format(SVR_random_search.best_params_))

        best_Ty = y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]
        best_Vy = y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]

        Result_csv.write("Observed Tr :," + ",".join(map(str, OTy)) + ",\n")
        Result_csv.write("predicted Tr :," + ",".join(map(str, best_Ty)) + ",\n")
        Result_csv.write("Observed Vl :," + ",".join(map(str, OVy)) + ",\n")
        Result_csv.write("predicted Vl :," + ",".join(map(str, best_Vy)) + ",\n")

        ############## ANN #####################################################################################
        neurons = [5, 10, 20, 30, 50, 80]
        activations = ['tanh', 'elu', 'relu', 'linear']
        batch_num = [10]

        BestRMSE = 999999
        best_Ty_ann = None
        best_Vy_ann = None
        Best_ANN_model = None
        best_ann_params = ""

        xT_ann, xV_ann, yT_ann, yV_ann = train_test_split(xTrain, yTrain, test_size=0.2, random_state=42)

        for neuron in neurons:
            for act1 in activations:
                for act2 in activations:
                    for batch in batch_num:

                        model = Sequential()
                        model.add(Dense(neuron, activation=act1, input_dim=xTrain.shape[1]))
                        model.add(Dense(units=1, activation=act2))
                        model.compile(optimizer='adam', loss='mean_squared_error')

                        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=15,
                                                       restore_best_weights=True)

                        model.fit(xT_ann, yT_ann, batch_size=batch, callbacks=[early_stopping],
                                  validation_data=(xV_ann, yV_ann), verbose=0, epochs=300)

                        # 평가 (1D 배열로 평탄화)
                        y_pred_tr = model.predict(xTrain, verbose=0).flatten()
                        y_pred_vl = model.predict(xTest, verbose=0).flatten()

                        # RMSE 평가 (스케일 복원 후)
                        tr_rmse = RMSE(OTy, y_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
                        vl_rmse = RMSE(OVy, y_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])

                        avg_rmse = (tr_rmse + vl_rmse) / 2

                        if BestRMSE > avg_rmse:
                            BestRMSE = avg_rmse
                            best_ann_params = f"Neu:{neuron}_Act1:{act1}_Act2:{act2}_Bat:{batch}"
                            by_pred_tr = y_pred_tr
                            by_pred_vl = y_pred_vl
                            Best_ANN_model = model

        R2C_ANN = np.corrcoef(OTy, by_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        R2V_ANN = np.corrcoef(OVy, by_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])[0, 1] ** 2
        Best_NSEC = NSE(OTy, by_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_NSEV = NSE(OVy, by_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEC = RMSE(OTy, by_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_RMSEV = RMSE(OVy, by_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBC = PB(OTy, by_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])
        Best_PBV = PB(OVy, by_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1])

        ann_model_path = os.path.join(pkl_folder, f"{DBname.split('.')[0]}_ANN.h5")
        Best_ANN_model.save(ann_model_path)

        Result_csv.write("ANN,{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
            R2C_ANN, R2V_ANN, Best_NSEC, Best_NSEV, Best_RMSEC, Best_RMSEV, Best_PBC, Best_PBV, DBname))
        Result_csv.write("{0}\n".format(best_ann_params))

        best_Ty = by_pred_tr * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]
        best_Vy = by_pred_vl * (ymaxmin[0] - ymaxmin[1]) + ymaxmin[1]

        Result_csv.write("Observed Tr :," + ",".join(map(str, OTy)) + ",\n")
        Result_csv.write("predicted Tr :," + ",".join(map(str, best_Ty)) + ",\n")
        Result_csv.write("Observed Vl :," + ",".join(map(str, OVy)) + ",\n")
        Result_csv.write("predicted Vl :," + ",".join(map(str, best_Vy)) + ",\n")

        # 파일 핸들 닫기
        Result_csv.close()

        print(f"[{DBname}] Train Complete!")