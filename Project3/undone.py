
def load_COMEX_Copper_train():
    COMEX_Copper_train_names = ["id","date","Coppper_Open","Coppper_High","Coppper_Low","Coppper_Close","Coppper_Volumn","Coppper_Open_Interest"]
    COMEX_Copper_train_path = "Train_data" + "/" + "COMEX" + "/"+"COMEX_Copper_train" + ".csv"
    COMEX_Copper_train = pd.read_csv(COMEX_Copper_train_path, skiprows = 1,names = COMEX_Copper_train_names)
    COMEX_Copper_train.drop(labels = "id", axis = 1,inplace = True)
    COMEX_Copper_train.dropna(axis = 0, how = "any",inplace = True)
    max_min_Scale(COMEX_Copper_train)
    COMEX_Copper_train.to_csv("./DataFolders/Train/COMEX_Copper_train_handler.csv",index=False,sep=',')
    return COMEX_Copper_train

def load_COMEX_Gold_train():
    COMEX_Gold_train_names = ["id","date","Gold_Open","Gold_High","Gold_Low","Gold_Close","Gold_Volumn","Gold_Open_Interest"]
    COMEX_Gold_train_path = "Train_data" + "/" + "COMEX" + "/"+"COMEX_Gold_train" + ".csv"
    COMEX_Gold_train = pd.read_csv(COMEX_Gold_train_path, skiprows = 1,names = COMEX_Gold_train_names)
    COMEX_Gold_train.drop(labels = "id", axis = 1,inplace = True)
    COMEX_Gold_train.dropna(axis = 0, how = "any",inplace = True)
    max_min_Scale(COMEX_Gold_train)
    COMEX_Gold_train.to_csv("./DataFolders/Train/COMEX_Gold_train_handler.csv",index=False,sep=',')
    return COMEX_Gold_train

def load_COMEX_Palladium_train():
    COMEX_Palladium_train_names = ["id","date","Palladium_Open","Palladium_High","Palladium_Low","Palladium_Close","Palladium_Volumn","Palladium_Open_Interest"]
    COMEX_Palladium_train_path = "Train_data" + "/" + "COMEX" + "/"+"COMEX_Palladium_train" + ".csv"
    COMEX_Palladium_train = pd.read_csv(COMEX_Palladium_train_path, skiprows = 1,names = COMEX_Palladium_train_names)
    COMEX_Palladium_train.drop(labels = "id", axis = 1,inplace = True)
    COMEX_Palladium_train.dropna(axis = 0, how = "any",inplace = True)
    max_min_Scale(COMEX_Palladium_train)
    COMEX_Palladium_train.to_csv("./DataFolders/Train/COMEX_Palladium_train_handler.csv",index=False,sep=',')
    return COMEX_Palladium_train

def load_COMEX_Platinum_train():
    COMEX_Platinum_train_names = ["id","date","Platinum_Open","Platinum_High","Platinum_Low","Platinum_Close","Platinum_Volumn","Platinum_Open_Interest"]
    COMEX_Platinum_train_path = "Train_data" + "/" + "COMEX" + "/"+"COMEX_Platinum_train" + ".csv"
    COMEX_Platinum_train = pd.read_csv(COMEX_Platinum_train_path, skiprows = 1,names = COMEX_Platinum_train_names)
    COMEX_Platinum_train.drop(labels = "id", axis = 1,inplace = True)
    COMEX_Platinum_train.dropna(axis = 0, how = "any",inplace = True)
    max_min_Scale(COMEX_Platinum_train)
    COMEX_Platinum_train.to_csv("./DataFolders/Train/COMEX_Platinum_train_handler.csv",index=False,sep=',')
    return COMEX_Platinum_train

def load():
    

    AI_train_names = ["id","date","attr"]
    Al_train_path = "Train_data" + "/" + "LME" + "/"+"LMEAluminium_OI_train" + ".csv"
    Al_train = pd.read_csv(Al_train_path, skiprows = 1,names = AI_train_names)
    Al_train.drop(labels = "id", axis = 1,inplace = True)

    Al_label_1d_name = ["id","date", "label"]
    Al_label_1d_path = "Train_data" + "/" + "Label" + "/"+"Label_LMEAluminium_train_1d" + ".csv"
    Al_label_1d = pd.read_csv(Al_label_1d_path,skiprows = 1, names = Al_label_1d_name)
    Al_label_1d.drop(labels = "id", axis = 1,inplace = True)


    Al_train_label_1d = pd.merge(Al_train,COMEX_Copper_train, how = 'inner', on = 'date')
    Al_train_label_1d = pd.merge(Al_train_label_1d,Al_label_1d, how = 'inner', on = 'date')
    Al_train_label_1d.dropna(axis = 0, how = "any",inplace = True)


    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    Attr = Al_train_label_1d.columns.tolist()[1:]
    for att in Attr:
        Al_train_label_1d[att] = Al_train_label_1d[[att]].apply(max_min_scaler)
    Al_train_label_1d.to_csv("Al_train_label_1d.csv",index=False,sep=',')
    
    # s = Al_train_label_1d.isnull().any(axis=0)
    # print(s)
    return Al_train_label_1d