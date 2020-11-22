import numpy as np
import pandas as pd

def max_min_Scale(data):
    # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    max_min_scaler = lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)) # Copper_1d
    # max_min_scaler = lambda x : (x - np.mean(x)) / np.std(x)
    Attr = data.columns.tolist()[1:]
    for att in Attr:
        data[att] = data[[att]].apply(max_min_scaler)

def load_COMEX_Data(kind = "Copper", usage = "train"):
    COMEX_names = ["id","date"]
    COMEX_names.append("COMEX_" + kind + "_Open")
    COMEX_names.append("COMEX_" + kind + "_High")
    COMEX_names.append("COMEX_" + kind + "_Low")
    COMEX_names.append("COMEX_" + kind + "_Close")
    COMEX_names.append("COMEX_" + kind + "_Volumn")
    COMEX_names.append("COMEX_" + kind + "_Open_Interest")

    FolderName = "Train_data" if usage == "train" else "Validation_data"
    COMEX_path = FolderName + "/" + "COMEX" + "/" + "COMEX_" + kind + "_"+ usage + ".csv"
    COMEX = pd.read_csv(COMEX_path, skiprows = 1,names = COMEX_names)
    COMEX.drop(labels = "id", axis = 1,inplace = True)
    COMEX.dropna(axis = 0, how = "any",inplace = True)

    max_min_Scale(COMEX)

    outFolderName = "./DataFolders/" + usage + "/COMEX_" + kind + "_" + usage + "_handler.csv"
    COMEX.to_csv(outFolderName,index = False,sep=',')
    return COMEX

def load_Indices_Data(kind = "NKY", usage = "train"):
    Indices_names = ["id","date"]
    Indices_names.append(kind)

    FolderName = "Train_data" if usage == "train" else "Validation_data"
    Indices_path = FolderName + "/" + "Indices" + "/" + "Indices_" + kind + " Index_"+ usage + ".csv"
    Indices = pd.read_csv(Indices_path, skiprows = 1,names = Indices_names)
    Indices.drop(labels = "id", axis = 1,inplace = True)
    Indices.dropna(axis = 0, how = "any",inplace = True)

    max_min_Scale(Indices)

    outFolderName = "./DataFolders/" + usage + "/Indices_" + kind + "_" + usage + "_handler.csv"
    Indices.to_csv(outFolderName,index = False,sep=',')
    return Indices

def load_LME_Data(kind = "Copper", usage = "train"):
    LME_names = ["id","date","LME_Attr"]

    FolderName = "Train_data" if usage == "train" else "Validation_data"
    LME_path = FolderName + "/" + "LME" + "/" + "LME" + kind + "_OI_"+ usage + ".csv"
    LME = pd.read_csv(LME_path, skiprows = 1,names = LME_names)
    LME.drop(labels = "id", axis = 1,inplace = True)
    LME.dropna(axis = 0, how = "any",inplace = True)

    max_min_Scale(LME)

    outFolderName = "./DataFolders/" + usage + "/LME" + kind + "_OI_" + usage + "_handler.csv"
    LME.to_csv(outFolderName,index = False,sep=',')
    return LME

def load_LME_3M_Data(kind = "Copper", usage = "train"):
    LME_3M_names = ["id","date"]
    LME_3M_names.append(kind + "_Open")
    LME_3M_names.append(kind + "_High")
    LME_3M_names.append(kind + "_Low")
    LME_3M_names.append(kind + "_Close")
    LME_3M_names.append(kind + "_Volumn")

    FolderName = "Train_data" if usage == "train" else "Validation_data"
    LME_3M_path = FolderName + "/" + "LME" + "/" + "LME" + kind + "3M_"+ usage + ".csv"
    LME_3M = pd.read_csv(LME_3M_path, skiprows = 1,names = LME_3M_names)
    LME_3M.drop(labels = "id", axis = 1,inplace = True)
    LME_3M.dropna(axis = 0, how = "any",inplace = True)

    max_min_Scale(LME_3M)

    outFolderName = "./DataFolders/" + usage + "/LME" + kind + "3M_" + usage + "_handler.csv"
    LME_3M.to_csv(outFolderName,index = False,sep=',')
    return LME_3M

def load_LME_Label(kind = "Copper", seq = "1d"):
    LME_Label_1d_name = ["id","date", "label"]
    LME_Label_1d_path = "Train_data" + "/" + "Label" + "/" + "Label_LME" + kind + "_train_" + seq + ".csv"
    LME_Label_1d = pd.read_csv(LME_Label_1d_path,skiprows = 1, names = LME_Label_1d_name)
    LME_Label_1d.drop(labels = "id", axis = 1,inplace = True)

    outFolderName = "./DataFolders/" + "train" + "/Label_LME" + kind + "_train_" + seq  + "_handler.csv"
    LME_Label_1d.to_csv(outFolderName,index = False,sep=',')
    return  LME_Label_1d

def load_COMEX_Train_Validation():
    s_train = {}
    Copper = load_COMEX_Data(kind = "Copper", usage = "train")
    Gold = load_COMEX_Data(kind = "Gold", usage = "train")
    Palladium = load_COMEX_Data(kind = "Palladium", usage = "train")
    Platinum = load_COMEX_Data(kind = "Platinum", usage = "train")
    Silver = load_COMEX_Data(kind = "Silver", usage = "train")
    s_train["Copper"] = Copper
    s_train["Gold"] = Gold
    s_train["Palladium"] = Palladium
    s_train["Platinum"] = Platinum
    s_train["Silver"] = Silver

    s_validation = {}
    Copper = load_COMEX_Data(kind = "Copper", usage = "validation")
    Gold = load_COMEX_Data(kind = "Gold", usage = "validation")
    Palladium = load_COMEX_Data(kind = "Palladium", usage = "validation")
    Platinum = load_COMEX_Data(kind = "Platinum", usage = "validation")
    Silver = load_COMEX_Data(kind = "Silver", usage = "validation")
    s_validation["Copper"] = Copper
    s_validation["Gold"] = Gold
    s_validation["Palladium"] = Palladium
    s_validation["Platinum"] = Platinum
    s_validation["Silver"] = Silver

    return s_train, s_validation

def load_Indices_Train_Validation():
    s_train = {}
    NKY = load_Indices_Data(kind = "NKY", usage = "train")
    SHSZ300 = load_Indices_Data(kind = "SHSZ300", usage = "train")
    SPX = load_Indices_Data(kind = "SPX", usage = "train")
    SX5E = load_Indices_Data(kind = "SX5E", usage = "train")
    UKX = load_Indices_Data(kind = "UKX", usage = "train")
    VIX = load_Indices_Data(kind = "VIX", usage = "train")
    s_train["NKY"] = NKY
    s_train["SHSZ300"] = SHSZ300
    s_train["SPX"] = SPX
    s_train["SX5E"] = SX5E
    s_train["UKX"] = UKX
    s_train["VIX"] = VIX

    s_validation = {}
    NKY = load_Indices_Data(kind = "NKY", usage = "validation")
    SHSZ300 = load_Indices_Data(kind = "SHSZ300", usage = "validation")
    SPX = load_Indices_Data(kind = "SPX", usage = "validation")
    SX5E = load_Indices_Data(kind = "SX5E", usage = "validation")
    UKX = load_Indices_Data(kind = "UKX", usage = "validation")
    VIX = load_Indices_Data(kind = "VIX", usage = "validation")
    s_validation["NKY"] = NKY
    s_validation["SHSZ300"] = SHSZ300
    s_validation["SPX"] = SPX
    s_validation["SX5E"] = SX5E
    s_validation["UKX"] = UKX
    s_validation["VIX"] = VIX

    return s_train, s_validation

def load_LME_Train_Validation():
    s_train = {}
    Copper = load_LME_Data(kind = "Copper", usage = "train")
    Aluminium = load_LME_Data(kind = "Aluminium", usage = "train")
    Lead = load_LME_Data(kind = "Lead", usage = "train")
    Nickel = load_LME_Data(kind = "Nickel", usage = "train")
    Tin = load_LME_Data(kind = "Tin", usage = "train")
    Zinc = load_LME_Data(kind = "Zinc", usage = "train")
    s_train["Copper"] = Copper
    s_train["Aluminium"] = Aluminium
    s_train["Lead"] = Lead
    s_train["Nickel"] = Nickel
    s_train["Tin"] = Tin
    s_train["Zinc"] = Zinc

    s_validation = {}
    Copper = load_LME_Data(kind = "Copper", usage = "validation")
    Aluminium = load_LME_Data(kind = "Aluminium", usage = "validation")
    Lead = load_LME_Data(kind = "Lead", usage = "validation")
    Nickel = load_LME_Data(kind = "Nickel", usage = "validation")
    Tin = load_LME_Data(kind = "Tin", usage = "validation")
    Zinc = load_LME_Data(kind = "Zinc", usage = "validation")
    s_validation["Copper"] = Copper
    s_validation["Aluminium"] = Aluminium
    s_validation["Lead"] = Lead
    s_validation["Nickel"] = Nickel
    s_validation["Tin"] = Tin
    s_validation["Zinc"] = Zinc
    return s_train, s_validation

def load_LME_3M_Train_Validation():
    s_train = {}
    Copper = load_LME_3M_Data(kind = "Copper", usage = "train")
    Aluminium = load_LME_3M_Data(kind = "Aluminium", usage = "train")
    Lead = load_LME_3M_Data(kind = "Lead", usage = "train")
    Nickel = load_LME_3M_Data(kind = "Nickel", usage = "train")
    Tin = load_LME_3M_Data(kind = "Tin", usage = "train")
    Zinc = load_LME_3M_Data(kind = "Zinc", usage = "train")
    s_train["Copper"] = Copper
    s_train["Aluminium"] = Aluminium
    s_train["Lead"] = Lead
    s_train["Nickel"] = Nickel
    s_train["Tin"] = Tin
    s_train["Zinc"] = Zinc

    s_validation = {}
    Copper = load_LME_3M_Data(kind = "Copper", usage = "validation")
    Aluminium = load_LME_3M_Data(kind = "Aluminium", usage = "validation")
    Lead = load_LME_3M_Data(kind = "Lead", usage = "validation")
    Nickel = load_LME_3M_Data(kind = "Nickel", usage = "validation")
    Tin = load_LME_3M_Data(kind = "Tin", usage = "validation")
    Zinc = load_LME_3M_Data(kind = "Zinc", usage = "validation")
    s_validation["Copper"] = Copper
    s_validation["Aluminium"] = Aluminium
    s_validation["Lead"] = Lead
    s_validation["Nickel"] = Nickel
    s_validation["Tin"] = Tin
    s_validation["Zinc"] = Zinc
    return s_train, s_validation

def load_LME_Label_1d():
    s = {}
    Copper = load_LME_Label(kind = "Copper", seq = "1d")
    Aluminium = load_LME_Label(kind = "Aluminium", seq = "1d")
    Lead = load_LME_Label(kind = "Lead", seq = "1d")
    Nickel = load_LME_Label(kind = "Nickel", seq = "1d")
    Tin = load_LME_Label(kind = "Tin", seq = "1d")
    Zinc = load_LME_Label(kind = "Zinc", seq = "1d")
    s["Copper"] = Copper
    s["Aluminium"] = Aluminium
    s["Lead"] = Lead
    s["Nickel"] = Nickel
    s["Tin"] = Tin
    s["Zinc"] = Zinc
    return s

def load_LME_Label_20d():
    s = {}
    Copper = load_LME_Label(kind = "Copper", seq = "20d")
    Aluminium = load_LME_Label(kind = "Aluminium", seq = "20d")
    Lead = load_LME_Label(kind = "Lead", seq = "20d")
    Nickel = load_LME_Label(kind = "Nickel", seq = "20d")
    Tin = load_LME_Label(kind = "Tin", seq = "20d")
    Zinc = load_LME_Label(kind = "Zinc", seq = "20d")
    s["Copper"] = Copper
    s["Aluminium"] = Aluminium
    s["Lead"] = Lead
    s["Nickel"] = Nickel
    s["Tin"] = Tin
    s["Zinc"] = Zinc
    return s

def load_LME_Label_60d():
    s = {}
    Copper = load_LME_Label(kind = "Copper", seq = "60d")
    Aluminium = load_LME_Label(kind = "Aluminium", seq = "60d")
    Lead = load_LME_Label(kind = "Lead", seq = "60d")
    Nickel = load_LME_Label(kind = "Nickel", seq = "60d")
    Tin = load_LME_Label(kind = "Tin", seq = "60d")
    Zinc = load_LME_Label(kind = "Zinc", seq = "60d")
    s["Copper"] = Copper
    s["Aluminium"] = Aluminium
    s["Lead"] = Lead
    s["Nickel"] = Nickel
    s["Tin"] = Tin
    s["Zinc"] = Zinc
    return s

def load_Validation_Label():
    Validation_Label_name = ["raw_id", "label"]
    Validation_Label_path = "Validation_data" + "/" + "validation_label_file" + ".csv"
    Validation_Label = pd.read_csv(Validation_Label_path,skiprows = 1, names = Validation_Label_name)

    Label_name = ["Aluminium","Copper","Lead","Nickel","Tin","Zinc"]
    Seq_name =  ["1d","20d","60d"]

    s = {}
    for label_name in Label_name:
        for seq_name in Seq_name:
            p = "LME" + label_name + "-validation-" + seq_name + "-.*"
            # t = Validation_Label[Validation_Label["raw_id"].str.extract(r'(LMELead-validation-60d-2018-01-02)(\d)')]
            t = pd.DataFrame(Validation_Label.loc[Validation_Label["raw_id"].str.contains(p)])
            t["raw_id"] = t["raw_id"].apply(lambda x: x[-10:])
            t.rename(columns={'raw_id':'date'},inplace=True) 
            s[label_name + seq_name] = t
            outFolderName = "Validation_data" + "/Split_Validation_Label/" + label_name + "_" + seq_name + "_split_handler.csv"
            t.to_csv(outFolderName,index = False,sep=',')
    return  s

def main():
    s = load_LME_Label_60d()
    print(s["Copper"])

if __name__ == "__main__":
    main()