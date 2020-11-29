import numpy as np
import pandas as pd

def max_min_Scale(data, MAX = None, MIN = None):
    if not MAX and not MIN:
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        Attr = data.columns.tolist()[1:]
        for att in Attr:
            data[att] = data[[att]].apply(max_min_scaler)
    else:
        Attr = data.columns.tolist()[1:]
        for i in range(len(Attr)):
            att = Attr[i]
            data[att] = data[[att]].apply(lambda x : (x - MAX[i]) / (MAX[i] - MIN[i]) )

def load_COMEX_Data(kind = "Copper", usage = "train", MAX = None, MIN = None, delay = 1):
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
    COMEX.fillna(method = 'ffill', inplace = True)
    COMEX.dropna(axis = 0, how = "any",inplace = True)
    
    if usage == "train":
        COMEX_MAX = COMEX.max().tolist()[1:]
        COMEX_MIN = COMEX.min().tolist()[1:]
        max_min_Scale(COMEX, MAX, MIN)
        outFolderName = "./DataFolders/" + usage + "/COMEX_" + kind + "_" + usage + "_handler.csv"
        COMEX.to_csv(outFolderName,index = False,sep=',')
        return COMEX, COMEX_MAX, COMEX_MIN, COMEX.iloc[-delay : , :]
    else:
        max_min_Scale(COMEX, MAX, MIN)
        outFolderName = "./DataFolders/" + usage + "/COMEX_" + kind + "_" + usage + "_handler.csv"
        COMEX.to_csv(outFolderName,index = False,sep=',')  
        return COMEX

def load_Indices_Data(kind = "NKY", usage = "train", MAX = None, MIN = None, delay = 1):
    Indices_names = ["id","date"]
    Indices_names.append(kind)

    FolderName = "Train_data" if usage == "train" else "Validation_data"
    Indices_path = FolderName + "/" + "Indices" + "/" + "Indices_" + kind + " Index_"+ usage + ".csv"
    Indices = pd.read_csv(Indices_path, skiprows = 1,names = Indices_names)
    Indices.drop(labels = "id", axis = 1,inplace = True)
    Indices.fillna(method = 'ffill', inplace = True)
    Indices.dropna(axis = 0, how = "any",inplace = True)

    if usage == "train":
        Indices_MAX = Indices.max().tolist()[1:]
        Indices_MIN = Indices.min().tolist()[1:]
        max_min_Scale(Indices, MAX, MIN)
        outFolderName = "./DataFolders/" + usage + "/Indices_" + kind + "_" + usage + "_handler.csv"
        Indices.to_csv(outFolderName,index = False,sep=',')
        return Indices, Indices_MAX, Indices_MIN, Indices.iloc[-delay : , :]
    else:
        max_min_Scale(Indices, MAX, MIN)
        outFolderName = "./DataFolders/" + usage + "/Indices_" + kind + "_" + usage + "_handler.csv"
        Indices.to_csv(outFolderName,index = False,sep=',')  
        return Indices

def load_LME_Data(kind = "Copper", usage = "train", MAX = None, MIN = None, delay = 1):
    LME_names = ["id","date","LME_" + kind]

    FolderName = "Train_data" if usage == "train" else "Validation_data"
    LME_path = FolderName + "/" + "LME" + "/" + "LME" + kind + "_OI_"+ usage + ".csv"
    LME = pd.read_csv(LME_path, skiprows = 1,names = LME_names)
    LME.drop(labels = "id", axis = 1,inplace = True)
    LME.fillna(method = 'ffill', inplace = True)
    LME.dropna(axis = 0, how = "any",inplace = True)

    if usage == "train":
        LME_MAX = LME.max().tolist()[1:]
        LME_MIN = LME.min().tolist()[1:]
        max_min_Scale(LME, MAX, MIN)
        outFolderName = "./DataFolders/" + usage + "/LME" + kind + "_OI_" + usage + "_handler.csv"
        LME.to_csv(outFolderName,index = False,sep=',')
        return LME, LME_MAX, LME_MIN, LME.iloc[-delay : , :]
    else:
        max_min_Scale(LME, MAX, MIN)
        outFolderName = "./DataFolders/" + usage + "/LME" + kind + "_OI_" + usage + "_handler.csv"
        LME.to_csv(outFolderName,index = False,sep=',')
        return LME

def load_LME_3M_Data(kind = "Copper", usage = "train", MAX = None, MIN = None, delay = 1):
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
    LME_3M.fillna(method = 'ffill', inplace = True)
    LME_3M.dropna(axis = 0, how = "any",inplace = True)

    if usage == "train":
        LME_3M_MAX = LME_3M.max().tolist()[1:]
        LME_3M_MIN = LME_3M.min().tolist()[1:]
        max_min_Scale(LME_3M, MAX, MIN)
        outFolderName = "./DataFolders/" + usage + "/LME" + kind + "3M_" + usage + "_handler.csv"
        LME_3M.to_csv(outFolderName,index = False,sep=',')
        return LME_3M, LME_3M_MAX, LME_3M_MIN, LME_3M.iloc[-delay : , :]
    else:
        max_min_Scale(LME_3M, MAX, MIN)
        outFolderName = "./DataFolders/" + usage + "/LME" + kind + "3M_" + usage + "_handler.csv"
        LME_3M.to_csv(outFolderName,index = False,sep=',')
        return LME_3M

def load_LME_Label(kind = "Copper", seq = "1d", delay = 1):
    LME_Label_name = ["id","date", "label_" + kind]
    LME_Label_path = "Train_data" + "/" + "Label" + "/" + "Label_LME" + kind + "_train_" + seq + ".csv"
    LME_Label = pd.read_csv(LME_Label_path,skiprows = 1, names = LME_Label_name)
    LME_Label.drop(labels = "id", axis = 1,inplace = True)

    outFolderName = "./DataFolders/" + "train" + "/Label_LME" + kind + "_train_" + seq  + "_handler.csv"
    LME_Label.to_csv(outFolderName,index = False,sep=',')
    return  LME_Label, LME_Label.iloc[-delay : , :]

def load_COMEX_Train_Validation(delay = 1):
    s_train = {}
    Copper, Copper_MAX, Copper_MIN, Copper_Extra = load_COMEX_Data(kind = "Copper", usage = "train", delay = delay)
    Gold, Gold_MAX, Gold_MIN, Gold_Extra = load_COMEX_Data(kind = "Gold", usage = "train", delay = delay)
    Palladium, Palladium_MAX, Palladium_MIN, Palladium_Extra = load_COMEX_Data(kind = "Palladium", usage = "train", delay = delay)
    Platinum, Platinum_MAX, Platinum_MIN, Platinum_Extra = load_COMEX_Data(kind = "Platinum", usage = "train", delay = delay)
    Silver, Silver_MAX, Silver_MIN, Silver_Extra = load_COMEX_Data(kind = "Silver", usage = "train", delay = delay)
    s_train["Copper"] = Copper
    s_train["Gold"] = Gold
    s_train["Palladium"] = Palladium
    s_train["Platinum"] = Platinum
    s_train["Silver"] = Silver

    s_validation = {}
    Copper = load_COMEX_Data(kind = "Copper", usage = "validation", MAX = Copper_MAX, MIN = Copper_MIN)
    Gold = load_COMEX_Data(kind = "Gold", usage = "validation", MAX = Gold_MAX, MIN = Gold_MIN)
    Palladium = load_COMEX_Data(kind = "Palladium", usage = "validation", MAX = Palladium_MAX, MIN = Palladium_MIN)
    Platinum = load_COMEX_Data(kind = "Platinum", usage = "validation", MAX = Platinum_MAX, MIN = Platinum_MIN)
    Silver = load_COMEX_Data(kind = "Silver", usage = "validation", MAX = Silver_MAX, MIN = Silver_MIN)
    s_validation["Copper"] = pd.concat([Copper_Extra, Copper])
    s_validation["Gold"] = pd.concat([Gold_Extra, Gold])
    s_validation["Palladium"] = pd.concat([Palladium_Extra, Palladium])
    s_validation["Platinum"] = pd.concat([Platinum_Extra, Platinum])
    s_validation["Silver"] = pd.concat([Silver_Extra, Silver])
    return s_train, s_validation

def load_Indices_Train_Validation(delay = 1):
    s_train = {}
    NKY, NKY_MAX, NKY_MIN, NKY_Extra = load_Indices_Data(kind = "NKY", usage = "train", delay = delay)
    SHSZ300, SHSZ300_MAX, SHSZ300_MIN, SHSZ300_Extra = load_Indices_Data(kind = "SHSZ300", usage = "train", delay = delay)
    SPX, SPX_MAX, SPX_MIN, SPX_Extra = load_Indices_Data(kind = "SPX", usage = "train", delay = delay)
    SX5E, SX5E_MAX, SX5E_MIN, SX5E_Extra = load_Indices_Data(kind = "SX5E", usage = "train", delay = delay)
    UKX, UKX_MAX, UKX_MIN, UKX_Extra = load_Indices_Data(kind = "UKX", usage = "train", delay = delay)
    VIX, VIX_MAX, VIX_MIN, VIX_Extra = load_Indices_Data(kind = "VIX", usage = "train", delay = delay)
    s_train["NKY"] = NKY
    s_train["SHSZ300"] = SHSZ300
    s_train["SPX"] = SPX
    s_train["SX5E"] = SX5E
    s_train["UKX"] = UKX
    s_train["VIX"] = VIX

    s_validation = {}
    NKY = load_Indices_Data(kind = "NKY", usage = "validation", MAX = NKY_MAX, MIN = NKY_MIN)
    SHSZ300 = load_Indices_Data(kind = "SHSZ300", usage = "validation", MAX = SHSZ300_MAX, MIN = SHSZ300_MIN)
    SPX = load_Indices_Data(kind = "SPX", usage = "validation", MAX = SPX_MAX, MIN = SPX_MIN)
    SX5E = load_Indices_Data(kind = "SX5E", usage = "validation", MAX = SX5E_MAX, MIN = SX5E_MIN)
    UKX = load_Indices_Data(kind = "UKX", usage = "validation", MAX = UKX_MAX, MIN = UKX_MIN)
    VIX = load_Indices_Data(kind = "VIX", usage = "validation", MAX = VIX_MAX, MIN = VIX_MIN)
    s_validation["NKY"] = pd.concat([NKY_Extra, NKY])
    s_validation["SHSZ300"] = pd.concat([SHSZ300_Extra, SHSZ300])
    s_validation["SPX"] = pd.concat([SPX_Extra, SPX])
    s_validation["SX5E"] = pd.concat([SX5E_Extra, SX5E])
    s_validation["UKX"] = pd.concat([UKX_Extra, UKX])
    s_validation["VIX"] = pd.concat([VIX_Extra, VIX])
    return s_train, s_validation

def load_LME_Train_Validation(delay = 1):
    s_train = {}
    Copper, Copper_MAX, Copper_MIN, Copper_Extra = load_LME_Data(kind = "Copper", usage = "train", delay = delay)
    Aluminium, Aluminium_MAX, Aluminium_MIN, Aluminium_Extra = load_LME_Data(kind = "Aluminium", usage = "train", delay = delay)
    Lead, Lead_MAX, Lead_MIN, Lead_Extra = load_LME_Data(kind = "Lead", usage = "train", delay = delay)
    Nickel, Nickel_MAX, Nickel_MIN, Nickel_Extra = load_LME_Data(kind = "Nickel", usage = "train", delay = delay)
    Tin, Tin_MAX, Tin_MIN, Tin_Extra = load_LME_Data(kind = "Tin", usage = "train", delay = delay)
    Zinc, Zinc_MAX, Zinc_MIN, Zinc_Extra = load_LME_Data(kind = "Zinc", usage = "train", delay = delay)
    s_train["Copper"] = Copper
    s_train["Aluminium"] = Aluminium
    s_train["Lead"] = Lead
    s_train["Nickel"] = Nickel
    s_train["Tin"] = Tin
    s_train["Zinc"] = Zinc

    s_validation = {}
    Copper = load_LME_Data(kind = "Copper", usage = "validation", MAX = Copper_MAX, MIN = Copper_MIN)
    Aluminium = load_LME_Data(kind = "Aluminium", usage = "validation", MAX = Aluminium_MAX, MIN = Aluminium_MIN)
    Lead = load_LME_Data(kind = "Lead", usage = "validation", MAX = Lead_MAX, MIN = Lead_MIN)
    Nickel = load_LME_Data(kind = "Nickel", usage = "validation", MAX = Nickel_MAX, MIN = Nickel_MIN)
    Tin = load_LME_Data(kind = "Tin", usage = "validation", MAX = Tin_MAX, MIN = Tin_MIN)
    Zinc = load_LME_Data(kind = "Zinc", usage = "validation", MAX = Zinc_MAX, MIN = Zinc_MIN)
    s_validation["Copper"] = pd.concat([Copper_Extra, Copper])
    s_validation["Aluminium"] = pd.concat([Aluminium_Extra, Aluminium])
    s_validation["Lead"] = pd.concat([Lead_Extra, Lead])
    s_validation["Nickel"] = pd.concat([Nickel_Extra, Nickel])
    s_validation["Tin"] = pd.concat([Tin_Extra, Tin])
    s_validation["Zinc"] = pd.concat([Zinc_Extra, Zinc])
    return s_train, s_validation

def load_LME_3M_Train_Validation(delay = 1):
    s_train = {}
    Copper, Copper_MAX, Copper_MIN, Copper_Extra = load_LME_3M_Data(kind = "Copper", usage = "train", delay = delay)
    Aluminium, Aluminium_MAX, Aluminium_MIN, Aluminium_Extra = load_LME_3M_Data(kind = "Aluminium", usage = "train", delay = delay)
    Lead, Lead_MAX, Lead_MIN, Lead_Extra = load_LME_3M_Data(kind = "Lead", usage = "train", delay = delay)
    Nickel, Nickel_MAX, Nickel_MIN, Nickel_Extra = load_LME_3M_Data(kind = "Nickel", usage = "train", delay = delay)
    Tin, Tin_MAX, Tin_MIN, Tin_Extra = load_LME_3M_Data(kind = "Tin", usage = "train", delay = delay)
    Zinc, Zinc_MAX, Zinc_MIN, Zinc_Extra = load_LME_3M_Data(kind = "Zinc", usage = "train", delay = delay)
    s_train["Copper"] = Copper
    s_train["Aluminium"] = Aluminium
    s_train["Lead"] = Lead
    s_train["Nickel"] = Nickel
    s_train["Tin"] = Tin
    s_train["Zinc"] = Zinc

    s_validation = {}
    Copper = load_LME_3M_Data(kind = "Copper", usage = "validation", MAX = Copper_MAX, MIN = Copper_MIN)
    Aluminium = load_LME_3M_Data(kind = "Aluminium", usage = "validation", MAX = Aluminium_MAX, MIN = Aluminium_MIN)
    Lead = load_LME_3M_Data(kind = "Lead", usage = "validation", MAX = Lead_MAX, MIN = Lead_MIN)
    Nickel = load_LME_3M_Data(kind = "Nickel", usage = "validation", MAX = Nickel_MAX, MIN = Nickel_MIN)
    Tin = load_LME_3M_Data(kind = "Tin", usage = "validation", MAX = Tin_MAX, MIN = Tin_MIN)
    Zinc = load_LME_3M_Data(kind = "Zinc", usage = "validation", MAX = Zinc_MAX, MIN = Zinc_MIN)
    s_validation["Copper"] = pd.concat([Copper_Extra, Copper])
    s_validation["Aluminium"] = pd.concat([Aluminium_Extra, Aluminium])
    s_validation["Lead"] = pd.concat([Lead_Extra, Lead])
    s_validation["Nickel"] = pd.concat([Nickel_Extra, Nickel])
    s_validation["Tin"] = pd.concat([Tin_Extra, Tin])
    s_validation["Zinc"] = pd.concat([Zinc_Extra, Zinc])
    return s_train, s_validation

def load_LME_Label_1d(delay = 1):
    s = {}
    Copper, Copper_Extra = load_LME_Label(kind = "Copper", seq = "1d", delay = delay)
    Aluminium, Aluminium_Extra = load_LME_Label(kind = "Aluminium", seq = "1d", delay = delay)
    Lead, Lead_Extra = load_LME_Label(kind = "Lead", seq = "1d", delay = delay)
    Nickel, Nickel_Extra = load_LME_Label(kind = "Nickel", seq = "1d", delay = delay)
    Tin, Tin_Extra = load_LME_Label(kind = "Tin", seq = "1d", delay = delay)
    Zinc, Zinc_Extra = load_LME_Label(kind = "Zinc", seq = "1d", delay = delay)
    s["Copper"] = Copper
    s["Aluminium"] = Aluminium
    s["Lead"] = Lead
    s["Nickel"] = Nickel
    s["Tin"] = Tin
    s["Zinc"] = Zinc
    extra = {}
    extra["Copper"] = Copper_Extra
    extra["Aluminium"] = Aluminium_Extra
    extra["Lead"] = Lead_Extra
    extra["Nickel"] = Nickel_Extra
    extra["Tin"] = Tin_Extra
    extra["Zinc"] = Zinc_Extra
    return s, extra

def load_LME_Label_20d(delay = 20):
    s = {}
    Copper, Copper_Extra = load_LME_Label(kind = "Copper", seq = "20d", delay = delay)
    Aluminium, Aluminium_Extra = load_LME_Label(kind = "Aluminium", seq = "20d", delay = delay)
    Lead, Lead_Extra = load_LME_Label(kind = "Lead", seq = "20d", delay = delay)
    Nickel, Nickel_Extra = load_LME_Label(kind = "Nickel", seq = "20d", delay = delay)
    Tin, Tin_Extra = load_LME_Label(kind = "Tin", seq = "20d", delay = delay)
    Zinc, Zinc_Extra = load_LME_Label(kind = "Zinc", seq = "20d", delay = delay)
    s["Copper"] = Copper
    s["Aluminium"] = Aluminium
    s["Lead"] = Lead
    s["Nickel"] = Nickel
    s["Tin"] = Tin
    s["Zinc"] = Zinc
    extra = {}
    extra["Copper"] = Copper_Extra
    extra["Aluminium"] = Aluminium_Extra
    extra["Lead"] = Lead_Extra
    extra["Nickel"] = Nickel_Extra
    extra["Tin"] = Tin_Extra
    extra["Zinc"] = Zinc_Extra
    return s, extra

def load_LME_Label_60d(delay = 60):
    s = {}
    Copper, Copper_Extra = load_LME_Label(kind = "Copper", seq = "60d", delay = delay)
    Aluminium, Aluminium_Extra = load_LME_Label(kind = "Aluminium", seq = "60d", delay = delay)
    Lead, Lead_Extra = load_LME_Label(kind = "Lead", seq = "60d", delay = delay)
    Nickel, Nickel_Extra = load_LME_Label(kind = "Nickel", seq = "60d", delay = delay)
    Tin, Tin_Extra = load_LME_Label(kind = "Tin", seq = "60d", delay = delay)
    Zinc, Zinc_Extra = load_LME_Label(kind = "Zinc", seq = "60d", delay = delay)
    s["Copper"] = Copper
    s["Aluminium"] = Aluminium
    s["Lead"] = Lead
    s["Nickel"] = Nickel
    s["Tin"] = Tin
    s["Zinc"] = Zinc
    extra = {}
    extra["Copper"] = Copper_Extra
    extra["Aluminium"] = Aluminium_Extra
    extra["Lead"] = Lead_Extra
    extra["Nickel"] = Nickel_Extra
    extra["Tin"] = Tin_Extra
    extra["Zinc"] = Zinc_Extra
    return s, extra

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
            t.rename(columns={'label':'label_' + label_name},inplace=True) 
            s[label_name + seq_name] = t
            outFolderName = "Validation_data" + "/Split_Validation_Label/" + label_name + "_" + seq_name + "_split_handler.csv"
            t.to_csv(outFolderName,index = False,sep=',')
    return  s

def merge_test():
    df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']})
    df2 = pd.DataFrame({'B': ['B1', 'B2', 'B3', 'B4'],'E': ['D2', 'D3', 'D6', 'D7'],'F': ['F2', 'F3', 'F6', 'F7']})
    result1 = pd.concat([df1, df2], axis = 0)
    result2 = pd.merge(df1,df2, how = 'inner', on = 'B')
    result3 = pd.merge(df1,df2, how = 'outer', on = 'B')
    print(df1)
    print(df2)
    # print(result1)
    # print(result2)
    print(result3)
    print(result3.fillna(method="ffill").fillna(method = "bfill"))

def merge_test2(sequence_length = 7):
    # 训练集 测试集的features
    LME_train, LME_validation = load_LME_Train_Validation(sequence_length)
    LME_3M_train, LME_3M_validation = load_LME_3M_Train_Validation(sequence_length)
    # merge main features
    keys = ["Aluminium", "Lead", "Nickel", "Tin", "Zinc"]
    LME_train_all, LME_validation_all = LME_train["Copper"], LME_validation["Copper"]
    LME_3M_train_all, LME_3M_validation_all = LME_3M_train["Copper"], LME_3M_validation["Copper"]
    print(LME_train_all)
    print(LME_3M_train_all)
    LME_train_all = pd.merge(LME_train_all,LME_3M_train_all, how = 'outer', on = 'date', sort = True)
    print(LME_train_all)
    LME_train_all.to_csv("ss.csv",index = False,sep=',')
    LME_train_all.fillna(method="ffill", inplace=True)
    LME_train_all.fillna(method="bfill", inplace=True)
    print(LME_train_all)
    print(LME_train_all.iloc[-1,:])
    print(LME_train_all.isna().sum())
    
def main():
    # S, mx, mn, ds = load_COMEX_Data(kind = "Copper", usage = "train", delay = 20)
    # Ss = load_COMEX_Data(kind = "Copper", usage = "validation", MAX = mx, MIN = mn)
    # a = load_LME_3M_Data(kind = "Zinc", usage = "train")

    # s_train, s_validation = load_COMEX_Train_Validation(20)
    # print(s_validation["Copper"])
    # s_train, s_validation = load_Indices_Train_Validation(20)
    # print(s_validation["NKY"])
    s_train, s_validation = load_LME_Train_Validation(20)
    print(s_train["Copper"])
    # s_train, s_validation = load_LME_3M_Train_Validation(20)
    # print(s_validation["Copper"])
    # s_label1, s_label2 = load_LME_Label_1d(delay = 12)
    # print(s_label2["Copper"])

if __name__ == "__main__":
    main()
    # merge_test()
    # merge_test2()