import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from scipy.stats import weibull_min
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm


# register the TUM font from the ttf file
tum_font_path = "fonts/TUMNeueHelvetica-Regular.ttf"   # adjust path to your ttf
fm.fontManager.addfont(tum_font_path)
tum_name = fm.FontProperties(fname=tum_font_path).get_name()

#plt.rcParams.update({"font.size": 17.5, "font.family": "Times New Roman", "axes.labelsize": 17.5, "axes.titlesize": 17.5, "xtick.labelsize": 17.5, "ytick.labelsize": 17.5, "legend.fontsize": 17.5})

plt.rcParams.update({"font.size": 17.5, "font.family": tum_name, "axes.labelsize": 17.5, "axes.titlesize": 17.5, "xtick.labelsize": 17.5, "ytick.labelsize": 17.5, "legend.fontsize": 17.5})


def dictionary_building_heat_sink_temp():

    df_02U_1 = pd.read_parquet("Simulation_results_heat_sink/Simulation_1/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_02U_1 = max((max(df_02U_1["Tj_igbt"]) - 273.15), (max(df_02U_1["Tj_diode"]) - 273.15))

    df_02U_2 = pd.read_parquet("Simulation_results_heat_sink/Simulation_2/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_02U_2 = max((max(df_02U_2["Tj_igbt"]) - 273.15), (max(df_02U_2["Tj_diode"]) - 273.15))

    df_02U_3 = pd.read_parquet("Simulation_results_heat_sink/Simulation_3/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_02U_3 = max((max(df_02U_3["Tj_igbt"]) - 273.15), (max(df_02U_3["Tj_diode"]) - 273.15))

    df_02U_4 = pd.read_parquet("Simulation_results_heat_sink/Simulation_4/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_02U_4 = max((max(df_02U_4["Tj_igbt"]) - 273.15), (max(df_02U_4["Tj_diode"]) - 273.15))

    dic_02U = {1:Tj_module_max_02U_1,
               2:Tj_module_max_02U_2,
               3:Tj_module_max_02U_3,
               4:Tj_module_max_02U_4}

    del df_02U_1, df_02U_2, df_02U_3, df_02U_4, Tj_module_max_02U_1, Tj_module_max_02U_2, Tj_module_max_02U_3, Tj_module_max_02U_4


    df_04U_1 = pd.read_parquet("Simulation_results_heat_sink/Simulation_5/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_04U_1 = max((max(df_04U_1["Tj_igbt"]) - 273.15), (max(df_04U_1["Tj_diode"]) - 273.15))

    df_04U_2 = pd.read_parquet("Simulation_results_heat_sink/Simulation_6/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_04U_2 = max((max(df_04U_2["Tj_igbt"]) - 273.15), (max(df_04U_2["Tj_diode"]) - 273.15))

    df_04U_3 = pd.read_parquet("Simulation_results_heat_sink/Simulation_7/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_04U_3 = max((max(df_04U_3["Tj_igbt"]) - 273.15), (max(df_04U_3["Tj_diode"]) - 273.15))

    df_04U_4 = pd.read_parquet("Simulation_results_heat_sink/Simulation_8/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_04U_4 = max((max(df_04U_4["Tj_igbt"]) - 273.15), (max(df_04U_4["Tj_diode"]) - 273.15))

    dic_04U = {1:Tj_module_max_04U_1,
               2:Tj_module_max_04U_2,
               3:Tj_module_max_04U_3,
               4:Tj_module_max_04U_4}

    del df_04U_1, df_04U_2, df_04U_3, df_04U_4, Tj_module_max_04U_1, Tj_module_max_04U_2, Tj_module_max_04U_3, Tj_module_max_04U_4


    df_06U_1 = pd.read_parquet("Simulation_results_heat_sink/Simulation_9/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_06U_1 = max((max(df_06U_1["Tj_igbt"]) - 273.15), (max(df_06U_1["Tj_diode"]) - 273.15))

    df_06U_2 = pd.read_parquet("Simulation_results_heat_sink/Simulation_10/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_06U_2 = max((max(df_06U_2["Tj_igbt"]) - 273.15), (max(df_06U_2["Tj_diode"]) - 273.15))

    df_06U_3 = pd.read_parquet("Simulation_results_heat_sink/Simulation_11/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_06U_3 = max((max(df_06U_3["Tj_igbt"]) - 273.15), (max(df_06U_3["Tj_diode"]) - 273.15))

    df_06U_4 = pd.read_parquet("Simulation_results_heat_sink/Simulation_12/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_06U_4 = max((max(df_06U_4["Tj_igbt"]) - 273.15), (max(df_06U_4["Tj_diode"]) - 273.15))

    dic_06U = {1:Tj_module_max_06U_1,
               2:Tj_module_max_06U_2,
               3:Tj_module_max_06U_3,
               4:Tj_module_max_06U_4}

    del df_06U_1, df_06U_2, df_06U_3, df_06U_4, Tj_module_max_06U_1, Tj_module_max_06U_2, Tj_module_max_06U_3, Tj_module_max_06U_4


    df_08U_1 = pd.read_parquet("Simulation_results_heat_sink/Simulation_13/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_08U_1 = max((max(df_08U_1["Tj_igbt"]) - 273.15), (max(df_08U_1["Tj_diode"]) - 273.15))

    df_08U_2 = pd.read_parquet("Simulation_results_heat_sink/Simulation_14/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_08U_2 = max((max(df_08U_2["Tj_igbt"]) - 273.15), (max(df_08U_2["Tj_diode"]) - 273.15))

    df_08U_3 = pd.read_parquet("Simulation_results_heat_sink/Simulation_15/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_08U_3 = max((max(df_08U_3["Tj_igbt"]) - 273.15), (max(df_08U_3["Tj_diode"]) - 273.15))

    df_08U_4 = pd.read_parquet("Simulation_results_heat_sink/Simulation_16/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_08U_4 = max((max(df_08U_4["Tj_igbt"]) - 273.15), (max(df_08U_4["Tj_diode"]) - 273.15))

    dic_08U = {1:Tj_module_max_08U_1,
               2:Tj_module_max_08U_2,
               3:Tj_module_max_08U_3,
               4:Tj_module_max_08U_4}

    del df_08U_1, df_08U_2, df_08U_3, df_08U_4, Tj_module_max_08U_1, Tj_module_max_08U_2, Tj_module_max_08U_3, Tj_module_max_08U_4


    df_10U_1 = pd.read_parquet("Simulation_results_heat_sink/Simulation_17/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_10U_1 = max((max(df_10U_1["Tj_igbt"]) - 273.15), (max(df_10U_1["Tj_diode"]) - 273.15))

    df_10U_2 = pd.read_parquet("Simulation_results_heat_sink/Simulation_18/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_10U_2 = max((max(df_10U_2["Tj_igbt"]) - 273.15), (max(df_10U_2["Tj_diode"]) - 273.15))

    df_10U_3 = pd.read_parquet("Simulation_results_heat_sink/Simulation_19/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_10U_3 = max((max(df_10U_3["Tj_igbt"]) - 273.15), (max(df_10U_3["Tj_diode"]) - 273.15))

    df_10U_4 = pd.read_parquet("Simulation_results_heat_sink/Simulation_20/df_thermal/df_1.parquet",engine="pyarrow")
    Tj_module_max_10U_4 = max((max(df_10U_4["Tj_igbt"]) - 273.15), (max(df_10U_4["Tj_diode"]) - 273.15))

    dic_10U = {1:Tj_module_max_10U_1,
               2:Tj_module_max_10U_2,
               3:Tj_module_max_10U_3,
               4:Tj_module_max_10U_4}

    del df_10U_1, df_10U_2, df_10U_3, df_10U_4, Tj_module_max_10U_1, Tj_module_max_10U_2, Tj_module_max_10U_3, Tj_module_max_10U_4


    print("dic_02U", dic_02U)
    print("dic_04U", dic_04U)
    print("dic_06U", dic_06U)
    print("dic_08U", dic_08U)
    print("dic_10U", dic_10U)

    dict_heat_sink = {"02U":dic_02U,
                      "04U":dic_04U,
                      "06U":dic_06U,
                      "08U":dic_08U,
                      "10U":dic_10U}


    with open("Simulation_results_heat_sink/dict_heat_sink.txt", "w") as f:
        json.dump(dict_heat_sink, f, indent=4)

#dictionary_building()
#plotting_heat_sink_temp_limit()

def dictionary_building_heat_sink_life():

    df_02U_1_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_1/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_02U_1_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_1/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_02U_1 = min(df_02U_1_igbt["lifetime_years_igbt_actual"][0],df_02U_1_diode["lifetime_years_diode_actual"][0] )

    df_02U_2_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_2/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_02U_2_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_2/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_02U_2 = min(df_02U_2_igbt["lifetime_years_igbt_actual"][0],df_02U_2_diode["lifetime_years_diode_actual"][0] )

    df_02U_3_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_3/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_02U_3_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_3/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_02U_3 = min(df_02U_3_igbt["lifetime_years_igbt_actual"][0],df_02U_3_diode["lifetime_years_diode_actual"][0] )

    df_02U_4_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_4/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_02U_4_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_4/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_02U_4 = min(df_02U_4_igbt["lifetime_years_igbt_actual"][0],df_02U_4_diode["lifetime_years_diode_actual"][0] )

    dic_02U = {1: Switch_lifetime_02U_1,
               2: Switch_lifetime_02U_2,
               3: Switch_lifetime_02U_3,
               4: Switch_lifetime_02U_4}

    del df_02U_1_igbt, df_02U_1_diode, df_02U_2_igbt, df_02U_2_diode, df_02U_3_igbt, df_02U_3_diode, df_02U_4_igbt, df_02U_4_diode


    df_04U_1_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_5/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_04U_1_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_5/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_04U_1 = min(df_04U_1_igbt["lifetime_years_igbt_actual"][0],df_04U_1_diode["lifetime_years_diode_actual"][0] )

    df_04U_2_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_6/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_04U_2_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_6/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_04U_2 = min(df_04U_2_igbt["lifetime_years_igbt_actual"][0],df_04U_2_diode["lifetime_years_diode_actual"][0] )

    df_04U_3_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_7/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_04U_3_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_7/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_04U_3 = min(df_04U_3_igbt["lifetime_years_igbt_actual"][0],df_04U_3_diode["lifetime_years_diode_actual"][0] )

    df_04U_4_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_8/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_04U_4_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_8/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_04U_4 = min(df_04U_4_igbt["lifetime_years_igbt_actual"][0],df_04U_4_diode["lifetime_years_diode_actual"][0] )

    dic_04U = {1: Switch_lifetime_04U_1,
               2: Switch_lifetime_04U_2,
               3: Switch_lifetime_04U_3,
               4: Switch_lifetime_04U_4}

    del df_04U_1_igbt, df_04U_1_diode, df_04U_2_igbt, df_04U_2_diode, df_04U_3_igbt, df_04U_3_diode, df_04U_4_igbt, df_04U_4_diode


    df_06U_1_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_9/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_06U_1_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_9/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_06U_1 = min(df_06U_1_igbt["lifetime_years_igbt_actual"][0],df_06U_1_diode["lifetime_years_diode_actual"][0] )

    df_06U_2_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_10/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_06U_2_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_10/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_06U_2 = min(df_06U_2_igbt["lifetime_years_igbt_actual"][0],df_06U_2_diode["lifetime_years_diode_actual"][0] )

    df_06U_3_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_11/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_06U_3_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_11/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_06U_3 = min(df_06U_3_igbt["lifetime_years_igbt_actual"][0],df_06U_3_diode["lifetime_years_diode_actual"][0] )

    df_06U_4_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_12/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_06U_4_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_12/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_06U_4 = min(df_06U_4_igbt["lifetime_years_igbt_actual"][0],df_06U_4_diode["lifetime_years_diode_actual"][0] )

    dic_06U = {1: Switch_lifetime_06U_1,
               2: Switch_lifetime_06U_2,
               3: Switch_lifetime_06U_3,
               4: Switch_lifetime_06U_4}

    del df_06U_1_igbt, df_06U_1_diode, df_06U_2_igbt, df_06U_2_diode, df_06U_3_igbt, df_06U_3_diode, df_06U_4_igbt, df_06U_4_diode


    df_08U_1_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_13/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_08U_1_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_13/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_08U_1 = min(df_08U_1_igbt["lifetime_years_igbt_actual"][0],df_08U_1_diode["lifetime_years_diode_actual"][0] )

    df_08U_2_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_14/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_08U_2_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_14/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_08U_2 = min(df_08U_2_igbt["lifetime_years_igbt_actual"][0],df_08U_2_diode["lifetime_years_diode_actual"][0] )

    df_08U_3_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_15/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_08U_3_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_15/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_08U_3 = min(df_08U_3_igbt["lifetime_years_igbt_actual"][0],df_08U_3_diode["lifetime_years_diode_actual"][0] )


    df_08U_4_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_16/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_08U_4_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_16/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_08U_4 = min(df_08U_4_igbt["lifetime_years_igbt_actual"][0],df_08U_4_diode["lifetime_years_diode_actual"][0] )


    dic_08U = {1: Switch_lifetime_08U_1,
               2: Switch_lifetime_08U_2,
               3: Switch_lifetime_08U_3,
               4: Switch_lifetime_08U_4}

    del df_08U_1_igbt, df_08U_1_diode, df_08U_2_igbt, df_08U_2_diode, df_08U_3_igbt, df_08U_3_diode, df_08U_4_igbt, df_08U_4_diode


    df_10U_1_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_17/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_10U_1_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_17/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_10U_1 = min(df_10U_1_igbt["lifetime_years_igbt_actual"][0],df_10U_1_diode["lifetime_years_diode_actual"][0] )

    df_10U_2_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_18/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_10U_2_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_18/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_10U_2 = min(df_10U_2_igbt["lifetime_years_igbt_actual"][0],df_10U_2_diode["lifetime_years_diode_actual"][0] )

    df_10U_3_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_19/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_10U_3_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_19/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_10U_3 = min(df_10U_3_igbt["lifetime_years_igbt_actual"][0],df_10U_3_diode["lifetime_years_diode_actual"][0] )

    df_10U_4_igbt = pd.read_parquet("Simulation_results_heat_sink/Simulation_20/df_lifetime_IGBT/df_IGBT_final.parquet",engine="pyarrow")
    df_10U_4_diode = pd.read_parquet("Simulation_results_heat_sink/Simulation_20/df_lifetime_Diode/df_Diode_final.parquet",engine="pyarrow")
    Switch_lifetime_10U_4 = min(df_10U_4_igbt["lifetime_years_igbt_actual"][0],df_10U_4_diode["lifetime_years_diode_actual"][0] )



    dic_10U = {1: Switch_lifetime_10U_1,
               2: Switch_lifetime_10U_2,
               3: Switch_lifetime_10U_3,
               4: Switch_lifetime_10U_4}

    del df_10U_1_igbt, df_10U_1_diode, df_10U_2_igbt, df_10U_2_diode, df_10U_3_igbt, df_10U_3_diode, df_10U_4_igbt, df_10U_4_diode

    #print("dic_02U", dic_02U)
    #print("dic_04U", dic_04U)
    #print("dic_06U", dic_06U)
    #print("dic_08U", dic_08U)
    #print("dic_10U", dic_10U)

    dict_heat_sink_life = {"02U": dic_02U,
                      "04U": dic_04U,
                      "06U": dic_06U,
                      "08U": dic_08U,
                      "10U": dic_10U}


    with open("Simulation_results_heat_sink/dict_heat_sink.txt", "r") as f:
        dict_heat_sink = json.load(f)



    for hs in dict_heat_sink:  # 02U, 04U, ...
        for level in dict_heat_sink[hs]:  # "1","2","3","4"

            temp = dict_heat_sink[hs][level]

            if temp > 175:
                dict_heat_sink_life[hs][int(level)] = 0



    with open("Simulation_results_heat_sink/dict_heat_sink_life.txt", "w") as f:
        json.dump(dict_heat_sink_life, f, indent=4)

#dictionary_building_heat_sink_life()

def plotting_heat_sink_temp_limit():

    with open("Simulation_results_heat_sink/dict_heat_sink.txt", "r") as f:
        dict_heat_sink = json.load(f)

    keys = list(dict_heat_sink.keys())              # bar names
    x = np.arange(len(keys))

    # values for each stack level
    v1 = [dict_heat_sink[k]["1"] for k in keys]
    v2 = [dict_heat_sink[k]["2"] for k in keys]
    v3 = [dict_heat_sink[k]["3"] for k in keys]
    v4 = [dict_heat_sink[k]["4"] for k in keys]

    fig, ax = plt.subplots(figsize=(6.4,4.8))

    # ---- overlapping bars ----
    b1 = ax.bar(x, v1, label="1 m/s", color="tab:red")
    b2 = ax.bar(x, v2, label="2 m/s", color="tab:green")
    b3 = ax.bar(x, v3, label="3 m/s", color="tab:orange")
    b4 = ax.bar(x, v4, label="4 m/s", color="tab:blue")

    # ---- formatting ----
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Temperature [°C]")
    ax.set_xlabel("Heat sink configurations")
    ax.set_ylim(0,700)

    limit_line = ax.axhline(y=175, color='black', linestyle='--', linewidth=1.5, label="175 °C limit")

    # legend order: bars first, limit line last
    ax.legend(handles=[b1, b2, b3, b4, limit_line])

    plt.tight_layout()
    plt.savefig("Paper_figures/Heat_sink_case_study_2.png")


#plotting_heat_sink_temp_limit()


def plotting_heat_sink_temp_limit_new():

    with open("Simulation_results_heat_sink/dict_heat_sink.txt", "r") as f:
        dict_heat_sink = json.load(f)

    keys = list(dict_heat_sink.keys())
    x = np.arange(len(keys))
    speeds = ["1", "2", "3", "4"]
    colors = ["tab:red", "tab:green", "tab:orange", "tab:blue"]
    width = 0.2

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for i, (s, c) in enumerate(zip(speeds, colors)):
        vals = [dict_heat_sink[k][s] for k in keys]
        ax.bar(x + (i - 1.5) * width, vals, width,
               label=f"{s} m/s", color=c)

    ax.axhline(175, color="black", ls="--", lw=1.5, label="175 °C limit")
    ax.set_xticks(x, keys)
    ax.set_xlabel("Heat sink configurations")
    ax.set_ylabel("Temperature [°C]")
    ax.legend()
    fig.tight_layout()
    fig.savefig("Paper_figures/Heat_sink_case_study_2.png", dpi=300)

#plotting_heat_sink_temp_limit_new()


def plotting_heat_sink_life():

    with open("Simulation_results_heat_sink/dict_heat_sink_life.txt", "r") as f:
        dict_heat_sink = json.load(f)

    keys = list(dict_heat_sink.keys())              # bar names
    x = np.arange(len(keys))

    # values for each stack level
    v1 = [dict_heat_sink[k]["1"] for k in keys]
    v2 = [dict_heat_sink[k]["2"] for k in keys]
    v3 = [dict_heat_sink[k]["3"] for k in keys]
    v4 = [dict_heat_sink[k]["4"] for k in keys]

    v3 = [0, 0, 1.5167619122326557, 2.523142786274132, 3.1656798323087103]

    fig, ax = plt.subplots(figsize=(6.4,4.8))

    # ---- overlapping bars ----

    ax.bar(x, v4, label="4 m/s", color="tab:blue")
    ax.bar(x, v3, label="3 m/s", color="tab:orange")
    ax.bar(x, v2, label="2 m/s", color="tab:green")
    ax.bar(x, v1, label="1 m/s", color="tab:red")

    # ---- formatting ----
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Lifetime [Years]")
    ax.set_xlabel("Heat sink configurations")
    #ax.set_ylim(0,470)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    plt.tight_layout()
    plt.savefig("Paper_figures/Heat_sink_life.pdf")

#plotting_heat_sink_life()

def plotting_heat_sink_life_new():

    with open("Simulation_results_heat_sink/dict_heat_sink_life.txt", "r") as f:
        dict_heat_sink = json.load(f)

    keys = list(dict_heat_sink.keys())
    x = np.arange(len(keys))
    speeds = ["1", "2", "3", "4"]
    colors = ["tab:red", "tab:green", "tab:orange", "tab:blue"]
    width = 0.2

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for i, (s, c) in enumerate(zip(speeds, colors)):
        vals = [dict_heat_sink[k][s] for k in keys]
        ax.bar(x + (i - 1.5) * width, vals, width,
               label=f"{s} m/s", color=c)

    ax.set_xticks(x, keys)
    ax.set_xlabel("Heat sink configurations")
    ax.set_ylabel("Lifetime [years]")
    ax.legend(title="Air speed")
    fig.tight_layout()
    fig.savefig("Paper_figures/Heat_sink_life.png")

#plotting_heat_sink_life_new()

def plotting_heat_sink_combined():

    # ---- load both datasets ----
    with open("Simulation_results_heat_sink/dict_heat_sink.txt", "r") as f:
        dict_temp = json.load(f)

    with open("Simulation_results_heat_sink/dict_heat_sink_life.txt", "r") as f:
        dict_life = json.load(f)

    keys = list(dict_temp.keys())          # bar names: 02U, 04U, ...
    x = np.arange(len(keys))

    fig, (ax_temp, ax_life) = plt.subplots(1, 2, figsize=(12, 4))

    # ---------- left: junction temperature ----------
    t1 = [dict_temp[k]["1"] for k in keys]
    t2 = [dict_temp[k]["2"] for k in keys]
    t3 = [dict_temp[k]["3"] for k in keys]
    t4 = [dict_temp[k]["4"] for k in keys]

    ax_temp.bar(x, t1, label="1 m/s", color="tab:red")
    ax_temp.bar(x, t2, label="2 m/s", color="tab:green")
    ax_temp.bar(x, t3, label="3 m/s", color="tab:orange")
    ax_temp.bar(x, t4, label="4 m/s", color="tab:blue")

    ax_temp.axhline(y=175, color='black', linestyle=':', linewidth=1.5, label="175 °C limit", zorder=10)

    ax_temp.set_xticks(x)
    ax_temp.set_xticklabels(keys)
    ax_temp.set_ylabel("Temperature [°C]")
    ax_temp.set_xlabel("Heat sink configurations")
    ax_temp.set_ylim(0, 625)
    ax_temp.set_title("Maximum junction temperature")

    handles, labels = ax_temp.get_legend_handles_labels()
    # move the "175 °C limit" entry to the end
    limit_idx = labels.index("175 °C limit")
    order = [i for i in range(len(labels)) if i != limit_idx] + [limit_idx]
    ax_temp.legend([handles[i] for i in order], [labels[i] for i in order],handlelength=1.5, handletextpad=0.5,
                   labelspacing=0.25, borderpad=0.25, framealpha=1.0)

    # ---------- right: lifetime ----------
    l1 = [dict_life[k]["1"] for k in keys]
    l2 = [dict_life[k]["2"] for k in keys]
    l3 = [dict_life[k]["3"] for k in keys]
    l4 = [dict_life[k]["4"] for k in keys]

    l3 = [0, 0, 1.5167619122326557, 2.523142786274132, 3.1656798323087103]

    ax_life.bar(x, l4, label="4 m/s", color="tab:blue")
    ax_life.bar(x, l3, label="3 m/s", color="tab:orange")
    ax_life.bar(x, l2, label="2 m/s", color="tab:green")
    ax_life.bar(x, l1, label="1 m/s", color="tab:red")

    ax_life.set_xticks(x)
    ax_life.set_xticklabels(keys)
    ax_life.set_ylabel("Lifetime [Years]")
    ax_life.set_xlabel("Heat sink configurations")
    ax_life.set_title("Estimated lifetime")
    handles, labels = ax_life.get_legend_handles_labels()
    ax_life.legend(handles[::-1], labels[::-1],
                   handlelength=1.5, handletextpad=0.5,
                   labelspacing=0.25, borderpad=0.25, framealpha=1.0)

    plt.tight_layout()
    plt.savefig("Paper_figures/Heat_sink_combined.pdf")


def plotting_heat_sink_combined_modified():

    # ---- load both datasets ----
    with open("Simulation_results_heat_sink/dict_heat_sink.txt", "r") as f:
        dict_temp = json.load(f)

    with open("Simulation_results_heat_sink/dict_heat_sink_life.txt", "r") as f:
        dict_life = json.load(f)

    keys = list(dict_temp.keys())          # bar names: 02U, 04U, ...
    x = np.arange(len(keys))

    # ---- red (1 m/s, hottest) -> blue (4 m/s, coolest) ----
    #speeds = [1, 2, 3, 4]
    #cmap = plt.get_cmap("RdBu")            # red -> blue
    #speed_colors = {s: cmap(i / (len(speeds) - 1)) for i, s in enumerate(speeds)}
    # speed_colors[1] = red, speed_colors[4] = blue

    # pure red (1 m/s) -> pure blue (4 m/s)
    cmap = mcolors.LinearSegmentedColormap.from_list("red_blue", ["red", "blue"])
    speeds = [1, 2, 3, 4]
    speed_colors = {s: cmap(i / (len(speeds) - 1)) for i, s in enumerate(speeds)}
    # speed_colors[1] = pure red, speed_colors[4] = pure blue, 2/3 = purples in between

    speed_colors = {1: "red",
                    2: "darkorange",
                    3: "cornflowerblue",
                    4: "blue"}

    fig, (ax_temp, ax_life) = plt.subplots(1, 2, figsize=(12, 4))

    # ---------- left: junction temperature ----------
    t1 = [dict_temp[k]["1"] for k in keys]
    t2 = [dict_temp[k]["2"] for k in keys]
    t3 = [dict_temp[k]["3"] for k in keys]
    t4 = [dict_temp[k]["4"] for k in keys]

    ax_temp.bar(x, t1, label="1 m/s", color=speed_colors[1])
    ax_temp.bar(x, t2, label="2 m/s", color=speed_colors[2])
    ax_temp.bar(x, t3, label="3 m/s", color=speed_colors[3])
    ax_temp.bar(x, t4, label="4 m/s", color=speed_colors[4])

    ax_temp.axhline(y=175, color='black', linestyle=':', linewidth=1.5, label="175 °C limit", zorder=10)

    ax_temp.set_xticks(x)
    ax_temp.set_xticklabels(keys)
    ax_temp.set_ylabel("Temperature [°C]")
    ax_temp.set_xlabel("Heat sink configurations")
    ax_temp.set_ylim(0, 625)
    ax_temp.set_title("Maximum junction temperature")

    handles, labels = ax_temp.get_legend_handles_labels()
    # move the "175 °C limit" entry to the end
    limit_idx = labels.index("175 °C limit")
    order = [i for i in range(len(labels)) if i != limit_idx] + [limit_idx]
    ax_temp.legend([handles[i] for i in order], [labels[i] for i in order],
                   handlelength=1.5, handletextpad=0.5,
                   labelspacing=0.25, borderpad=0.25, framealpha=1.0)

    # ---------- right: lifetime ----------
    l1 = [dict_life[k]["1"] for k in keys]
    l2 = [dict_life[k]["2"] for k in keys]
    l3 = [dict_life[k]["3"] for k in keys]
    l4 = [dict_life[k]["4"] for k in keys]

    l3 = [0, 0, 1.5167619122326557, 2.523142786274132, 3.1656798323087103]

    ax_life.bar(x, l4, label="4 m/s", color=speed_colors[4])
    ax_life.bar(x, l3, label="3 m/s", color=speed_colors[3])
    ax_life.bar(x, l2, label="2 m/s", color=speed_colors[2])
    ax_life.bar(x, l1, label="1 m/s", color=speed_colors[1])

    ax_life.set_xticks(x)
    ax_life.set_xticklabels(keys)
    ax_life.set_ylabel("Lifetime [Years]")
    ax_life.set_xlabel("Heat sink configurations")
    ax_life.set_title("Estimated lifetime")
    handles, labels = ax_life.get_legend_handles_labels()
    ax_life.legend(handles[::-1], labels[::-1],
                   handlelength=1.5, handletextpad=0.5,
                   labelspacing=0.25, borderpad=0.25, framealpha=1.0)

    plt.tight_layout()
    plt.savefig("Paper_figures/Heat_sink_combined.pdf")

#plotting_heat_sink_combined()

'''
def switch_MC(Simulation_number):

    df_diode = pd.read_parquet(f"Simulation_results_heat_sink/{Simulation_number}/df_lifetime_Diode_MC/df.parquet",engine="pyarrow")
    lifetimes_diode = df_diode["Lifetime_diode_MC"].values
    df_igbt = pd.read_parquet(f"Simulation_results_heat_sink/{Simulation_number}/df_lifetime_IGBT_MC/df.parquet",engine="pyarrow")
    lifetimes_igbt = df_igbt["Lifetime_igbt_MC"].values
    lifetimes_switch = np.minimum(lifetimes_diode, lifetimes_igbt)
    return lifetimes_switch

lifetimes_switch_MC_02U_4 = switch_MC(Simulation_number = "Simulation_4")
lifetimes_switch_MC_04U_4 = switch_MC(Simulation_number = "Simulation_8")
lifetimes_switch_MC_06U_4 = switch_MC(Simulation_number = "Simulation_12")
lifetimes_switch_MC_08U_4 = switch_MC(Simulation_number = "Simulation_16")
lifetimes_switch_MC_10U_4 = switch_MC(Simulation_number = "Simulation_20")
'''

def plotting_heat_sink_unreliability():

    def switch_MC(Simulation_number):
        df_diode = pd.read_parquet(
            f"Simulation_results_heat_sink/{Simulation_number}/df_lifetime_Diode_MC/df.parquet",
            engine="pyarrow")
        lifetimes_diode = df_diode["Lifetime_diode_MC"].values
        df_igbt = pd.read_parquet(
            f"Simulation_results_heat_sink/{Simulation_number}/df_lifetime_IGBT_MC/df.parquet",
            engine="pyarrow")
        lifetimes_igbt = df_igbt["Lifetime_igbt_MC"].values
        return np.minimum(lifetimes_diode, lifetimes_igbt)

    def weibull_fit(lifetimes_switch):
        if np.std(lifetimes_switch) < 1e-12:
            lifetimes_switch = lifetimes_switch + np.random.normal(
                loc=0.0, scale=0.0001 * lifetimes_switch[0], size=len(lifetimes_switch))
        beta, _, eta = weibull_min.fit(lifetimes_switch, floc=0.0)
        return beta, eta

    # 4 m/s simulations for each heat sink
    sims = {"02U": "Simulation_4",
            "04U": "Simulation_8",
            "06U": "Simulation_12",
            "08U": "Simulation_16",
            "10U": "Simulation_20"}

    # red (02U, hottest) -> blue (10U, coolest), sampled across the configs
    labels = list(sims.keys())
    cmap = plt.get_cmap("coolwarm_r")   # _r so it runs red -> blue
    colors = {lab: cmap(i / (len(labels) - 1)) for i, lab in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(6.4, 4.8 ))

    t_vals = np.linspace(0, 15, 1000)

    for label, sim in sims.items():
        beta, eta = weibull_fit(switch_MC(sim))
        F = weibull_min.cdf(t_vals, beta, loc=0.0, scale=eta)  # unreliability
        ax.plot(t_vals, F, color=colors[label], linewidth=2, label=label)

    # B10 reference line
    ax.axhline(y=0.10, color="black", linestyle="--", linewidth=1.2)

    ax.set_xlabel("Lifetime [years]")
    ax.set_ylabel("Unreliability [-]")
    ax.set_xlim(0, 2.75)
    ax.set_ylim(0, 0.10625)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05),
              ncol=5, columnspacing=1.0, handlelength=1.5,
              handletextpad=0.5, labelspacing=0.25, borderpad=0.25)

    plt.tight_layout(rect=[0, 0, 1, 1.025])
    plt.savefig("Paper_figures/Heat_sink_unreliability.pdf")

#plotting_heat_sink_unreliability()


def plotting_heat_sink_unreliability_custom_colors():

    # ------------------------------------------------------------------
    # EDIT COLOURS HERE — one entry per heat sink (any matplotlib colour)
    # ------------------------------------------------------------------
    line_colors = {
        "02U": "#b2182b",   # hottest
        "04U": "#ef8a62",
        "06U": "#999999",
        "08U": "#67a9cf",
        "10U": "#2166ac",   # coolest
    }

    def switch_MC(Simulation_number):
        df_diode = pd.read_parquet(
            f"Simulation_results_heat_sink/{Simulation_number}/df_lifetime_Diode_MC/df.parquet",
            engine="pyarrow")
        lifetimes_diode = df_diode["Lifetime_diode_MC"].values
        df_igbt = pd.read_parquet(
            f"Simulation_results_heat_sink/{Simulation_number}/df_lifetime_IGBT_MC/df.parquet",
            engine="pyarrow")
        lifetimes_igbt = df_igbt["Lifetime_igbt_MC"].values
        return np.minimum(lifetimes_diode, lifetimes_igbt)

    def weibull_fit(lifetimes_switch):
        if np.std(lifetimes_switch) < 1e-12:
            lifetimes_switch = lifetimes_switch + np.random.normal(
                loc=0.0, scale=0.0001 * lifetimes_switch[0], size=len(lifetimes_switch))
        beta, _, eta = weibull_min.fit(lifetimes_switch, floc=0.0)
        return beta, eta

    # 4 m/s simulations for each heat sink
    sims = {"02U": "Simulation_4",
            "04U": "Simulation_8",
            "06U": "Simulation_12",
            "08U": "Simulation_16",
            "10U": "Simulation_20"}

    fig, ax = plt.subplots(figsize=(6.4*1.15, 4.8 ))

    t_vals = np.linspace(0, 15, 1000)

    for label, sim in sims.items():
        beta, eta = weibull_fit(switch_MC(sim))
        F = weibull_min.cdf(t_vals, beta, loc=0.0, scale=eta)  # unreliability
        ax.plot(t_vals, F, color=line_colors[label], linewidth=5, label=label)

    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: "" if abs(v) < 1e-9 else f"{v:.2f}")
    )

    # B10 reference line
    ax.axhline(y=0.10, color="black", linestyle="--", linewidth=1.2)

    ax.set_xlabel("Lifetime [years]")
    ax.set_ylabel("Unreliability [-]")
    ax.set_xlim(0, 2.75)
    ax.set_ylim(0, 0.10625)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99),
              ncol=5, columnspacing=1.0, handlelength=1.5,
              handletextpad=0.5, labelspacing=0.25, borderpad=0.25)

    plt.tight_layout(rect=[0, 0, 1, 1.03])
    plt.savefig("Paper_figures/Heat_sink_unreliability.pdf")

#plotting_heat_sink_unreliability_custom_colors()


def plotting_air_speed_unreliability_custom_colors():

    # ------------------------------------------------------------------
    # EDIT COLOURS HERE — one entry per air speed (any matplotlib colour)
    # ------------------------------------------------------------------
    line_colors = {
        "1m/s": "#b2182b",   # hottest
        "2m/s": "#ef8a62",
        "3m/s": "#67a9cf",
        "4m/s": "#2166ac",   # coolest
    }

    def switch_MC(Simulation_number):
        df_diode = pd.read_parquet(
            f"Simulation_results_heat_sink/{Simulation_number}/df_lifetime_Diode_MC/df.parquet",
            engine="pyarrow")
        lifetimes_diode = df_diode["Lifetime_diode_MC"].values
        df_igbt = pd.read_parquet(
            f"Simulation_results_heat_sink/{Simulation_number}/df_lifetime_IGBT_MC/df.parquet",
            engine="pyarrow")
        lifetimes_igbt = df_igbt["Lifetime_igbt_MC"].values
        return np.minimum(lifetimes_diode, lifetimes_igbt)

    def weibull_fit(lifetimes_switch):
        if np.std(lifetimes_switch) < 1e-12:
            lifetimes_switch = lifetimes_switch + np.random.normal(
                loc=0.0, scale=0.0001 * lifetimes_switch[0], size=len(lifetimes_switch))
        beta, _, eta = weibull_min.fit(lifetimes_switch, floc=0.0)
        return beta, eta

    # 10U heat sink at different air speeds
    sims = {"1m/s": "Simulation_17",
            "2m/s": "Simulation_18",
            "3m/s": "Simulation_19",
            "4m/s": "Simulation_20"}

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    t_vals = np.linspace(0, 15, 1000)

    for label, sim in sims.items():
        beta, eta = weibull_fit(switch_MC(sim))
        F = weibull_min.cdf(t_vals, beta, loc=0.0, scale=eta)  # unreliability
        ax.plot(t_vals, F, color=line_colors[label], linewidth=5, label=label)

    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, pos: "" if abs(v) < 1e-9 else f"{v:.2f}")
    )

    # B10 reference line
    ax.axhline(y=0.10, color="black", linestyle="--", linewidth=1.2)

    ax.set_xlabel("Lifetime [years]")
    ax.set_ylabel("Unreliability [-]")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 0.10625)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.99),
              ncol=4, columnspacing=1.0, handlelength=1.5,
              handletextpad=0.5, labelspacing=0.25, borderpad=0.25)

    plt.tight_layout(rect=[0, 0, 1, 1.03])
    plt.savefig("Paper_figures/Air_speed_unreliability.png")

plotting_air_speed_unreliability_custom_colors()