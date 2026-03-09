import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

plt.rcParams.update({"font.size": 15, "font.family": "Times New Roman", "axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

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

    fig, ax = plt.subplots(figsize=(6,4))



    # ---- overlapping bars ----


    ax.bar(x, v1, label="1 m/s", color="tab:red")
    ax.bar(x, v2, label="2 m/s", color="tab:green")
    ax.bar(x, v3, label="3 m/s", color="tab:orange")
    ax.bar(x, v4, label="4 m/s", color="tab:blue")

    ax.axhline(y=175, color='black', linestyle=':', linewidth=1.5, label="175 °C limit")

    # ---- formatting ----
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Temperature [°C]")
    ax.set_xlabel("Heat sink configurations")
    ax.set_ylim(0,470)
    ax.legend()

    plt.tight_layout()
    plt.savefig("Paper_figures/Heat_sink_case_study_2.pdf")


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

    fig, ax = plt.subplots(figsize=(6,4))

    # ---- overlapping bars ----

    ax.bar(x, v4, label="4 m/s", color="tab:blue")
    ax.bar(x, v3, label="3 m/s", color="tab:orange")
    ax.bar(x, v2, label="2 m/s", color="tab:green")
    ax.bar(x, v1, label="1 m/s", color="tab:red")

    # ---- formatting ----
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Life time [Years]")
    ax.set_xlabel("Heat sink type")
    #ax.set_ylim(0,470)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    plt.tight_layout()
    plt.savefig("Paper_figures/Heat_sink_life.pdf")

plotting_heat_sink_life()
plotting_heat_sink_temp_limit()