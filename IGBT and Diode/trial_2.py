df_Diode["Nf_diode_eq"] = 0.0
df_Diode["lifetime_years_diode"] = 0.0
df_Diode["Nf_target_diode_MC"] = 0.0
target_index = df_Diode.index[0]
df_Diode.loc[target_index, "Nf_diode_eq"] = float(Nf_diode_eq)
df_Diode.loc[target_index, "lifetime_years_diode"] = float(lifetime_years_diode)
df_Diode.loc[target_index, "Nf_target_diode_MC"] = float(Nf_target_diode_MC)
df_Diode.to_parquet(df_lifetime_IGBT_dir / "df_Diode_final.parquet",index=False,engine="pyarrow")