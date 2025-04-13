import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("CO₂ and Greenhouse Gas Emissions Analysis")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("owid-co2-data.csv")
    codebook = pd.read_csv("owid-co2-codebook.csv")
    return data, codebook

data, codebook = load_data()

st.header("Step 1: Data Sample")
st.markdown("**First few rows of the dataset and codebook**")
st.dataframe(data.head())
st.dataframe(codebook.head())

st.header("Step 2: Data Exploration and Preliminary Cleaning")
st.markdown("""
- `.info()` for data types & non-null counts  
- `.describe()` for summary statistics  
- Missing values per column  
""")
buf = []
data.info(buf=buf)
st.text("\n".join(buf))
st.write(data.describe().head())
st.write(data.isnull().sum().head())

st.header("Step 3: Data Cleaning and Preprocessing")
st.markdown("""
- Drop columns with >60% missing  
- Drop rows missing `country`, `year`, or `co2`  
""")
missing_pct = data.isnull().mean() * 100
missing_df = missing_pct.sort_values(ascending=False).to_frame("missing_pct")
st.write("Top 10 columns by missing %")
st.write(missing_df.head(10))
threshold = 60.0
cols_to_drop = missing_df[missing_df["missing_pct"] > threshold].index.tolist()
st.write(f"Dropping columns >{threshold}% missing:", cols_to_drop)
data_clean = data.drop(columns=cols_to_drop).dropna(subset=["country","year","co2"])
st.write("Cleaned data shape:", data_clean.shape)

st.header("Step 4.1: Global Emissions Trends")
grouped = data_clean.groupby("year").agg({
    "co2":"sum","co2_per_capita":"mean","population":"sum"
}).reset_index()

fig1, ax1 = plt.subplots()
ax1.plot(grouped["year"], grouped["co2"], marker="o", label="Total CO₂ Emissions")
ax1.set_xlabel("Year"); ax1.set_ylabel("Total CO₂ Emissions")
ax1.legend(); ax1.grid(True)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.plot(grouped["year"], grouped["co2_per_capita"], marker="o", label="Average CO₂ per Capita")
ax2.set_xlabel("Year"); ax2.set_ylabel("CO₂ per Capita")
ax2.legend(); ax2.grid(True)
st.pyplot(fig2)

st.markdown("""
**Hypothesis**: Mid‑20th century decoupling: total emissions ↑ sharply post‑1950, per‑capita plateaued.
""")

st.header("Step 4.2: Correlation Analysis of Emission Indicators")
cols = ['co2','cement_co2','coal_co2','oil_co2','methane','nitrous_oxide']
corr_data = data_clean[cols].dropna()
corr = corr_data.corr()
st.write(corr)

fig3, ax3 = plt.subplots()
im = ax3.imshow(corr, cmap="coolwarm", interpolation="none", aspect="auto")
fig3.colorbar(im, ax=ax3)
ax3.set_xticks(range(len(cols))); ax3.set_xticklabels(cols, rotation=45)
ax3.set_yticks(range(len(cols))); ax3.set_yticklabels(cols)
ax3.set_title("Correlation Matrix")
st.pyplot(fig3)

st.header("Country-Level Analysis")
latest = int(data_clean["year"].max())
st.markdown(f"**Top 10 emitters in {latest}**")
latest_df = data_clean[data_clean["year"]==latest]
top10 = latest_df.sort_values("co2",ascending=False)["country"].unique()[:10]
st.write(list(top10))

data_top = data_clean[data_clean["country"].isin(top10)]
fig4, ax4 = plt.subplots(figsize=(8,5))
for c in top10:
    dfc = data_top[data_top["country"]==c]
    ax4.plot(dfc["year"], dfc["co2"], marker="o", label=c)
ax4.set_title("Total CO₂ Emissions Over Time")
ax4.legend(); ax4.grid(True)
st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(8,5))
for c in top10:
    dfc = data_top[data_top["country"]==c]
    ax5.plot(dfc["year"], dfc["co2_per_capita"], marker="o", label=c)
ax5.set_title("CO₂ Emissions Per Capita Over Time")
ax5.legend(); ax5.grid(True)
st.pyplot(fig5)

st.header("Fuel‑Specific (Sectoral) Deep Dives")
fuel = data_clean.groupby("year").agg({
    "cement_co2":"sum","coal_co2":"sum","oil_co2":"sum"
}).reset_index()
fig6, ax6 = plt.subplots()
ax6.plot(fuel["year"], fuel["cement_co2"], marker="o", label="Cement")
ax6.plot(fuel["year"], fuel["coal_co2"], marker="o", label="Coal")
ax6.plot(fuel["year"], fuel["oil_co2"], marker="o", label="Oil")
ax6.set_title("Global Fuel‑Specific CO₂ Emissions")
ax6.legend(); ax6.grid(True)
st.pyplot(fig6)

st.header("Temperature Change & Greenhouse Gases Analysis")
temp = data_clean.groupby("year").agg({
    "temperature_change_from_co2":"mean",
    "temperature_change_from_ch4":"mean",
    "temperature_change_from_n2o":"mean",
    "temperature_change_from_ghg":"mean"
}).reset_index()
fig7, ax7 = plt.subplots()
ax7.plot(temp["year"], temp["temperature_change_from_co2"], marker="o", label="CO₂")
ax7.plot(temp["year"], temp["temperature_change_from_ch4"], marker="o", label="CH₄")
ax7.plot(temp["year"], temp["temperature_change_from_n2o"], marker="o", label="N₂O")
ax7.plot(temp["year"], temp["temperature_change_from_ghg"], marker="o", label="Overall GHG")
ax7.set_title("Avg. Temperature Change Contributions")
ax7.legend(); ax7.grid(True)
st.pyplot(fig7)

fig8 = px.scatter(
    data_clean, x="co2", y="temperature_change_from_co2",
    opacity=0.5, title="CO₂ Emissions vs. Temp Change from CO₂"
)
st.plotly_chart(fig8, use_container_width=True)

st.header("Interactive Country Selector")
country = st.selectbox("Select country", sorted(data_clean["country"].unique()))
dfc = data_clean[data_clean["country"]==country].sort_values("year")
fig9, ax9 = plt.subplots()
ax9.plot(dfc["year"], dfc["co2"], marker="o", label="Total CO₂")
ax9.set_xlabel("Year"); ax9.set_ylabel("Total CO₂")
ax9.legend(); ax9.grid(True)
st.pyplot(fig9)

fig10, ax10 = plt.subplots()
ax10.plot(dfc["year"], dfc["co2_per_capita"], marker="o", label="CO₂ per Capita")
ax10.set_xlabel("Year"); ax10.set_ylabel("CO₂ per Capita")
ax10.legend(); ax10.grid(True)
st.pyplot(fig10)

st.header("Choropleth Map of CO₂ Emissions")
df_map = latest_df.dropna(subset=["iso_code"]).groupby(["iso_code","country"], as_index=False)["co2"].sum()
fig11 = px.choropleth(
    df_map, locations="iso_code", color="co2", hover_name="country",
    color_continuous_scale="Plasma", projection="natural earth",
    title=f"Global CO₂ Emissions in {latest}"
)
st.plotly_chart(fig11, use_container_width=True)

st.header("Pie Chart: Emission by Source")
cement = latest_df["cement_co2"].sum()
coal = latest_df["coal_co2"].sum()
oil = latest_df["oil_co2"].sum()
others = latest_df["co2"].sum() - (cement+coal+oil)
pie_df = pd.DataFrame({
    "Source":["Cement","Coal","Oil","Other"],
    "Emissions":[cement,coal,oil,others]
})
fig12 = px.pie(
    pie_df, names="Source", values="Emissions", hole=0.4,
    title=f"CO₂ Emission by Source in {latest}"
)
st.plotly_chart(fig12, use_container_width=True)

st.header("Pie Chart: Top 10 Countries")
cty = df_map.sort_values("co2",ascending=False)
top_cty = cty.head(10)
others_cty = pd.DataFrame([{
    "iso_code":"OTH","country":"Others",
    "co2":cty["co2"].iloc[10:].sum()
}])
pie2 = pd.concat([top_cty.rename(columns={"country":"Source","co2":"Emissions"})[["Source","Emissions"]], 
                  others_cty.rename(columns={"country":"Source","co2":"Emissions"})[["Source","Emissions"]]])
fig13 = px.pie(pie2, names="Source", values="Emissions", hole=0.4,
               title=f"Top 10 Emitters in {latest} (plus Others)")
st.plotly_chart(fig13, use_container_width=True)

st.header("Developed vs Developing Comparison")
dev = ['United States','Germany','Japan','United Kingdom','Canada']
dvg = ['India','Brazil','South Africa','Indonesia','Mexico']
df_grp = latest_df[latest_df["country"].isin(dev+dvg)].copy()
df_grp["group"] = df_grp["country"].apply(lambda c: "Developed" if c in dev else "Developing")
agg = df_grp.groupby(["group","country"]).agg(
    total_co2=("co2","sum"),
    co2_pc=("co2_per_capita","mean")
).reset_index()
fig14, (ax14,ax15) = plt.subplots(1,2,figsize=(12,4))
for grp in ["Developed","Developing"]:
    sub = agg[agg["group"]==grp]
    ax14.bar(sub["country"], sub["total_co2"], label=grp)
    ax15.bar(sub["country"], sub["co2_pc"], label=grp)
ax14.set_title("Total CO₂"); ax15.set_title("CO₂ per Capita")
for ax in (ax14,ax15):
    ax.tick_params(axis="x",rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
ax14.legend(); ax15.legend()
st.pyplot(fig14)

st.header("Population vs Total CO₂ (log‑log)")
df_pp = data_clean[
    (data_clean["year"]==latest)&
    data_clean["population"].notnull()
][["country","population","co2"]]
X = np.log10(df_pp[["population"]])
y = np.log10(df_pp["co2"])
model = LinearRegression().fit(X,y)
y_pred = model.predict(X)
fig15, ax15 = plt.subplots()
ax15.scatter(np.log10(df_pp["population"]), np.log10(df_pp["co2"]), alpha=0.7)
ax15.plot(np.log10(df_pp["population"]), y_pred, color="red",
          label=f"log₁₀(CO₂)={model.coef_[0]:.2f}·log₁₀(pop)+{model.intercept_:.2f}")
for c in ["China","United States","India","Russia","Japan"]:
    r = df_pp[df_pp["country"]==c]
    if not r.empty:
        ax15.text(np.log10(r["population"].iloc[0]), np.log10(r["co2"].iloc[0]), c)
ax15.set_xlabel("log₁₀(Population)"); ax15.set_ylabel("log₁₀(CO₂)")
ax15.legend(); ax15.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig15)

st.header("Cumulative CO₂ vs Share of Global CO₂")
df_hist = data_clean[
    (data_clean["year"]==latest)&
    data_clean["cumulative_co2"].notnull()&
    data_clean["share_global_co2"].notnull()
][["country","cumulative_co2","share_global_co2"]]
X2 = np.log10(df_hist[["cumulative_co2"]])
y2 = df_hist["share_global_co2"]
model2 = LinearRegression().fit(X2,y2)
y2p = model2.predict(X2)
fig16, ax16 = plt.subplots()
ax16.scatter(np.log10(df_hist["cumulative_co2"]), df_hist["share_global_co2"], alpha=0.7)
ax16.plot(np.log10(df_hist["cumulative_co2"]), y2p, color="red",
          label=f"Share={model2.coef_[0]:.2f}·log₁₀(CumCO₂)+{model2.intercept_:.2f}")
for c in ["United States","China","India","Russia","Saudi Arabia"]:
    r = df_hist[df_hist["country"]==c]
    if not r.empty:
        ax16.text(np.log10(r["cumulative_co2"].iloc[0]), r["share_global_co2"].iloc[0], c)
ax16.set_xlabel("log₁₀(Cumulative CO₂)"); ax16.set_ylabel("Share Global CO₂ (%)")
ax16.legend(); ax16.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig16)

st.header("CO₂ Growth Rate vs Per Capita")
df_dyn = data_clean[
    (data_clean["year"]==latest)&
    data_clean["co2_growth_prct"].notnull()
][["country","co2_growth_prct","co2_per_capita"]]
fig17, ax17 = plt.subplots()
ax17.scatter(df_dyn["co2_per_capita"], df_dyn["co2_growth_prct"], alpha=0.7)
for c in ["United States","China","India","Germany","Brazil"]:
    r = df_dyn[df_dyn["country"]==c]
    if not r.empty:
        ax17.text(r["co2_per_capita"].iloc[0], r["co2_growth_prct"].iloc[0], c)
ax17.axhline(0, color="grey", linestyle="--")
ax17.set_xlabel("CO₂ per Capita"); ax17.set_ylabel("Growth Rate (%)")
ax17.set_title(f"Growth vs Per Capita in {latest}")
ax17.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig17)
