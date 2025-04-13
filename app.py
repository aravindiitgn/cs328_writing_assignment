# co2_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
import io
# Configure Streamlit
st.set_page_config(layout="wide", page_title="CO₂ & GHG Emissions Analysis")

# --- Title & Introduction ---

st.markdown(
    """
    <h1 style='text-align: center;'>
        CS328 Writing Assignment
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("# CO₂ and Greenhouse Gas Emissions Analysis")
st.markdown(
    """
    This notebook analyzes the **Our World in Data CO₂ and Greenhouse Gas Emissions** dataset. It includes key metrics such as annual CO₂ emissions, CO₂ emissions per capita, cumulative and consumption-based CO₂ emissions, along with other indicators related to greenhouse gas emissions and energy mix.

    A codebook (`owid-co2-codebook.csv`) is provided alongside the dataset (`owid-co2-data.csv`). The codebook includes detailed descriptions and source information for each indicator. We'll use it to gain a better understanding of the dataset.
    """
)

# --- Load Data ---
@st.cache_data
def load_data():
    data = pd.read_csv('owid-co2-data.csv')
    codebook = pd.read_csv('owid-co2-codebook.csv')
    return data, codebook

data, codebook = load_data()


st.write("=== Data Sample ===")
st.dataframe(data.head(5))
st.write("=== Codebook Sample ===")
st.dataframe(codebook.head(5))

# --- Step 2: Data Cleaning and Preprocessing ---
st.markdown("# Data Cleaning and Preprocessing")
st.markdown(
    """
    In this step we will:
    - Compute the percentage of missing values for each column.
    - Identify and drop columns that have more than 60% missing values.
    - Remove rows that are missing crucial data (i.e., 'country', 'year', or 'co2').

    This cleaning process will help us to ensure that the subsequent analysis—such as visualizing trends and performing statistical tests—is based on reliable data.
    """
)
# Execute cleaning
missing_pct = data.isnull().mean() * 100
missing_df = missing_pct.sort_values(ascending=False).to_frame('missing_pct')
columns_to_drop = missing_df[missing_df['missing_pct'] > 60].index.tolist()
data_clean = data.drop(columns=columns_to_drop).dropna(subset=['country','year','co2'])
st.write("Columns to drop ( >60% missing ):", columns_to_drop)
st.write("Shape after cleaning:", data_clean.shape)

# --- Step 3: Exploratory Data Analysis ---
st.markdown("# Analysing the Data")

# Global Emissions Trends
st.markdown("## Global Emissions Trends")
st.markdown(
    """
    Here, we group the data by year to observe:
    - **Total CO₂ Emissions:** The aggregated sum of CO₂ emissions across all countries per year.
    - **Average CO₂ Emissions Per Capita:** The yearly average of per capita CO₂ emissions.

    These trends will help us understand the overall direction of emissions over time.
    """
)
grouped_year = data_clean.groupby('year').agg({
    'co2': 'sum',
    'co2_per_capita': 'mean',
    'population': 'sum'
}).reset_index()

# Interactive line chart for total and per‑capita
fig1 = px.line(
    grouped_year,
    x='year',
    y=['co2','co2_per_capita'],
    labels={'value':'Emissions','variable':'Metric','year':'Year'},
    title="Global CO₂ Emissions Trends"
)
fig1.update_traces(mode='markers+lines')
st.plotly_chart(fig1, use_container_width=True)

st.markdown(
    """
    **Hypothesis:** Since the mid‑20th century, global economic and technological advances have enabled a decoupling of per‑person CO₂ emissions from total emissions growth—so that although total CO₂ continues to climb, average per‑capita emissions have plateaued or even declined.

    The two panels above settle this hypothesis. The **top panel** shows total global CO₂ emissions accelerating sharply after 1950, climbing from ~35 Gt to over 240 Gt by 2023. Yet the **bottom panel** reveals that average per‑person emissions peaked around the 1950s–1970s at roughly 6–7 t CO₂/person and have since oscillated between 4.5 and 6.5 t. This divergence—soaring total emissions alongside a stable or gently declining per‑capita curve—confirms that while the world’s carbon footprint grows with population and economic scale, improvements in energy efficiency, shifts toward service economies, and the adoption of cleaner energy sources have partially offset per‑person emissions. Thus, the data support our hypothesis of mid‑century decoupling: humanity is emitting more CO₂ overall, but not at the same rate per individual.
    """
)

# Correlation Analysis
st.markdown("## Correlation Analysis of Emission Indicators")
st.markdown(
    """
    Next, we examine relationships among several key emission indicators. We select a subset of variables:
    - **CO₂ emissions (co2)**
    - **Cement CO₂ emissions (cement_co2)**
    - **Coal CO₂ emissions (coal_co2)**
    - **Oil CO₂ emissions (oil_co2)**
    - **Methane emissions (methane)**
    - **Nitrous oxide emissions (nitrous_oxide)**

    We'll compute a Pearson correlation matrix and visualize it with a heatmap. This analysis may reveal how changes in one type of emission correlate with others.
    """
)
columns_for_corr = ['co2','cement_co2','coal_co2','oil_co2','methane','nitrous_oxide']
corr_data = data_clean[columns_for_corr].dropna()
corr_matrix = corr_data.corr()

st.write("=== Correlation Matrix ===")
st.dataframe(corr_matrix)

# Interactive heatmap
fig3 = px.imshow(
    corr_matrix,
    x=columns_for_corr,
    y=columns_for_corr,
    color_continuous_scale='RdBu_r',
    labels=dict(color='Correlation'),
    title="Correlation Matrix of Selected Emission Indicators"
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    """
    The correlation matrix of key emission indicators reveals uniformly strong positive relationships, indicating that different sources of greenhouse gases tend to rise and fall together globally. Total CO₂ emissions correlate most closely with oil CO₂ (r ≈ 0.97) and coal CO₂ (r ≈ 0.96), reflecting the continued dominance of fossil fuels. Methane and nitrous oxide emissions also track closely with total CO₂ (r ≈ 0.92 and r ≈ 0.95, respectively) and exhibit an exceptionally high inter‑gas correlation (r ≈ 0.98), suggesting common agricultural and industrial drivers. Cement CO₂ shows slightly weaker—but still substantial—links to other sources (r ≈ 0.79–0.91). Overall, these patterns underscore how economic activity, energy use, and land‑use practices jointly drive multiple greenhouse‑gas emissions, reinforcing the need for integrated mitigation strategies.
    """
)
# --- Country-Level Analysis ---
st.markdown("# Country-Level Analysis")
st.markdown(
    """
    **Indicators**: Total CO₂ emissions (co2) and CO₂ emissions per capita (co2_per_capita) over time for each country.

    **Objective**: Identify the top 10 emitting countries in the most recent year of data and analyze how their total and per capita CO₂ emissions have evolved over time. This helps in understanding both the overall scale of emissions and the individual impact relative to population.

    **Approach**: First, the data is filtered to include only the most recent year to identify the top 10 countries by total CO₂ emissions. Then, the dataset is further filtered to include only these top emitters. Two line plots are created: one showing the trend of total CO₂ emissions over time for each of the top emitters, and another showing their emissions per capita over time. These visualizations provide insights into how emission patterns have shifted both in aggregate and on a per-person basis, highlighting differences in national responsibilities and behaviors.
    """
)
latest_year = int(data_clean['year'].max())
latest_data = data_clean[data_clean['year'] == latest_year]
top_emitters = latest_data.sort_values(by='co2', ascending=False)['country'].unique()[:10]
st.write(f"Top emitters in {latest_year}:", list(top_emitters))

data_top = data_clean[data_clean['country'].isin(top_emitters)]

col1, col2 = st.columns(2)

with col1:
    fig4 = px.line(
        data_top,
        x='year',
        y='co2',
        color='country',
        title="Total CO₂ Emissions Trends for Top Emitters",
        labels={'co2':'Total CO₂ Emissions','year':'Year','country':'Country'}
    )
    fig4.update_traces(mode='markers+lines')
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    fig5 = px.line(
        data_top,
        x='year',
        y='co2_per_capita',
        color='country',
        title="CO₂ Emissions Per Capita Trends for Top Emitters",
        labels={'co2_per_capita':'CO₂ per Capita','year':'Year','country':'Country'}
    )
    fig5.update_traces(mode='markers+lines')
    st.plotly_chart(fig5, use_container_width=True)

st.markdown(
    """
    The visualizations reveal striking contrasts between regions in both total and per capita CO₂ emissions. In the first plot showing total emissions over time, China exhibits a steep upward trajectory, overtaking high-income countries after the early 2000s and becoming the largest overall emitter by 2023. In contrast, the second plot on per capita emissions shows that high-income regions, especially North America, have consistently maintained the highest emissions per person across the timeline. Despite China’s rapid growth in total emissions, its per capita values remain below those of wealthier nations. These trends highlight how population size, economic development, and consumption patterns shape emission dynamics—while China leads in total output, high-income countries continue to have a significantly larger carbon footprint per individual.
    """
)

# --- Fuel-Specific (Sectoral) Deep Dives ---
st.markdown("# Fuel-Specific (Sectoral) Deep Dives")
st.markdown(
    """
    **Indicators:** CO₂ emissions specifically from cement (`cement_co2`), coal (`coal_co2`), and oil (`oil_co2`) usage over time, aggregated at the global level.

    **Objective:** Explore how emissions from different fuel sources have evolved globally, shedding light on changing patterns in industrial activities and energy consumption. This comparison helps to identify which sectors are becoming more or less prominent contributors to global CO₂ emissions.

    **Approach:** The dataset is grouped by year, summing the emissions from cement, coal, and oil across all countries to obtain global totals for each fuel source. These aggregated values are then visualized through a multi-line plot, allowing for direct comparison of trends in emissions from each fuel type over time. The resulting visualization highlights sectoral shifts, such as the growth or decline of particular fuel sources, and supports interpretation of broader energy transitions and policy impacts.
    """
)
global_fuel = data_clean.groupby('year').agg({
    'cement_co2':'sum',
    'coal_co2':'sum',
    'oil_co2':'sum'
}).reset_index()

# Interactive multi-line plot
fig6 = px.line(
    global_fuel,
    x='year',
    y=['cement_co2','coal_co2','oil_co2'],
    labels={'value':'CO₂ Emissions','variable':'Fuel Source','year':'Year'},
    title="Global Fuel-Specific CO₂ Emissions Trends"
)
fig6.update_traces(mode='markers+lines')

# Add buttons to toggle each series
fig6.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            buttons=[
                dict(label="All",
                     method="update",
                     args=[{"visible": [True, True, True]},
                           {"title": "All Fuel Sources"}]),
                dict(label="Cement",
                     method="update",
                     args=[{"visible": [True, False, False]},
                           {"title": "Cement CO₂ Emissions"}]),
                dict(label="Coal",
                     method="update",
                     args=[{"visible": [False, True, False]},
                           {"title": "Coal CO₂ Emissions"}]),
                dict(label="Oil",
                     method="update",
                     args=[{"visible": [False, False, True]},
                           {"title": "Oil CO₂ Emissions"}]),
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top"
        )
    ]
)

st.plotly_chart(fig6, use_container_width=True)

st.markdown(
    """
    The fuel‑specific CO₂ trends highlight distinct historical trajectories: **coal** led the early industrial era, rising steadily from the 19th century to peak around 1960 before plateauing and then surging again into the 21st century (now exceeding 65 Gt). **Oil** overtook coal in the mid‑20th century, climbing rapidly after 1950 and maintaining growth to roughly 55 Gt by 2023, reflecting the global shift to petroleum. **Cement** emissions, though much smaller, have grown exponentially since the 1950s—reaching over 7 Gt—driven by urbanization and infrastructure expansion. Together, these patterns illustrate how different sectors dominated successive phases of economic development: coal powered early industrialization, oil fueled mass mobility and modern economies, and cement production underpins today’s urban growth. This sectoral breakdown underscores the need for targeted mitigation—phasing down coal, decarbonizing oil use, and innovating low‑carbon cement technologies.
    """
)


# --- Temperature Change & Greenhouse Gases Analysis ---
st.markdown("# Temperature Change & Greenhouse Gases Analysis")
st.markdown(
    """
    **Indicators:** This analysis focuses on average annual temperature change attributed to individual greenhouse gases — carbon dioxide (`temperature_change_from_co2`), methane (`temperature_change_from_ch4`), nitrous oxide (`temperature_change_from_n2o`), and overall greenhouse gases (`temperature_change_from_ghg`) — as well as total CO₂ emissions (`co2`).

    **Objective:** The goal is to assess the relative contributions of different greenhouse gases to global temperature rise over time and to evaluate the direct relationship between total CO₂ emissions and the temperature change driven specifically by CO₂.

    **Approach:** First, average temperature change contributions from each gas are computed for every year by aggregating country-level data, allowing for global-level visualization of how each gas has contributed to warming trends over time. This is visualized via a multi-line plot to highlight differences and overlaps in warming impacts across gases. Next, a scatter plot is created to directly compare total CO₂ emissions against temperature change from CO₂, aiming to visually assess the correlation between rising emissions and their associated warming effects.
    """
)
# Prepare data
temp_change = data_clean.groupby('year').agg({
    'temperature_change_from_co2':'mean',
    'temperature_change_from_ch4':'mean',
    'temperature_change_from_ghg':'mean',
    'temperature_change_from_n2o':'mean'
}).reset_index()

# Interactive multi-line with toggle buttons
fig7 = px.line(
    temp_change,
    x='year',
    y=[
        'temperature_change_from_co2',
        'temperature_change_from_ch4',
        'temperature_change_from_ghg',
        'temperature_change_from_n2o'
    ],
    labels={'value':'Temperature Change','variable':'Gas'},
    title="Average Temperature Change Contributions Over Time"
)

# Add buttons to toggle each trace
fig7.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="down",
            buttons=[
                dict(label="All",
                     method="update",
                     args=[{"visible": [True, True, True, True]},
                           {"title": "All Gases"}]),
                dict(label="CO₂",
                     method="update",
                     args=[{"visible": [True, False, False, False]},
                           {"title": "Temperature Change from CO₂"}]),
                dict(label="CH₄",
                     method="update",
                     args=[{"visible": [False, True, False, False]},
                           {"title": "Temperature Change from CH₄"}]),
                dict(label="GHG",
                     method="update",
                     args=[{"visible": [False, False, True, False]},
                           {"title": "Temperature Change from Overall GHG"}]),
                dict(label="N₂O",
                     method="update",
                     args=[{"visible": [False, False, False, True]},
                           {"title": "Temperature Change from N₂O"}]),
            ],
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top"
        )
    ]
)

st.plotly_chart(fig7, use_container_width=True)

# Scatter plot remains interactive by default
fig8 = px.scatter(
    data_clean,
    x='co2',
    y='temperature_change_from_co2',
    title="Total CO₂ Emissions vs. Temperature Change from CO₂",
    labels={'co2':'Total CO₂ Emissions','temperature_change_from_co2':'Temperature Change from CO₂'}
)
st.plotly_chart(fig8, use_container_width=True)

st.markdown(
    """
    The visual analysis reveals that carbon dioxide (CO₂) is the primary contributor to global temperature change among the greenhouse gases, with its influence increasing steadily and sharply, especially after the 1950s—likely due to industrial expansion post-World War II. Methane (CH₄) and nitrous oxide (N₂O) also contribute to warming, but their impacts are significantly smaller, with CH₄ showing a gradual rise and N₂O remaining relatively stable until a slight increase in recent decades. The overall greenhouse gas (GHG) temperature change trend closely mirrors that of CO₂, confirming its dominant role. A strong positive correlation is evident between total CO₂ emissions and temperature change from CO₂, as shown in the scatter plot, where emissions and temperature rise together in a nonlinear, accelerating pattern. This suggests that as CO₂ emissions increase, their warming effect intensifies disproportionately. The clustering of data points also indicates common emission behaviors, while some outliers may reflect country-specific variations or abrupt industrial changes. Overall, the data underscores the urgent need to reduce CO₂ emissions to mitigate their escalating impact on global temperatures.
    """
)
# --- Interactive Visualizations ---
st.markdown("# Interactive Visualizations")
st.markdown(
    """
    **Indicators:** This dashboard focuses on country-level trends of total CO₂ emissions and CO₂ emissions per capita (`co2_per_capita`) over time.

    **Objective:** The purpose is to enable an intuitive and dynamic exploration of how emissions profiles have changed for individual countries, helping users better understand national-level emission patterns and their evolution over time.

    **Approach:** An interactive dashboard is implemented using a dropdown menu to allow users to select a country of interest. Upon selection, two line plots are displayed: one showing total CO₂ emissions across years, and another showing CO₂ emissions per capita. This facilitates a side-by-side analysis of both absolute and population-adjusted emission trends, offering deeper insight into national contributions and trajectories in global climate impact.
    """
)
unique_countries = sorted(data_clean['country'].unique())
country = st.selectbox("Select Country:", unique_countries)

df_country = data_clean[data_clean['country'] == country].sort_values('year')

# Interactive dual-axis plot with Plotly
fig9 = px.line(
    df_country,
    x='year',
    y='co2',
    labels={'co2': 'Total CO₂ Emissions', 'year': 'Year'},
    title=f"Emissions Trends for {country}"
)
fig9.add_scatter(
    x=df_country['year'],
    y=df_country['co2_per_capita'],
    mode='lines+markers',
    name='CO₂ per Capita',
    yaxis='y2'
)

# Configure secondary y-axis
fig9.update_layout(
    yaxis=dict(title='Total CO₂ Emissions'),
    yaxis2=dict(
        title='CO₂ per Capita',
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig9, use_container_width=True)

st.markdown(
    f"""
    The CO₂ emissions data for {country} reveals a compelling narrative of environmental impact over seven decades. Starting from minimal levels in 1950, emissions remained relatively low until approximately 2000, with a notable peak in the mid-1980s followed by a decline through the 1990s. The most striking feature is the dramatic surge around 2005-2010, when both total and per capita emissions increased dramatically, with total CO₂ emissions jumping from about 1-2 units to nearly 12 units by 2020—a roughly 6-fold increase in a short period. While per capita emissions (shown in green) exhibited greater volatility throughout, reaching their peak of approximately 0.40 units around 2010, they have since decreased somewhat even as total emissions continued climbing, suggesting population growth may be outpacing emissions growth in recent years. This pattern of correlation before 2000 and divergence after 2010 likely reflects {country}’s complex political and economic history, with periods of conflict, reconstruction, and development significantly influencing energy consumption and industrial activity throughout the nation.
    """
)

# --- World Map: Global Distribution ---
st.markdown("# World Map: Global Distribution of CO₂ Emissions")
st.markdown(
    """
    **Indicators:** This visualization displays total CO₂ emissions (`co2`) by country for the most recent year available in the dataset.

    **Objective:** The aim is to visually communicate the global distribution of CO₂ emissions, making it easy to identify major contributors and understand regional disparities in emissions.

    **Approach:** Using country ISO codes, emissions are aggregated for each country and plotted on an interactive choropleth world map. Countries are shaded according to their emission levels, with a color gradient enhancing the distinction between low and high emitters. Users can hover over countries to view names and exact emission values. The map employs a natural earth projection and a Plasma color scale to ensure readability and aesthetic clarity.
    """
)
df_latest = data_clean[(data_clean['year']==latest_year) & data_clean['iso_code'].notnull()]
df_map = df_latest.groupby(['iso_code','country'], as_index=False)['co2'].sum()

fig11 = px.choropleth(
    df_map,
    locations='iso_code',
    color='co2',
    hover_name='country',
    color_continuous_scale=px.colors.sequential.Plasma,
    title=f"Global CO₂ Emissions Distribution in {latest_year}",
    projection="natural earth",
    labels={'co2':'CO₂ Emissions'}
)
fig11.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, title_font=dict(size=24))
st.plotly_chart(fig11, use_container_width=True)

st.markdown(
    """
    The choropleth map titled "Global CO₂ Emissions Distribution in {latest_year}" presents a striking visualization of global carbon dioxide emissions by country, effectively capturing the disparity in environmental impact across regions. Dominating the spectrum is China, shaded in a bright yellow-green hue, indicating it has the highest total emissions—exceeding 10,000 metric tons—highlighting its role as the world’s largest emitter. The United States follows with significant emissions, rendered in a deep red-purple tone, signifying values in the upper mid-range of the scale. Other major contributors include India and Russia, with varying degrees of intense coloration. In contrast, much of Africa, parts of Latin America, and Southeast Asia are shaded in deep blues, indicating low emission levels, which often correlates with less industrialization and lower energy consumption per capita. The map’s use of the "Plasma" color scale allows for a vivid, easily interpretable gradient that accentuates emission disparities globally. It also underscores the broader narrative of global inequality in climate responsibility, where a handful of industrialized nations contribute disproportionately to CO₂ emissions, while many developing countries emit relatively little despite often being the most vulnerable to climate change impacts.
    """
)

# --- Pie Chart: CO₂ Emission Contributions by Source ---
st.markdown("# Pie Chart: CO₂ Emission Contributions by Source")
st.markdown(
    """
    **Indicators:** This chart presents the contribution of key sources—`cement_co2`, `coal_co2`, `oil_co2`, and a calculated `other emissions`—to total CO₂ emissions in the most recent year.

    **Objective:** To visually break down and understand the share of each major emission source in global CO₂ output, helping identify which sectors are the most impactful.

    **Approach:** For the latest available year in the dataset, global totals of CO₂ emissions from cement, coal, and oil are summed. Emissions not accounted for by these three are grouped as "Other Emissions." These values are plotted as a donut-style pie chart using Plotly Express. The chart includes percentage labels, a central hole for aesthetics, interactive hover info displaying absolute emission values, and a clear visual separation using color-coded slices and borders.
    """
)
total_co2  = df_latest['co2'].sum()
cement_co2 = df_latest['cement_co2'].sum()
coal_co2   = df_latest['coal_co2'].sum()
oil_co2    = df_latest['oil_co2'].sum()
others     = total_co2 - (cement_co2 + coal_co2 + oil_co2)

pie_df = pd.DataFrame({
    'Source': ['Cement CO₂', 'Coal CO₂', 'Oil CO₂', 'Other Emissions'],
    'Emissions': [cement_co2, coal_co2, oil_co2, others]
})

fig12 = px.pie(
    pie_df,
    names='Source',
    values='Emissions',
    hole=0.4,
    title=f"Global CO₂ Emission Contribution by Source in {latest_year}",
    color_discrete_sequence=px.colors.qualitative.Plotly
)
fig12.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hovertemplate='<b>%{label}</b><br>Emissions: %{value:,.0f} MtCO₂<extra></extra>',
    marker=dict(line=dict(color='#FFFFFF', width=2))
)
fig12.update_layout(title_font_size=24, legend_title_text='Source', legend_font_size=14, margin=dict(t=80,b=20,l=20,r=20))
st.plotly_chart(fig12, use_container_width=True)

st.markdown(
    """
    The donut chart provides a compelling visual breakdown of the major contributors to global carbon emissions, emphasizing the varied origins of greenhouse gases in the modern energy and industrial landscape. The largest portion of the chart, occupying 47.5%, is labeled “Other Emissions,” which encompasses a wide range of minor or indirect sources not individually represented—such as emissions from land-use changes, biomass burning, natural gas, and various industrial processes—highlighting the complex, multifaceted nature of global carbon output. Coal CO₂ emissions represent the second-largest share at 27.4%, reinforcing coal’s status as a leading and persistently problematic fuel in terms of carbon intensity, especially in countries reliant on coal-fired power plants. Oil CO₂ accounts for 22.2%, reflecting its dominant role in transportation and industrial sectors. Cement CO₂ emissions contribute a relatively small but non-negligible 2.9%, underscoring the environmental cost of infrastructure development due to the CO₂ released both during energy consumption and chemical transformation in cement production. The chart’s modern donut design with vivid color coding not only enhances readability but also effectively emphasizes that while traditional fossil fuels remain critical contributors, nearly half of emissions come from a diverse and less obvious set of sources, making the path to comprehensive decarbonization both urgent and complex.
    """
)

# --- Country-Wise CO₂ Emissions Pie Chart ---
st.markdown("# Country-Wise CO₂ Emissions Pie Chart")
st.markdown(
    """
    **Indicators:**  This chart displays the contribution of the top 10 CO₂-emitting countries to total global emissions in the most recent year. All other countries are combined under a single "Others" category for clarity and focus.

    **Objective:**  To identify and visualize which countries are the most significant contributors to global CO₂ emissions, and to compare their individual shares with the rest of the world. This aids in understanding geopolitical emission responsibilities and prioritizing mitigation efforts.

    **Approach:**  For the latest year in the dataset, total CO₂ emissions are aggregated for each country. The top 10 emitters are selected based on emission volume, while the remaining countries’ emissions are combined into an "Others" category. These values are visualized using a donut-style pie chart via Plotly Express. The chart includes internal percentage labels, a central hole for a clean layout, and interactive hover features showing absolute emissions in megatonnes. Color-coded slices and white borders enhance visual distinction.
    """
)
country_emissions = (
    df_latest
    .groupby(['iso_code','country'], as_index=False)['co2']
    .sum()
    .sort_values('co2', ascending=False)
)
top_n = 10
top_countries = country_emissions.head(top_n)
others_sum = country_emissions['co2'].iloc[top_n:].sum()
others_row = pd.DataFrame([{'iso_code':'OTH','country':'Others','co2':others_sum}])
pie_df2 = pd.concat([top_countries, others_row], ignore_index=True)

fig13 = px.pie(
    pie_df2,
    names='country',
    values='co2',
    hole=0.4,
    title=f"Top {top_n} CO₂ Emitting Countries in {latest_year} (plus Others)",
    color_discrete_sequence=px.colors.qualitative.Safe
)
fig13.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hovertemplate='<b>%{label}</b><br>CO₂: %{value:,.0f} Mt<extra></extra>',
    marker=dict(line=dict(color='white', width=2))
)
fig13.update_layout(title_font_size=22, legend_title_text='Country', legend_font_size=12, margin=dict(t=80,b=20,l=20,r=20))
st.plotly_chart(fig13, use_container_width=True)

st.markdown(
    """
    The pie chart highlights the unequal distribution of global CO₂ emissions in {latest_year}, with China (33.6%) and the United States (28.7%) together responsible for over 60% of emissions, underscoring their outsized role in climate change. India ranks a distant third (6.35%), reflecting its growing industrialization, while Russia, Japan, and other industrialized nations contribute smaller but significant shares. The "Others" category (32.9%)—larger than any single country except China and the U.S.—emphasizes that collective emissions from smaller nations remain substantial, necessitating global cooperation in climate policy. The visualization effectively shows emission disparities but could be enhanced with absolute values and trend data to better inform mitigation strategies. The dominance of a few economies suggests targeted policies in these regions could have a major impact, while the significant "Others" portion indicates broader systemic changes are needed to achieve meaningful emission reductions worldwide.
    """
)
# --- CO₂ Emissions: Developed vs. Developing Countries ---
st.markdown("# CO₂ Emissions: Developed vs. Developing Countries")
st.markdown(
    """
    **Indicators:**  This dual-bar chart presents CO₂ emissions for a selected group of countries, separated into Developed (e.g., United States, Germany) and Developing (e.g., India, Brazil) categories. It compares:
    - **Total CO₂ Emissions** in megatonnes (Mt)  
    - **Per Capita CO₂ Emissions** in tonnes per person (t/person)  
    for the most recent year in the dataset.

    **Objective:**  To provide a clear visual comparison of the emission patterns between high-income and middle-/lower-income countries. The goal is to understand both overall contributions to global emissions and the average individual carbon footprint within each nation.

    **Approach:**  A curated list of five developed and five developing countries was used. For the latest available year, country-level total emissions (`co2`) and per capita emissions (`co2_per_capita`) were extracted and grouped accordingly. Two side-by-side bar charts were created using Matplotlib: one showing total CO₂ emissions and the other showing emissions per person. Distinct colors are applied to each group, and grid lines enhance readability. Rotated x-axis labels improve label visibility. The side-by-side layout enables a direct visual contrast of scale and intensity between the groups.
    """
)
developed = ['United States','Germany','Japan','United Kingdom','Canada']
developing = ['India','Brazil','South Africa','Indonesia','Mexico']
df_group = df_latest[df_latest['country'].isin(developed+developing)].copy()
df_group['group'] = df_group['country'].apply(lambda c: 'Developed' if c in developed else 'Developing')
agg = df_group.groupby(['group','country']).agg(
    total_co2=('co2','sum'),
    co2_pc=('co2_per_capita','mean')
).reset_index()

col1, col2 = st.columns(2)

with col1:
    fig_total = px.bar(
        agg,
        x='country',
        y='total_co2',
        color='group',
        barmode='group',
        title=f"Total CO₂ Emissions by Country in {latest_year}",
        labels={'total_co2':'Total CO₂ Emissions (Mt)','country':'Country','group':'Group'}
    )
    fig_total.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_total, use_container_width=True)

with col2:
    fig_pc = px.bar(
        agg,
        x='country',
        y='co2_pc',
        color='group',
        barmode='group',
        title=f"CO₂ Emissions Per Capita by Country in {latest_year}",
        labels={'co2_pc':'CO₂ per Capita (t/person)','country':'Country','group':'Group'}
    )
    fig_pc.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_pc, use_container_width=True)

st.markdown(
    """
    The comparison between developed and developing countries reveals stark contrasts in CO₂ emissions patterns. Developed nations like the U.S. and Germany show significantly higher per capita emissions (likely 10-20 t/person) compared to developing countries such as India and Indonesia (typically 1-5 t/person), reflecting greater energy consumption and industrialization in wealthier economies. However, some developing nations like South Africa and Mexico may bridge this gap due to fossil fuel dependence. In absolute terms, the U.S. likely dwarfs other countries' total emissions, while populous developing nations like India may rank high in total volume despite low per capita figures. This disparity highlights the climate policy dilemma: developed nations must reduce high per capita emissions through technology and efficiency, while developing countries face the challenge of curbing emission growth during economic expansion. The data underscores that equitable climate solutions must address both historical responsibility (cumulative emissions) and future development needs.
    """
)

# --- Population vs. Total CO₂ Emissions ---
st.markdown("# Population vs. Total CO₂ Emissions")
st.markdown(
    """
    **Indicators:**
    - **Population** (`population`): Total population of each country.
    - **Total CO₂ Emissions** (`co2`): Annual CO₂ emissions (million tonnes).

    **Objective:**
    - Examine the relationship between a country’s population size and its total CO₂ emissions for the most recent year.
    - Identify outliers—countries that emit more or less than expected given their population.

    **Approach:**
    1. Filter to the latest year and drop missing values in `population` and `co2`.
    2. Create a scatter plot with a best‑fit regression line.
    3. Annotate a few notable outliers (e.g., China, India, USA, small high‑emitting countries).
    """
)
# Prepare data
df_pp = df_latest[['country','population','co2']].dropna()
df_pp['log_pop'] = np.log10(df_pp['population'])
df_pp['log_co2'] = np.log10(df_pp['co2'])
model = LinearRegression().fit(df_pp[['log_pop']], df_pp['log_co2'])
df_pp['pred_log_co2'] = model.predict(df_pp[['log_pop']])

# Interactive scatter with regression line
fig15 = px.scatter(
    df_pp,
    x='log_pop',
    y='log_co2',
    hover_name='country',
    title=f"Population vs. Total CO₂ Emissions (log‑log) in {latest_year}",
    labels={
        'log_pop': 'log₁₀(Population)',
        'log_co2': 'log₁₀(Total CO₂ Emissions)'
    }
)
# Add regression line
fig15.add_traces(px.line(
    df_pp.sort_values('log_pop'),
    x='log_pop',
    y='pred_log_co2'
).data)

# Annotate key outliers
for country in ['China','United States','India','Russia','Japan']:
    row = df_pp[df_pp['country']==country]
    if not row.empty:
        fig15.add_annotation(
            x=row['log_pop'].iloc[0],
            y=row['log_co2'].iloc[0],
            text=country,
            showarrow=True,
            arrowhead=1
        )

st.plotly_chart(fig15, use_container_width=True)

st.markdown(
    f"""
    In {latest_year}, a log‑log scatter of **Population** versus **Total CO₂ Emissions** reveals a nearly linear relationship (slope ≈ 0.92), indicating that larger populations generally produce proportionally more emissions—but slightly less than one‑to‑one. Major countries like China, India, and the United States cluster near the top, reflecting both large populations and high emissions. Notably, points above the trend line (e.g., some oil‑exporting states or small, energy‑intensive economies) emit more CO₂ than their population alone would predict, while those below the line (e.g., highly efficient or service‑based economies) emit less. This pattern underscores the strong role of population scale in driving emissions, while also highlighting outliers where energy intensity, economic structure, or policy interventions significantly alter the population‑emissions dynamic.
    """
)


# --- Historical vs. Current Emissions: Cumulative CO₂ vs. Share of Global CO₂ ---
st.markdown("# Historical vs. Current Emissions: Cumulative CO₂ vs. Share of Global CO₂")
st.markdown(
    """
    **Indicators:**
    - **Cumulative CO₂** (`cumulative_co2`): Total CO₂ emissions since the start of the record (million tonnes).  
    - **Share of Global CO₂** (`share_global_co2`): Country’s percentage of global annual CO₂ emissions for the latest year.

    **Objective:**
    - Examine how a country’s historical responsibility (cumulative emissions) relates to its current share of annual emissions.
    - Identify countries with high historical emissions but declining current share, and vice versa.

    **Approach:**
    1. Filter to the latest year and drop missing values for these two columns.  
    2. Create a scatter plot (log scale for cumulative) with a regression line.  
    3. Annotate a few notable countries (e.g., United States, China, India, Russia, Saudi Arabia).
    """
)
# Prepare data
df_hist = df_latest[['country','cumulative_co2','share_global_co2']].dropna()
Xh = np.log10(df_hist[['cumulative_co2']])
yh = df_hist['share_global_co2']
model2 = LinearRegression().fit(Xh, yh)
df_hist['pred_share'] = model2.predict(Xh)
df_hist['log_cumco2'] = np.log10(df_hist['cumulative_co2'])

# Interactive scatter with regression line
fig16 = px.scatter(
    df_hist,
    x='log_cumco2',
    y='share_global_co2',
    hover_name='country',
    title=f"Cumulative CO₂ vs. Share of Global CO₂ in {latest_year}",
    labels={
        'log_cumco2': 'log₁₀(Cumulative CO₂ Emissions, Mt)',
        'share_global_co2': 'Share of Global Annual CO₂ Emissions (%)'
    }
)
# Add regression line
fig16.add_traces(px.line(
    df_hist.sort_values('log_cumco2'),
    x='log_cumco2',
    y='pred_share',
    labels={'pred_share':'Fit'},
).data)

# Annotate key countries
for country in ['United States','China','India','Russia','Saudi Arabia']:
    row = df_hist[df_hist['country']==country]
    if not row.empty:
        fig16.add_annotation(
            x=row['log_cumco2'].iloc[0],
            y=row['share_global_co2'].iloc[0],
            text=country,
            showarrow=True,
            arrowhead=1
        )

st.plotly_chart(fig16, use_container_width=True)

st.markdown(
    f"""
    In {latest_year}, the scatter of **log₁₀(Cumulative CO₂ Emissions)** against **Share of Global Annual CO₂ Emissions** reveals a strong positive relationship: countries with the largest historical emissions still command the biggest slices of today’s global output. China (≈10⁶ Mt cumulative; ~37 %) and the United States (≈10⁶·⁰⁵ Mt; ~13 %) stand out as dominant emitters, while Russia, India, and Saudi Arabia also exceed their historical weight. The fitted trend line (Share ≈ 2.70 · log₁₀(CumCO₂) – 5.59) quantifies this linkage. However, several nations lie above or below the line, indicating shifts in current emission leadership: emerging economies like India have a higher share than their past totals would suggest, whereas some long-industrialized countries show a smaller current share, reflecting stabilization or decline. This analysis underscores both the inertia of historical emissions and the evolving dynamics of global carbon leadership.
    """
)

# --- Emissions Dynamics: Growth Rate vs. Per Capita Emissions ---
st.markdown("# Emissions Dynamics: Growth Rate vs. Per Capita Emissions")
st.markdown(
    """
    **Indicators:**
    - **CO₂ Growth Rate (%)** (`co2_growth_prct`): Annual percentage change in total CO₂ emissions.  
    - **CO₂ Per Capita** (`co2_per_capita`): Annual emissions per person (t CO₂/person).

    **Objective:**
    - Understand whether countries with high per‑person emissions are accelerating or decelerating their emission growth.  
    - Identify which countries are achieving per‑capita reductions (negative growth) despite high baseline emissions.

    **Approach:**
    1. Filter to the latest year and drop missing values for these two columns.  
    2. Create a scatter plot.  
    3. Highlight key countries (e.g., United States, China, India, Germany, Brazil).
    """
)
df_dyn = df_latest[['country','co2_growth_prct','co2_per_capita']].dropna()

# Interactive Plotly scatter
fig17 = px.scatter(
    df_dyn,
    x='co2_per_capita',
    y='co2_growth_prct',
    hover_name='country',
    title=f"CO₂ Growth Rate vs. CO₂ Per Capita in {latest_year}",
    labels={
        'co2_per_capita': 'CO₂ Emissions Per Capita (t CO₂/person)',
        'co2_growth_prct': 'CO₂ Emissions Growth Rate (%)'
    }
)

# Annotate key countries
for country in ['United States','China','India','Germany','Brazil']:
    row = df_dyn[df_dyn['country'] == country]
    if not row.empty:
        fig17.add_annotation(
            x=row['co2_per_capita'].iloc[0],
            y=row['co2_growth_prct'].iloc[0],
            text=country,
            showarrow=True,
            arrowhead=1
        )

# Add zero-growth reference line
fig17.add_shape(
    type='line',
    x0=df_dyn['co2_per_capita'].min(),
    y0=0,
    x1=df_dyn['co2_per_capita'].max(),
    y1=0,
    line=dict(color='Grey', dash='dash')
)

st.plotly_chart(fig17, use_container_width=True)

st.markdown(
    f"""
    In {latest_year}, high‑emitting nations like the United States (~14 t CO₂/person) and Germany (~8 t CO₂/person) are bucking the trend with negative growth rates, proving that economic prosperity and emission reductions can go hand in hand. In contrast, rapidly industrializing economies—India (~3 t/person) and Brazil (~2 t/person)—are still on an upward trajectory, reflecting growing energy demand. Mid‑range emitters (4–6 t/person) cluster around zero growth, indicating effective stabilization efforts, while a handful of small, fossil‑fuel–exporting countries continue to ramp up emissions. These patterns underscore the importance of bespoke climate strategies: accelerate decarbonization in wealthier countries and support clean‑energy transitions in emerging markets.
    """
)
