import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import altair as alt
import plotly.express as px



# Page configuration
st.set_page_config(
    page_title="Child Healthcare Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
    )

alt.themes.enable("dark")


# Title
st.title(" 📈 Child Healthcare Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Define the file path
file_path = "data/child_masterpy.csv"

# Read and process data
child_data = pd.read_csv(file_path, parse_dates=['date_of_birth', 'date_of_death', 'checkup_date'])

# Function to create KPIs with improved presentation
def kpi(label, value):
    # Determine background and text color based on theme (light/dark)
    if st.get_option("theme.base") == "light":
        bg_color = "#f0f0f0"  # Light gray background for light mode
        text_color = "#000000"  # Black text for light mode
    else:
        bg_color = "#333333"  # Dark gray background for dark mode
        text_color = "#ffffff"  # White text for dark mode
    
    st.markdown(
        f'<div style="background-color: {bg_color}; color: {text_color}; border-radius: 10px; padding: 1rem; margin: 1rem; text-align: center;"><b>{label}</b><br>{value}</div>',
        unsafe_allow_html=True
    )


# General Statistics
total_records = len(child_data)
total_children = child_data['child_id'].nunique()

# IMR rate
live_births_count = len(child_data)

# Filter out records where both date of birth and date of death are available
valid_records = child_data[(child_data['date_of_birth'].notnull()) & (child_data['date_of_death'].notnull())].copy()

# Assuming you're reading data from a CSV file
valid_records = pd.read_csv('data/child_masterpy.csv', parse_dates=['date_of_birth', 'date_of_death','checkup_date'])

# Now perform your calculations on datetime columns
valid_records['age_at_death'] = (valid_records['date_of_death'] - valid_records['date_of_birth']).dt.days


# Calculate age at death for each infant
valid_records['age_at_death'] = (valid_records['date_of_death'] - valid_records['date_of_birth']).dt.days

 # Calculate the number of infant deaths (those who died within 1 year)
infant_deaths_count = len(valid_records[valid_records['age_at_death'] <= 365])

print(infant_deaths_count)
# Calculate the infant mortality rate (IMR) per 1000 live births
infant_mortality_rate = (infant_deaths_count / live_births_count) * 1000

# Neonatal morality rate
# Filter out records where age at death is less than or equal to 28 days (neonatal period)
neonatal_deaths_count = len(valid_records[valid_records['age_at_death'] <= 28])
print(neonatal_deaths_count)

# Calculate the neonatal mortality rate (NMR) per 1000 live births
neonatal_mortality_rate = (neonatal_deaths_count / live_births_count) * 1000

# Under 5 mortality rate
# Calculate the number of deaths under 5 years of age
under_5_deaths_count = len(valid_records[valid_records['age_at_death'] < 1825])  # 1825 days = 5 years
print(under_5_deaths_count)

# Calculate the under-5 mortality rate (U5MR) per 1000 live births
under_5_mortality_rate = (under_5_deaths_count / live_births_count) * 1000

print(f"Under-5 Mortality Rate (U5MR): {under_5_mortality_rate:.2f} per 1000 live births")

# Health Checkups
checkup_records = child_data.dropna(subset=['checkup_id'])
num_checkups = checkup_records['child_id'].nunique()


# Disease Prevalence
disease_columns = ['has_ari', 'has_diarrhea', 'convulsion', 'fever', 'jaundice', 
                   'unable_to_suck', 'vomiting', 'rash_boils', 'swollen_crusted_eyes']

# Define disease names with custom labels
disease_labels = {
    'has_ari': 'Respiratory Infection',
    'has_diarrhea': 'Diarrhea',
    'convulsion': 'Convulsion',
    'fever': 'Fever',
    'jaundice': 'Jaundice',
    'unable_to_suck': 'Unable to Suck',
    'vomiting': 'Vomiting',
    'rash_boils': 'Rash/Boils',
    'swollen_crusted_eyes': 'Swollen/Crusted Eyes'
}

disease_columns = list(disease_labels.keys())

# Group disease counts by gender
disease_counts_by_gender = checkup_records.groupby('gender')[disease_columns].sum()

# Sort genders by total disease count
sorted_genders = disease_counts_by_gender.sum(axis=1).sort_values(ascending=False).index

# Create lists to store data for male and female separately
male_counts = []
female_counts = []

# Calculate total counts for each disease across both genders
total_counts = {}
for disease in disease_columns:
    total_counts[disease] = disease_counts_by_gender.loc['M', disease] + disease_counts_by_gender.loc['F', disease]

# Sort diseases based on total counts
sorted_diseases = sorted(total_counts.keys(), key=lambda x: total_counts[x], reverse=True)

# Loop through sorted diseases and get counts for male and female
for disease in sorted_diseases:
    male_counts.append(disease_counts_by_gender.loc['M', disease])
    female_counts.append(disease_counts_by_gender.loc['F', disease])

# Create a figure with separate bars for male and female counts
male_color = 'skyblue'
female_color = 'salmon'

# Create a figure with separate bars for male and female counts
fig = go.Figure()

fig.add_trace(go.Bar(y=[disease_labels[disease] for disease in sorted_diseases], 
                     x=male_counts, 
                     orientation='h', 
                     name='Male', 
                     showlegend=True,
                     marker=dict(color=male_color)))  # Assigning color for male bars

fig.add_trace(go.Bar(y=[disease_labels[disease] for disease in sorted_diseases], 
                     x=female_counts, 
                     orientation='h', 
                     name='Female', 
                     showlegend=True,
                     marker=dict(color=female_color)))  # Assigning color for female bars

# Update layout
fig.update_layout(title='Disease Prevalence by Gender',
                  xaxis_title='Number of Occurrences',
                  yaxis_title='Disease',
                  barmode='stack',
                  height=600)
col1,col2,col3 = st.columns((2,5.5, 3), gap='medium')

# Child Healthcare Overview
with col1:
    st.markdown("#### Child Healthcare Overview")
    kpi("Total Children", total_children)
    kpi("Children with Checkups", num_checkups)
    hbnc_records = child_data.dropna(subset=['hbncs_id'])
    num_hbnc_done_children = hbnc_records['child_id'].nunique()
    kpi("Atleast 1 hbnc done", num_hbnc_done_children)

with col1:
    kpi("Infant Mortality Rate", f"{infant_mortality_rate:.2f} % ")
    
with col1:
    kpi("Neonatal Mortality Rate", f"{neonatal_mortality_rate:.2f} %")
    
with col1:
    kpi("Under-5 Mortality Rate", f"{under_5_mortality_rate:.2f} %")
    
    
with col1:
    # Calculate sex ratio
    num_male = child_data[child_data['gender'] == 'M']['child_id'].nunique()
    num_female = child_data[child_data['gender'] == 'F']['child_id'].nunique()
    sex_ratio = num_male / num_female

    # Create labels and values for the pie chart
    labels = ['Male', 'Female']
    values = [num_male, num_female]

    # Define colors for male and female slices
    colors = ['#1f77b4', '#ff69b4']  # Blue for male, pink for female

    # Plot sex ratio as a donut chart
    fig_sex_ratio = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
    fig_sex_ratio.update_traces(marker=dict(colors=colors))  
    st.markdown("#### Sex Ratio")

    #fig_sex_ratio.update_layout(title='Sex Ratio')

    # Display the chart
    st.plotly_chart(fig_sex_ratio, use_container_width=True)

        
# Prevalence of Common Childhood Diseases
with col3:
    st.markdown("#### Prevalence of Common Childhood Diseases")
    st.plotly_chart(fig , use_container_width=True)

with col3:
    completed_hbncs = child_data.dropna(subset=['hbncs_id'])
    completed_hbncs_count = completed_hbncs.groupby('child_id')['hbncs_id'].nunique()

    hbnc_counts = {}
    for hbnc_type in range(1, 8):
        children_with_current_hbnc = completed_hbncs_count[completed_hbncs_count == hbnc_type].index
        
        num_children_with_current_hbnc = len(children_with_current_hbnc)
        
        hbnc_counts[f"HBNC Type {hbnc_type}"] = num_children_with_current_hbnc

    # Extract HBNC types and counts
    hbnc_types = list(hbnc_counts.keys())
    hbnc_type_counts = list(hbnc_counts.values())

    # Define colors for HBNC types
    colors = ['#008000', '#1F4D2B', '#2F4F4F', '#3D9140', '#6B8E23', '#7CFC00', '#9ACD32']

    # Function to generate shades of green
    def generate_shades_of_green(palette):
        shades_of_green = []
        for color in palette:
            # Convert hex color to RGB
            base_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            # Generate seven shades of green by manipulating the green component
            for i in range(1, 8):
                # Decrease brightness for darker shades, increase for lighter shades
                green_value = max(0, min(255, base_rgb[2] + 20 * i))
                shade = "#{:02x}{:02x}{:02x}".format(base_rgb[0], green_value, base_rgb[2])
                shades_of_green.append(shade)
        return shades_of_green

    # Create a single stacked bar graph
    fig = go.Figure()
    # Generate shades of green for each HBNC type
    shades_of_green = generate_shades_of_green(colors)

    # Add bars for each HBNC type
    for i in range(len(hbnc_types)):
        hover_text = f"Type: {hbnc_types[i]}<br>Count: {hbnc_type_counts[i]}"
        fig.add_trace(go.Bar(y=[hbnc_type_counts[i]], name=hbnc_types[i], marker_color=shades_of_green[i],
                            base=[sum(hbnc_type_counts[:i])], width=0.5,
                            hovertemplate=hover_text))

    # Update layout
    fig.update_layout(
        title="No. of Children who Completed Each HBNC Type",
        xaxis_title="HBNC Type",
        yaxis_title="Number of Children",
        barmode="stack",
        xaxis=dict(
            tickvals=list(range(1, len(hbnc_types)+1)),
            ticktext=hbnc_types
        ),
        width=800,
        height=600
    )

    # Display the graph using Streamlit
    st.plotly_chart(fig, use_container_width=True)

with col2:
    
    # Plot Total Disease Count by Region
    st.markdown("#### Total Disease Count by Region")

    # Group data by district name and calculate disease counts
    disease_counts_by_region = checkup_records.groupby('district_name')[disease_columns].sum()
    disease_counts_by_region['total_disease_count'] = disease_counts_by_region.sum(axis=1)

    # Sort regions by total disease count
    sorted_regions = disease_counts_by_region['total_disease_count'].sort_values(ascending=False)

    # Filter to display top N regions
    top_n = 5
    top_n_regions = sorted_regions.head(top_n)

    # Create dictionary to store data for top N regions
    data = {}
    for region_id, total_disease_count in top_n_regions.items():
        data[str(region_id)] = total_disease_count

    # Plot total disease count by region
    fig = go.Figure(data=go.Bar(x=list(data.keys()), y=list(data.values()), marker_color='blue'))
    fig.update_layout(
        xaxis_title='District Name',
        yaxis_title='Total Disease Count',
        width=800,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
 
    # Plot Total Disease Count by Region as a Heatmap
    st.markdown("#### Total Disease Count by Region (Heatmap)")

    # Group data by district name and calculate disease counts
    disease_counts_by_region = checkup_records.groupby('district_name')[disease_columns].sum()
    disease_counts_by_region['total_disease_count'] = disease_counts_by_region.sum(axis=1)

    # Sort regions by total disease count
    sorted_regions = disease_counts_by_region['total_disease_count'].sort_values(ascending=False)

    # Filter to display top N regions
    top_n = 5
    top_n_regions = sorted_regions.head(top_n)

    # Create heatmap data
    heatmap_data = pd.DataFrame(index=top_n_regions.index, columns=disease_labels)
    for column in disease_labels:
        heatmap_data[column] = top_n_regions.index.map(disease_counts_by_region[column])

    # Plot heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='blues',
        reversescale=False))

    fig_heatmap.update_layout(
        #title='Total Disease Count by Region',
        xaxis_title='Disease Type',
        yaxis_title='District Name',
        width=800,
        height=500)

    st.plotly_chart(fig_heatmap, use_container_width=True)


# Health Analysis
st.markdown("#### Health Analysis")

checkup_records = child_data[child_data['checkup_id'].notnull()]
grouped = checkup_records.groupby('child_id')

# Select the first row for each group (latest checkup)
latest_checkups = grouped.head(1).copy()

# Convert weight from grams to kilograms
latest_checkups['weight_kg'] = latest_checkups['weight_in_grams'] / 1000

# Convert height from centimeters to meters
latest_checkups['height_m'] = latest_checkups['height_cm'] / 100

# Calculate BMI
latest_checkups['bmi'] = latest_checkups['weight_kg'] / (latest_checkups['height_m'] ** 2)

# Define function to classify BMI
def classify_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    else:
        return 'Normal'

# Classify BMI
latest_checkups['bmi_category'] = latest_checkups['bmi'].apply(classify_bmi)

# Calculate median and standard deviation of height
median_height = latest_checkups['height_cm'].median()
std_height = latest_checkups['height_cm'].std()

# Calculate HAZ using the provided formula
latest_checkups['haz'] = (latest_checkups['height_cm'] - median_height) / std_height

# Define function to classify HAZ
def classify_haz(haz):
    if haz < 0:
        return 'Stunted'
    else:
        return 'Normal'

# Classify HAZ
latest_checkups['haz_category'] = latest_checkups['haz'].apply(classify_haz)

stunted_children = latest_checkups[latest_checkups['haz_category'] == 'Stunted'].shape[0]
underweight_children = latest_checkups[latest_checkups['bmi_category'] == 'Underweight'].shape[0]

# Calculate the ratio of stunted children to underweight children
ratio_stunted_to_underweight = stunted_children / underweight_children

col1, col2, col3 = st.columns(3)

# Plotting BMI categories as a donut pie chart
bmi_counts = latest_checkups['bmi_category'].value_counts()
fig1 = go.Figure(data=[go.Pie(labels=bmi_counts.index, values=bmi_counts.values, hole=0.4)])
fig1.update_layout(title='Normal vs Underweight')

# Plotting HAZ categories as a donut
haz_counts = latest_checkups['haz_category'].value_counts()
fig2 = go.Figure(data=[go.Pie(labels=haz_counts.index, values=haz_counts.values, hole=0.4)])
fig2.update_layout(title='Normal vs stunted')

# Define labels and values for the pie chart
labels = ['Stunted Children', 'Underweight Children']
values = [stunted_children, underweight_children]

# Create a pie chart figure
fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
fig3.update_layout(title='Ratio of Stunted Children to Underweight Children')

with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    st.plotly_chart(fig3, use_container_width=True)
    
valid_records = child_data[
    (child_data['height_cm'].notnull()) & 
    (child_data['weight_in_grams'].notnull()) & 
    (child_data['height_cm'] != 0) & 
    (child_data['weight_in_grams'] != 0) &
    (child_data['height_cm'] != 30000) & 
    (child_data['weight_in_grams'] != 30000)
]

# Create a scatter plot
fig = px.scatter(valid_records, 
                 x='height_cm', 
                 y='weight_in_grams', 
                 trendline="ols", 
                 trendline_color_override="red",  # Specify the color for the trendline
                 labels={'height_cm': 'Height (cm)', 'weight_in_grams': 'Weight (grams)'}, 
                 title='Height vs. Weight Scatter Plot with Trendline')

# Render the Plotly figure using Streamlit
st.plotly_chart(fig, use_container_width=True)


# Step 2: Trend Analysis
# Group the data by checkup_date and count the number of checkups per time unit (e.g., per month)
checkup_counts = child_data.groupby(child_data['checkup_date'].dt.to_period('M')).size().reset_index(name='count')

# Convert Period index to string for plotting with Plotly
checkup_counts['checkup_date'] = checkup_counts['checkup_date'].astype(str)

# Step 3: Visualization with Plotly
fig = go.Figure()

# Add trace for number of checkups over time
fig.add_trace(go.Scatter(x=checkup_counts['checkup_date'], y=checkup_counts['count'],
                         mode='lines+markers',
                         name='Number of Checkups',
                         marker=dict(color='blue', size=8),
                         line=dict(color='blue', width=2)))

# Update layout
fig.update_layout(title='Trend of Child Checkups Over Time',
                  xaxis_title='Checkup Date',
                  yaxis_title='Number of Checkups',
                  xaxis=dict(showgrid=True, zeroline=True),
                  yaxis=dict(showgrid=True, zeroline=True))

# Streamlit UI
st.plotly_chart(fig)
