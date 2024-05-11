import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import pandas_bokeh
import missingno

def create_correlation_chart(corr_df): ## Create Correlation Chart using Matplotlib
    fig = plt.figure(figsize=(15,15 ))
    plt.imshow(corr_df.values, cmap="Blues")
    plt.xticks(range(corr_df.shape[0]), corr_df.columns, rotation=90, fontsize=15)
    plt.yticks(range(corr_df.shape[0]), corr_df.columns, fontsize=15)
    plt.colorbar()

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[0]):
            plt.text(i,j, "{:.2f}".format(corr_df.values[i, j]), color="red", ha="center", fontsize=14, fontweight="bold")

    return fig

def create_missing_values_bar(df):
    missing_fig = plt.figure(figsize=(10,5))
    ax = missing_fig.add_subplot(111)
    missingno.bar(df, figsize=(10,5), fontsize=13, ax=ax)

    return missing_fig

def find_cat_cont_columns(df): ## Logic to Separate Continuous & Categorical Columns
    cont_columns, cat_columns = [],[]
    for col in df.columns:        
        if len(df[col].unique()) <= 25 or df[col].dtype == np.object_: ## If less than 25 unique values
            cat_columns.append(col.strip())
        else:
            cont_columns.append(col.strip())
    return cont_columns, cat_columns

### Web App / Dashboard Code
st.set_page_config(page_icon=":bar_chart:", page_title="Market Entry Strategy Optimization Tool")

st.title("Market Entry Strategy Optimization Tool")
st.caption("Upload CSV file to see various Charts related to EDA. Please upload file that has both continuous columns and categorical columns. Once you upload file, various charts, widgets and basic stats will be displayed.", unsafe_allow_html=True)
upload = st.file_uploader(label="Upload File Here:", type=["csv"])

if upload: ## File as Bytes
    df = pd.read_csv(upload)

    tab1, tab2, tab3 = st.tabs(["Dataset Overview :clipboard:", "Individual Column Stats :bar_chart:", "Explore Relation Between Features :chart:"])

    with tab1: ## Dataset Overview Tab        
        st.subheader("1. Dataset")
        st.write(df)

        cont_columns, cat_columns = find_cat_cont_columns(df)

        st.subheader("2. Dataset Overview")
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Rows", df.shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Duplicates", df.shape[0] - df.drop_duplicates().shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Features", df.shape[1]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Categorical Columns", len(cat_columns)), unsafe_allow_html=True)
        st.write(cat_columns)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Continuous Columns", len(cont_columns)), unsafe_allow_html=True)
        st.write(cont_columns)
        
        corr_df = df[cont_columns].corr()
        corr_fig = create_correlation_chart(corr_df)
        
        st.subheader("3. Correlation Chart")
        st.pyplot(corr_fig, use_container_width=True)

        st.subheader("4. Missing Values Distribution")
        missing_fig = create_missing_values_bar(df)
        st.pyplot(missing_fig, use_container_width=True)

    with tab2: ## Individual Column Stats
        df_descr = df.describe()
        st.subheader("Analyze Individual Feature Distribution")

        st.markdown("#### 1. Understand Continuous Feature")        
        feature = st.selectbox(label="Select Continuous Feature", options=cont_columns, index=0)

        na_cnt = df[feature].isna().sum()
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Count", df_descr[feature]['count']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {} / ({:.2f} %)".format("Missing Count", na_cnt, na_cnt/df.shape[0]), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {:.2f}".format("Mean", df_descr[feature]['mean']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {:.2f}".format("Standard Deviation", df_descr[feature]['std']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Minimum", df_descr[feature]['min']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> : {}".format("Maximum", df_descr[feature]['max']), unsafe_allow_html=True)
        st.markdown("<span style='font-weight:bold;'>{}</span> :".format("Quantiles"), unsafe_allow_html=True)
        st.write(df_descr[[feature]].T[['25%', "50%", "75%"]])
        ## Histogram
        hist_fig = df.plot_bokeh.hist(y=feature, bins=50, legend=False, vertical_xlabel=True, show_figure=False)
        st.bokeh_chart(hist_fig, use_container_width=True)

        st.markdown("#### 2. Understand Categorical Feature")
        feature = st.selectbox(label="Select Categorical Feature", options=cat_columns, index=0)
        ### Categorical Columns Distribution        
        cnts = Counter(df[feature].values)
        df_cnts = pd.DataFrame({"Type": cnts.keys(), "Values": cnts.values()})
        bar_fig = df_cnts.plot_bokeh.bar(x="Type", y="Values", color="tomato", legend=False, show_figure=False)
        st.bokeh_chart(bar_fig, use_container_width=True)

    with tab3: ## Explore Relation Between Features
        st.subheader("Explore Relationship Between Features of Dataset")
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(label="X-Axis", options=cont_columns, index=0)
        with col2:
            y_axis = st.selectbox(label="Y-Axis", options=cont_columns, index=1)

        color_encode = st.selectbox(label="Color-Encode", options=[None,] + cat_columns)

        scatter_fig = df.plot_bokeh.scatter(x=x_axis, y=y_axis, category=color_encode if color_encode else None, 
                                    title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()),
                                    figsize=(600,500), fontsize_title=20, fontsize_label=15,
                                    show_figure=False)
        st.bokeh_chart(scatter_fig, use_container_width=True)

# Set your OpenAI API key
openai.api_key = 'sk-proj-I7qbtei6AZXwk91uVF1xT3BlbkFJQ5avvdxGHVdfuaHGDNSb'

def load_data(file):
    # Load data from CSV
    df = pd.read_csv(file)
    return df

def extract_key_attributes(df):
    # Using OpenAI to extract key attributes from the data
    try:
        prompt = f"Analyze the following dataset and extract key attributes:\n\n{df.head().to_string(index=False)}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts key attributes from data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        key_attributes = response['choices'][0]['message']['content']
    except Exception as e:
        key_attributes = f"Error in extracting attributes: {str(e)}"
    return key_attributes

def generate_sales_report(attributes):
    # Generate a detailed sales report using the OpenAI chat model
    try:
        messages = [
            {"role": "system", "content": "You are an intelligent assistant capable of generating detailed sales reports based on key business attributes."},
            {"role": "user", "content": f"Based on these key attributes: {attributes}, generate a detailed sales report with recommendations for business decisions and give graphs and charts."}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.5
        )
        report = response['choices'][0]['message']['content']
    except Exception as e:
        report = f"Error in generating sales report: {str(e)}"
    return report

def main():
    st.title('Advanced Sales Data Analysis Tool')

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.dataframe(df)

        if st.button('Extract Key Attributes'):
            attributes = extract_key_attributes(df)
            st.session_state['extracted_attributes'] = attributes
            st.write('Extracted Key Attributes:', attributes)

        if 'extracted_attributes' in st.session_state:
            if st.button('Generate Detailed Sales Report'):
                report = generate_sales_report(st.session_state['extracted_attributes'])
                st.write('Detailed Sales Report:', report)

if __name__ == '__main__':
    main()
