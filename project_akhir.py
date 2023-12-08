import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = {'Transaction': [1, 1, 2, 2, 3, 3, 3],
        'Item': ['A', 'B', 'A', 'C', 'B', 'C', 'D']}
df = pd.read_csv('bread basket.csv')

st.sidebar.header('Apriori Algorithm Parameters')
min_support = st.sidebar.slider('Min Support', 0.01, 1.0, 0.1)
min_confidence = st.sidebar.slider('Min Confidence', 0.01, 1.0, 0.5)
lift_threshold = st.sidebar.slider('Min Lift', 0.0, 2.0, 1.0, step=0.01)


transactions = [group['Item'].tolist() for _, group in df.groupby(['Transaction'])]


te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
transactions_df = pd.DataFrame(te_ary, columns=te.columns_)

freq_items = apriori(transactions_df, min_support=min_support, use_colnames=True, verbose=1).sort_values(by='support', ascending=False)
freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))


def get_association_rules(min_confidence, lift_threshold):
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence, support_only=False)
    rules = rules[rules['lift'] >= lift_threshold]
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    return rules

rules = get_association_rules(min_confidence, lift_threshold)


st.title('Analisis Perilaku Konsumen Menggunakan Metode Asosiasi Dan Algoritma Apriori')

st.subheader('Raw Data')
st.dataframe(df)


st.subheader('Frequent Itemsets')
st.dataframe(freq_items)

st.subheader('Association Rules')
st.dataframe(rules)

def display_scatter_plot(data, x_col, y_col, size_col, title):
    fig = px.scatter(data, x=x_col, y=y_col, size=size_col, title=title)
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, showlegend=False)
    st.plotly_chart(fig)

st.header('Scatter Plot Lift vs Confidence')
display_scatter_plot(rules, 'lift', 'confidence', 'lift', 'Scatter Plot Lift vs Confidence')


st.header('Scatter Plot Support vs Confidence')
display_scatter_plot(rules, 'support', 'confidence', 'lift', 'Scatter Plot Support vs Confidence')

st.sidebar.subheader('Custom Association Analysis')
antecedents_input = st.sidebar.text_input('Antecedents (comma-separated)', '')
consequents_input = st.sidebar.text_input('Consequents (comma-separated)', '')

if antecedents_input and consequents_input:
    custom_antecedents = [item.strip() for item in antecedents_input.split(',')]
    custom_consequents = [item.strip() for item in consequents_input.split(',')]

    # Print statements for debugging
    print("Custom Antecedents:", custom_antecedents)
    print("Custom Consequents:", custom_consequents)
    
    custom_rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence, support_only=False)
    custom_rules = custom_rules[
        (custom_rules['antecedents'] == set(custom_antecedents)) &
        (custom_rules['consequents'] == set(custom_consequents))
    ]

    if not custom_rules.empty:
        st.subheader(f'Analisis Asosiasi Kustom untuk {antecedents_input} -> {consequents_input}')
        st.dataframe(custom_rules[['support', 'confidence', 'lift']])
        
        # Menampilkan support, confidence, dan lift dari aturan kustom
        st.subheader('Hasil Analisis Asosiasi Kustom')
        st.write(f"Support: {custom_rules['support'].values[0]}")
        st.write(f"Confidence: {custom_rules['confidence'].values[0]}")
        st.write(f"Lift: {custom_rules['lift'].values[0]}")
    else:
        st.warning('Tidak ada aturan yang ditemukan untuk antecedents dan consequents yang ditentukan.')