import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide")

st.title("Survey Dashboard")

if "df" not in st.session_state:
    nltk.download('vader_lexicon')
    st.session_state.df = pd.read_excel("Friends-and-Family-SMS-Patient-Feedback.xlsx")
    st.session_state.df['Entry Date'] = pd.to_datetime(st.session_state.df['Entry Date'], format='%B %d, %Y %I:%M %p')
    st.session_state.use_df = None

df = None
with st.sidebar:
    y_m = st.selectbox("Yearly/Monthly", ["Yearly", "Monthly"])
    
    roles = st.session_state.df['Are you? (2)'].unique().tolist()
    roles.insert(0, "All")
    participant = st.selectbox("Participant", roles)
    
    
    genders = st.session_state.df['Are you?'].unique().tolist()
    genders.insert(0, "All")
    gender = st.selectbox("Gender", genders)

       
    submit = st.button("Submit")

    if submit:
        if participant != "All":
            df = st.session_state.df[st.session_state.df['Are you? (2)'] == participant]
        else:
            df = st.session_state.df
        if gender != "All":
            df = df[df['Are you?'] == gender]
        if y_m == "Yearly":
            split = "Y"
            x_title = "Years"
        else:
            split = "M"
            x_title = "Month-Year"



if df is not None:
    df['month_year'] = pd.to_datetime(df['Entry Date']).dt.to_period(split).astype(str)

    other_service = df['Have you tried to access any other service prior to contacting BARDOC?'].replace("No", np.nan).count()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Responses", df['Entry Date'].count())
    c2.metric("Previously Used Other Services", other_service)


    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df['Thinking about your response to this question, what is the main reason why you feel this way?'].dropna().apply(lambda x: sid.polarity_scores(x)['compound'])

    bins = [-1, -0.5, 0.5, 1]
    labels = ['Negative', 'Neutral', 'Positive']
    df['sentiment_category'] = pd.cut(df['sentiment'], bins=bins, labels=labels)

    sentiment_over_time = df.groupby(['month_year', 'sentiment_category']).size().unstack(fill_value=0).reset_index()   
    
    fig = go.Figure()

    fig.add_trace(go.Bar(x=sentiment_over_time['month_year'], y=sentiment_over_time['Negative'], name='Negative', marker_color='red'))
    fig.add_trace(go.Bar(x=sentiment_over_time['month_year'], y=sentiment_over_time['Neutral'], name='Neutral', marker_color='LightSkyBlue'))
    fig.add_trace(go.Bar(x=sentiment_over_time['month_year'], y=sentiment_over_time['Positive'], name='Positive', marker_color='LightGreen'))

    sentiment_over_time['Total'] = sentiment_over_time[['Negative', 'Neutral', 'Positive']].sum(axis=1)
    fig.add_trace(go.Scatter(x=sentiment_over_time['month_year'], y=sentiment_over_time['Total'], mode='lines+markers', name='Total', line=dict(color='black', width=2)))

    fig.update_layout(title='Sentiment Analysis Over Time',
                    xaxis_title=x_title,
                    yaxis_title='Count',
                    xaxis_tickformat = '%b %Y',
                    xaxis=dict(
                      tickvals=sentiment_over_time['month_year'],
                    ),
                    barmode='group')

    st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with col1:
        improve_data = {}
        improve_cols = [
            "What area of our service could we improve?: Staff Attitude",
            "What area of our service could we improve?: Premises",
            "What area of our service could we improve?: Hygiene",
            "What area of our service could we improve?: Prescription issues",
            "What area of our service could we improve?: Delay in referral/diagnosis",
            "What area of our service could we improve?: Communication",
            "What area of our service could we improve?: Inaccurate records",
            "What area of our service could we improve?: Waiting time for call back",
            "What area of our service could we improve?: Waiting time for appointment",
            "What area of our service could we improve?: Refusal to prescribe",
            "What area of our service could we improve?: Refusal to see face to face",
            "What area of our service could we improve?: Referred to A&E",
            "What area of our service could we improve?: Waiting Times (modified)",
            "What area of our service could we improve?: Other (modified)",
        ]
        unique_keys = [
            "Staff Attitude",
            "Premises",
            "Hygiene",
            "Prescription issues",
            "Delay in referral/diagnosis",
            "Communication",
            "Inaccurate records",
            "Waiting time for call back",
            "Waiting time for appointment",
            "Refusal to prescribe",
            "Refusal to see face to face",
            "Referred to A&E",
            "Waiting Times",
            "Other",
        ]

        improve_data = {key: df[col].dropna().count() for key, col in zip(unique_keys, improve_cols)}
        data_df = pd.DataFrame(list(improve_data.items()), columns=['Area of Improvement', 'Count'])
        data_df = data_df.sort_values(by='Count', ascending=False)

        fig = px.bar(data_df, x='Area of Improvement', y='Count', title='Area of Improvement Distribution',
                    labels={'Area of Improvement': 'Count'},
                    color='Area of Improvement',
                    color_continuous_scale='Viridis')
        fig.update_traces(texttemplate='%{x}', textposition='outside')
        fig.update_layout(
            yaxis={'categoryorder':'total ascending'},
            xaxis_title="Area of Improvement",
            yaxis_title="",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rating_counts = df['How would you rate the service we provided you'].value_counts().reset_index()
        rating_counts.columns = ['Rating', 'Count']

        rating_counts = rating_counts.sort_values(by='Count', ascending=False)

        fig = px.pie(rating_counts, values='Count', names='Rating', 
                title='Service Rating Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel)

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            showlegend=True,
            height=500,
            legend_title="Rating"
        )
        st.plotly_chart(fig, use_container_width=True)


    monthly_age_counts = df.groupby(['month_year', 'What age are you?']).size().reset_index(name='count')


    age_counts = df['What age are you?'].value_counts().reset_index()
    fig = px.bar(monthly_age_counts, x='month_year', y='count', color='What age are you?', text='count',
                title='Monthly Age Group Distribution',
                labels={'count': 'Number of Occurrences', 'month_year': x_title},
                color_discrete_sequence=px.colors.qualitative.Pastel)

    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Count",
        xaxis_tickformat = '%b %Y',
        xaxis=dict(
            tickvals=sentiment_over_time['month_year'],
        ),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    monthly_gender_counts = df.groupby(['month_year', 'Are you?']).size().reset_index(name='Count')

    fig = px.bar(monthly_gender_counts, x='month_year', y='Count', color='Are you?', text='Count',
                title='Monthly Gender Distribution',
                labels={'Count': 'Number of Occurrences', 'month_year': x_title, 'Are you?': 'Gender'},
                color_discrete_sequence=px.colors.qualitative.Pastel)

    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Count",
        xaxis_tickformat = '%b %Y',
        xaxis=dict(
            tickvals=sentiment_over_time['month_year'],
        ),
        showlegend=True,
        height=500,
        legend_title="Gender"
    )
    st.plotly_chart(fig, use_container_width=True)


    role_counts = df['Are you? (2)'].value_counts().reset_index()
    role_counts.columns = ['Role', 'Count']

    fig = px.bar(role_counts, x='Role', y='Count', text='Count',
                title='Role Distribution',
                labels={'Count': 'Number of Occurrences', 'Role': 'Role'},
                color='Role',
                color_discrete_sequence=px.colors.qualitative.Pastel)

    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Role",
        yaxis_title="Count",
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


    
    colum1, colum2 = st.columns(2)

    with colum1:
        ethnic_counts = df['Which of the following best describes your ethnic background?'].value_counts().reset_index()
        ethnic_counts.columns = ['Ethnic Background', 'Count']

        ethnic_counts = ethnic_counts.sort_values('Count', ascending=True)

        fig = px.bar(ethnic_counts, x='Count', y='Ethnic Background', 
                    title='Ethnic Background Distribution',
                    labels={'Count': 'Number of Respondents'},
                    color='Ethnic Background',
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    orientation='h')

        fig.update_traces(texttemplate='%{x}', textposition='outside')
        fig.update_layout(
            height=500,
            yaxis={'categoryorder':'total ascending'},
            xaxis_title="Count",
            yaxis_title="",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)
    
    with colum2:
        response_counts = df['Would you be willing for us to contact you to discuss your responses further?'].value_counts().reset_index()
        response_counts.columns = ['Response', 'Count']

        response_counts = response_counts.sort_values(by='Count', ascending=False)

        fig = px.pie(response_counts, values='Count', names='Response', 
                title='Willingness to be contacted Again Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel)

        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            showlegend=True,
            height=500,
            legend_title="Rating"
        )
        st.plotly_chart(fig, use_container_width=True)
