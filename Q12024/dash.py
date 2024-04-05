# VC Report
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from metabasepy import Client
import polars as pl
import os
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from PIL import Image

PALETTE=['#00B0F0', '#0A113A',  '#002c5f',  '#007eba', '#96ffff', '#FC2F6A', '#ff67b0', '#ff9aee', '#d394e7']
st.set_page_config(layout="wide")
st.title("Performance Metrics Q1 2024")

logo = Image.open('Q12024/wegrow.png')
st.sidebar.image(logo)

def load_data(path):
    data = pd.read_csv(path)
    return data

start_date = "2022-04-01"
end_date = "2024-03-01"
date_range = pd.period_range(start=start_date, end=end_date, freq='M')

def report(df, title, col):
    """
    Generates a report including a heatmap and dataframe display based on specified metrics.
    
    Parameters:
    - df: DataFrame containing the data to be analyzed.
    - title: The title for the report, used as a header for one of the columns.
    - col: The name of the column in the DataFrame to be analyzed (for pivot).
    """
    # Table preparation
    if title != "Share of Active Users":
        # Error handme
        if col not in df.columns:
            st.error(f"Column {col} not found in DataFrame.")
            return
        if 'Client' not in df.columns or 'YearMonth' not in df.columns:
            st.error("Required columns 'Client' or 'YearMonth' not found in DataFrame.")
            return
        df_pivot = df.pivot(index='Client', columns='YearMonth', values=col)
        df_pivot = df_pivot.reindex(columns=date_range, fill_value=np.nan)
        df_pivot = df_pivot.round(2)
    else: 
        df_pivot = df 
    
    df_pivot = df_pivot.replace(0, np.nan)
    df_annotations = df_pivot.copy() 
    df_normalized = df_pivot.div(df_pivot.max(axis=1), axis=0)

    # Metrics calculation
    average_monthly = df_pivot.mean(axis=1).round(2)

    percent_change = -df_pivot.iloc[:, ::-1].pct_change(axis=1) *100
    average_pct_change = percent_change.mean(axis=1).round(2)

    df_pivot.insert(0, 'Avg Pct Change (%)', average_pct_change.map("{:.2f}%".format))
    df_pivot.insert(0, title, average_monthly)

    if title == "Share of Active Users":
        avg_monthly_str = average_monthly.mean().round(2) * 100
    else: 
        avg_monthly_str = average_monthly.mean().round()
    avg_pct_change_str = average_pct_change.mean().round(2)
    if (title != "Reuses") and (title != "Share of Active Users"): 
        st.write('Our platforms have in average,',avg_monthly_str, title, ' per month, with an average monthly percent change of', avg_pct_change_str, '%.')
    if title == "Share of Active Users":
        st.write('Our platforms have in average,',avg_monthly_str, "%", title, ' per month, with an average monthly percent change of', avg_pct_change_str, '%.')

    # Tabs for heatmap and DataFrame display
    tab1, tab2 = st.tabs(["Heatmap", "Dataframe"])
    
    with tab1:
        plt.figure(figsize=(24, 12))
        ax = sns.heatmap(df_normalized, annot=df_annotations, fmt="g", cmap="YlGnBu", cbar=True, annot_kws={'size': 10})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top') 
        ax.set_ylabel('')
        st.pyplot(plt)
    
    with tab2:
        # Assuming data_editor is a typo. Use st.dataframe for displaying DataFrame.
        st.data_editor(df_pivot, height=600)

client_list = {
                "Cora": {
                    "id": 10,
                    "url": "https://cora.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "Operations",
                },
                "Campari": {
                    "id": 12,
                    "url": "https://campari.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Barilla": {
                    "id": 15,
                    "url": "https://barilla.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Suntory Caribbean": {"id": 18, "url": "None", "email_exception": [], "use_case": "Sales"},
                "Henkel ACB": {
                    "id": 21,
                    "url": "https://henkel-acb.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Mondelez": {
                    "id": 23,
                    "url": "https://mondelez.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "Awards",
                },
                "Campari SEMEA": {
                    "id": 24,
                    "url": "https://campari-semea.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "KHFS": {
                    "id": 27,
                    "url": "https://kraftheinz-ne.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Suntory Beam": {
                    "id": 28,
                    "url": "https://beamsuntorygemba.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Henkel ACC": {
                    "id": 29,
                    "url": "https://henkel-acc.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Nestle": {
                    "id": 31,
                    "url": "https://nestle-sales.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Circet": {
                    "id": 32,
                    "url": "https://circet.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "Operations",
                },
                "Henkel ACC NA": {
                    "id": 33,
                    "url": "https://henkel-acc-na.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "Sales",
                },
                "Unilever Icecream": {
                    "id": 34,
                    "url": "https://unilever-icecream.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "PUIG": {
                    "id": 35,
                    "url": "https://puig.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Suntory DC": {
                    "id": 38,
                    "url": "https://suntory-dc.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Logitech": {
                    "id": 39,
                    "url": "https://logitech.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Coca-Cola": {
                    "id": 40,
                    "url": "https://coca-cola.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Reckitt": {
                    "id": 42,
                    "url": "https://weshareit.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Henkel AMI": {
                    "id": 43,
                    "url": "https://henkel-ami.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Suntory DEI": {
                    "id": 44,
                    "url": "https://suntorydei.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Unilever Lux": {
                    "id": 45,
                    "url": "https://unilever-lux.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Coca E@S": {
                    "id": 46,
                    "url": "https://experimentation-at-scale.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "HP": {
                    "id": 47,
                    "url": "https://hp.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "KHNE": {
                    "id": 49,
                    "url": "https://kraftheinz.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Organon": {
                    "id": 51,
                    "url": "https://apjaward.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Henkel ACM": {
                    "id": 52,
                    "url": "https://henkel-acm.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Bayer CH": {
                    "id": 53,
                    "url": "https://bayerchsharing.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Essity": {
                    "id": 54,
                    "url": "https://essity.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Henkel Ecommerce": {
                    "id": 56,
                    "url": "https://henkel-ecom.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Henkel Hair Professional": {
                    "id": 57,
                    "url": "https://henkelprofessionalgrowth.wegrow-app.com/#/auth/sign-in?type=form",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Bayer Fine": {
                    "id": 55,
                    "url": "https://bayer-fine.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Rexel": {
                    "id": 58,
                    "url": "https://rexel.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Nestle Confectionary": {
                    "id": 65,
                    "url": "https://nestle-confectionary.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "LVMH": {
                    "id": 60,
                    "url": "https://partnershipshub-lvmh.wegrow-app.com",
                    "email_exception": ["lvmh.accelerate@wegrow-app.com"],
                    "use_case": "BP Scaling",
                },
                "Sandoz": {
                    "id": 61,
                    "url": "https://sandoz.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Henkel AOD": {
                    "id": 63,
                    "url": "https://henkel-aod.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Mars": {
                    "id": 64,
                    "url": "https://mars.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Michelin": {
                    "id": 67,
                    "url": "https://michelin.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Merck": {
                    "id": 66,
                    "url": "https://vshare-weareoncology.wegrow-app.com/",
                    "email_exception": [],
                    "use_case": "BP Scaling",
                },
                "Diageo": {"id": 68, "url": "None", "email_exception": [], "use_case": "BP Scaling"},
            }
# client_id = {client: client_list[client]["id"] for client in client_list if client_list[client]["use_case"] == "BP Scaling"}
client_id = {client: client_list[client]["id"] for client in client_list}

# API authentication
cli = Client(
username="fernando@wegrow-app.com",
password="EnterSandman#120466",
base_url="https://metabase.wegrow-lab.com",
        )
cli.authenticate()

@st.cache_data()
def load_dfs(_mbapi: Client, clients: dict, query: str, timestamp: bool = True) -> pl.DataFrame:
        """
        Loads data from Metabase using the provided query. It iterates through the clients
        and fetches the data for each one of them. It then concatenates the dataframes
        vertically.
        """
        df_result = pl.DataFrame()

        for i, (name, dbid) in enumerate(clients.items()):
            timeout = time.time() + 30  # 30 seconds from now
            while time.time() < timeout:
                try:
                    print(f"Fetching data for {name}...")
                    temp_file = f"temp.csv"
                    _mbapi.dataset.export(
                        database_id=dbid,
                        query=query,
                        export_format="csv",
                        full_path=temp_file,
                    )
                    break
                except Exception as e:
                    print(e)
                    print(f"Retrying for {name}...")
                    continue
            else:
                raise TimeoutError(f"Timeout for {name}.")

            try:
                interim = pl.read_csv(temp_file)
                interim = interim.with_columns(pl.lit(name).alias("Client"))
                if i == 0:
                    df_result = pl.concat([df_result, interim])
                else:
                    df_result = pl.concat([df_result, interim], how="vertical_relaxed")
            except pl.NoDataError:
                continue
            os.remove(temp_file)

        return df_result
        


def main():
    # Load Data
    df = load_data('Q12024/q1.csv')
    total_users = """
        select count(*) 
        from users_detail_with_preferences
        where email not like '%wegrow%'
        and ban = false
        """

    active_clients = """
        SELECT
            CASE WHEN COUNT(DISTINCT email) > 0 THEN TRUE ELSE FALSE END AS "Mature"
        FROM
            metabase_login_report
        WHERE
            userlevel = 1
            AND DATE_PART('year', date_time) <= 2022
            AND email not like '%wegrow%'
            AND action <> 'LOGINFAIL'
        """
    cohort= """
    SELECT userid, 
        TO_CHAR(MIN(date_time), 'YYYY-MM-DD') AS "first_login",
        TO_CHAR(MAX(date_time), 'YYYY-MM-DD') AS "last_login"
    FROM metabase_login_report a
    left join users_detail_with_preferences b on a.userid = b.id
    where a.email not like '%wegrow%'
    and b.ban = false
    GROUP BY userid
    """
    active_clients = load_dfs(cli, client_id, active_clients, False)
    all_clients_id = {client: client_id[client] for client in client_id}

    df_total_users = load_dfs(cli, all_clients_id, total_users)
    df_total_users = df_total_users.to_pandas()
    df_total_users = df_total_users[['Client', 'count']].rename(columns={'count':'Total Users'})

    # Data Clean
    clients_to_drop = ['Coca-Cola', 'Coca E@S', 'Henkel AOD', 'Henkel Hair Professional', 'Henkel LHC', 'Cora',
                       'LVMH', 'Logitech', 'Bayer CH', 'Essity', 
                       'Rexel', 'Sandoz', 'Sanofi', 'Diageo', 'Merck', 'Michelin', 'Nestle Confectionary', 'KHNE', 
                       'Henkel ACB', 'Suntory DEI', 'Unilever Lux']
    df = df[~df['Client'].isin(clients_to_drop)]
    df_total_users = df_total_users[~df_total_users['Client'].isin(clients_to_drop)]

    # Fix dates
    df['Timeframe'] = pd.to_datetime(df['Timeframe'])
    df['YearMonth'] = df['Timeframe'].dt.to_period('M')
    start_date = "2022-01-01"
    end_date = "2024-03-01"
    date_range = pd.period_range(start=start_date, end=end_date, freq='M')

    # Create DF Share df
    df_pct_users = df.pivot(index='Client', columns='YearMonth', values='Unique Users')
    df_pct_users = df_pct_users.reindex(columns=date_range, fill_value=np.nan)
    df_pct_users = df_pct_users.reset_index()
    df_pct_users = df_pct_users.merge(df_total_users, 'left', on='Client')
    df_pct_users = df_pct_users.set_index('Client')
    columns_except_total = df_pct_users.columns.difference(['Total Users'])
    df_pct_users[columns_except_total] = df_pct_users[columns_except_total].div(df_pct_users['Total Users'], axis=0).round(2)
    df_pct_users = df_pct_users.drop('Total Users', axis=1)

# SIDEBAR ==========================================================================================================================================
    # Navigation 
    st.sidebar.subheader("1.Navigation")
    section = st.sidebar.radio('Select a tab:', ['Total Users', 
                                                 'Unique Users per Month', 
                                                 'Share of Active Users', 
                                                 'Total BP Shared', 
                                                 'BP Shared per User', 
                                                 'BP Read per User', 
                                                 'BP Reused', 
                                                 'Cohort Analysis'])

    # Client Filter
    st.sidebar.subheader("2.Filter by Client")
    all_clients = sorted(df['Client'].unique())
    selected_clients = st.sidebar.multiselect('Select (a) client(s) to display (Empty selects all clients)', options=all_clients, default=[])
    if selected_clients:
        df = df[df['Client'].isin(selected_clients)]
    else:
        df = df
# TOTAL USERS ==========================================================================================================================================
    if section == "Total Users":
        st.subheader("Total Users")
        
        tab1, tab2 = st.tabs(["Chart", "Dataframe"])
        with tab1:
            # Assuming data_editor is a typo. Use st.dataframe for displaying DataFrame.
            fig = px.histogram(df_total_users, x='Client', y='Total Users', title='Total Users per Platform', 
                               nbins=200, color_discrete_sequence=PALETTE, text_auto = True)
            fig.update_layout(yaxis_title='Total Users', xaxis_title='date', width = 1000)
            fig.update_yaxes(showgrid=False, visible=False)
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width = True)
            
        with tab2:
            st.data_editor(df_total_users)
            
    # UNIQUE USERS ==========================================================================================================================================
    if section == "Unique Users per Month":
        st.subheader("Unique Users per Month")
        report(df, "Average Monthly", "Unique Users")

# SHARE OF ACTIVE USERS ==========================================================================================================================================
    if section == "Share of Active Users":
        st.subheader("Share of Active Users")
        report(df_pct_users, "Share of Active Users", "test")

# BEST PRACTICES SHARED ==========================================================================================================================================
    if section == "Total BP Shared":
        st.subheader("Total BP Shared")
        report(df, "Avg Monthly", "Post Count")

# BP per User==========================================================================================================================================
    if section == "BP Shared per User":
        st.subheader("BP Shared per User")
        report(df, "Avg Monthly", "Post Count Ratio")

# BP Read per User==========================================================================================================================================
    if section == "BP Read per User":
        st.subheader("BP Read per User")
        report(df, "Avg Monthly", "Views Ratio")

# BP Read per User==========================================================================================================================================
    if section == "BP Reused":
        st.subheader("BP Reused")
        report(df, "Avg Monthly", "Reuses")

# Cohort ==========================================================================================================================================
    if section == "Cohort Analysis":
        st.subheader("Cumulative Retention of 1st Cohort")
        
        df_cohort = load_dfs(cli, all_clients_id, cohort)
        df_cohort = df_cohort.to_pandas()
        
        clients_release_dates = {
            'Barilla': '2021-10-01',
            'Bayer Fine': '2023-10-01',
            'Campari': '2021-07-01',
            'Campari SEMEA': '2022-04-01',
            'Circet': '2022-11-01',
            'HP': '2023-09-01',
            'Henkel ACC': '2022-09-01',
            'Henkel ACC NA': '2022-10-01',
            'Henkel ACM': '2023-08-01',
            'Henkel AMI': '2023-05-01',
            'KHFS': '2022-10-01',
            'Mondelez': '2023-01-01',
            'Nestle': '2022-11-01',
            'PUIG': '2023-02-01',
            'Reckitt': '2023-04-01',
            'Suntory Beam': '2022-09-01',
            'Suntory Caribbean': '2022-02-01',
            'Suntory DC': '2023-01-01',
            'Unilever Icecream': '2023-02-01'
        }
        columns = ['Client'] + [f'M+{i}' for i in range(25)]

        def months_between(d1, d2):
            return (d2.year - d1.year) * 12 + d2.month - d1.month

        # Adjusted function to calculate cohort data
        def calculate_cohort_data(user_data, clients_release_dates):
            columns = ['Client'] + [f'M+{i}' for i in range(25)]
            rows_list = []

            for client, release_date in clients_release_dates.items():
                client_users = user_data[user_data['Client'] == client].copy()
                release_date_dt = datetime.strptime(release_date, '%Y-%m-%d')
                client_users['first_login_dt'] = pd.to_datetime(client_users['first_login'])
                client_users['last_login_dt'] = pd.to_datetime(client_users['last_login'])

                # Determine the 1st cohort
                first_cohort_users = client_users[(client_users['first_login_dt'] >= release_date_dt) &
                                                (client_users['first_login_dt'] < release_date_dt + relativedelta(months=+1))]

                # Initialize row data with zeros
                row_data = [0] * 25

                for _, user in first_cohort_users.iterrows():
                    # Calculate the active months from first to last login
                    active_months = months_between(user['first_login_dt'], user['last_login_dt'])

                    # Increment the count for each month the user was active within the first 24 months
                    for month in range(active_months + 1):
                        if month < 25:
                            row_data[month] += 1

                rows_list.append({'Client': client, **{f'M+{i}': row_data[i] for i in range(25)}})

            df_cohort = pd.DataFrame(rows_list, columns=columns)
            return df_cohort
        cohort_data = calculate_cohort_data(df_cohort, clients_release_dates)

        for i in range(1, 25):  # Start from 1 because we don't divide M+0 by itself
            if f'M+{i}' in cohort_data.columns:  # Check if the column exists
                cohort_data[f'M+{i}'] = (cohort_data[f'M+{i}'] / cohort_data['M+0']) * 100
            else:
                # Optionally, handle the case where the column does not exist, e.g., by setting it to 0 or some default value
                cohort_data[f'M+{i}'] = 0

        # To represent the starting point as 100%
        cohort_data['M+0'] = 100

        # Optional: Round the percentages for a cleaner look
        cohort_data = cohort_data.round(2)

        cohort_data_with_drop_rate = cohort_data.copy()

        # Replace 0 with NaN
        cohort_data_with_drop_rate.replace(0, np.nan, inplace=True)

        # Calculate drop rates row-wise and store them in a list
        avg_drop_rates = []

        for index, row in cohort_data_with_drop_rate.iterrows():
            # Drop rates for the current row/client
            drop_rates = []
            
            for i in range(1, 25):
                current_month = f'M+{i}'
                previous_month = f'M+{i-1}'
                
                # Calculate drop rate if both months have data
                if pd.notnull(row[previous_month]) and pd.notnull(row[current_month]):
                    drop_rate = (row[previous_month] - row[current_month]) / row[previous_month] * 100
                    drop_rates.append(drop_rate)
            
            # Calculate the average drop rate for the client, ignoring NaNs
            avg_drop_rate = np.nanmean(drop_rates)
            avg_drop_rates.append(avg_drop_rate)

        # Add the average drop rates to the DataFrame
        cohort_data_with_drop_rate['Avg Drop Rate'] = avg_drop_rates

        # Rearrange the DataFrame columns to have 'Client', 'Avg Drop Rate', then the 0 to 24 columns
        cols = ['Client', 'Avg Drop Rate'] + [f'M+{i}' for i in range(25)]
        cohort_data_with_drop_rate = cohort_data_with_drop_rate[cols]

        # Optionally, round the 'Avg Drop Rate' for a cleaner look
        cohort_data_with_drop_rate['Avg Drop Rate'] = cohort_data_with_drop_rate['Avg Drop Rate'].round(2)

        cohort_data_with_drop_rate = cohort_data_with_drop_rate.set_index(['Client', 'Avg Drop Rate'])
        
        st.write("The following chart analyzes the monthly retention of the **first** cohort of clients registered into the platform.")
        tab1, tab2 = st.tabs(["Heatmap", "Dataframe"])
    
        with tab1:
            plt.figure(figsize=(24, 12))
            ax = sns.heatmap(cohort_data_with_drop_rate, annot=cohort_data_with_drop_rate, fmt="g", cmap="YlGnBu", cbar=True, annot_kws={'size': 10})
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
            ax.xaxis.tick_top()  # Position x-axis ticks at the top
            ax.xaxis.set_label_position('top') 
            ax.set_ylabel('')  # Remove the y-axis label if not needed
            st.pyplot(plt)
        
        with tab2:
            # Assuming data_editor is a typo. Use st.dataframe for displaying DataFrame.
            cohort_data_with_drop_rate = cohort_data_with_drop_rate.reset_index().set_index('Client')
            cohort_data_with_drop_rate['M+0'] = cohort_data_with_drop_rate['M+0'].astype(float)
            cohort_data_with_drop_rate = cohort_data_with_drop_rate.applymap(lambda x: f"{x:.1f}%" if isinstance(x, float) else x)
            st.dataframe(cohort_data_with_drop_rate, height=600)  
    
if __name__ == "__main__":
    main()

