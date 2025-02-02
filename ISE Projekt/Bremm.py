import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from sqlalchemy import create_engine, text
import datetime
import dash_bootstrap_components as dbc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input as KerasInput, Dense
from statsmodels.tsa.arima.model import ARIMA
import plotly.colors as pc
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
app.config.suppress_callback_exceptions = True
server = app.server

engine = create_engine('postgresql://postgres:Rayan1388@localhost:5432/pizza')

cache = {}

def load_data(store_ids=None, start_date=None, end_date=None):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)

    cache_key = (tuple(store_ids) if store_ids else 'all', start_date, end_date)
    if cache_key in cache:
        return cache[cache_key]

    if store_ids:
        placeholders = ','.join([':store_id' + str(i) for i in range(len(store_ids))])
        query = f"SELECT storeid, orderdate, total FROM orders WHERE storeid IN ({placeholders}) AND orderdate BETWEEN :start_date AND :end_date"
        params = {f'store_id{i}': store_id for i, store_id in enumerate(store_ids)}
        params['start_date'] = start_date
        params['end_date'] = end_date
    else:
        query = "SELECT storeid, orderdate, total FROM orders WHERE orderdate BETWEEN :start_date AND :end_date"
        params = {"start_date": start_date, "end_date": end_date}

    df = pd.read_sql(text(query), con=engine, params=params)
    df['orderdate'] = pd.to_datetime(df['orderdate']).dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    cache[cache_key] = df
    return df

def get_store_locations():
    query = "SELECT storeid, latitude, longitude, city FROM stores"
    df = pd.read_sql(query, con=engine)
    return df

def get_date_range():
    query = "SELECT MIN(orderdate) AS min_date, MAX(orderdate) AS max_date FROM orders"
    with engine.connect() as connection:
        result = connection.execute(text(query)).fetchone()
        min_date, max_date = result
        return min_date.date(), max_date.date()

min_date, max_date = get_date_range()

def load_initial_data():
    customers = pd.read_sql("SELECT * FROM customers", engine)
    orders = pd.read_sql("SELECT * FROM orders", engine)
    order_items = pd.read_sql("SELECT * FROM orders_items", engine)
    products = pd.read_sql("SELECT * FROM products", engine)

    orders['orderdate'] = pd.to_datetime(orders['orderdate']).dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

    customer_expenses = orders.groupby('customerid').agg({'total': 'sum'}).reset_index()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=0)
    customer_expenses['cluster'] = kmeans.fit_predict(customer_expenses[['total']])

    cluster_map = customer_expenses.groupby('cluster')['total'].sum().sort_values(ascending=False).index
    cluster_labels = {cluster_map[0]: 'A', cluster_map[1]: 'C', cluster_map[2]: 'B'}
    customer_expenses['cluster'] = customer_expenses['cluster'].map(cluster_labels)

    customers = customers.merge(customer_expenses[['customerid', 'cluster']], on='customerid', how='left')

    order_items = order_items.merge(products, left_on='sku', right_on='sku')
    orders = orders.merge(order_items, on='orderid')
    orders = orders.merge(customers[['customerid', 'cluster']], on='customerid', how='left')

    segment_expenses = orders.groupby(['cluster', 'category'], observed=True).agg({'total': 'sum'}).reset_index()

    return customers, orders, segment_expenses

customers, orders, segment_expenses = load_initial_data()

def load_store_metrics_comparison_data():
    orders_df = pd.read_sql("SELECT * FROM orders", engine)
    stores_df = pd.read_sql("SELECT * FROM stores", engine)
    customers_df = pd.read_sql("SELECT * FROM customers", engine)
    orderitems_df = pd.read_sql("SELECT * FROM orders_items", engine)

    orders_df['orderdate'] = pd.to_datetime(orders_df['orderdate'], utc=True)

    orders_customers_df = pd.merge(orders_df, customers_df, on='customerid')
    orders_customers_df = pd.merge(orders_customers_df, stores_df, on='storeid')

    avg_distance = (
        orders_customers_df
        .groupby('storeid')
        .agg(avg_distance=pd.NamedAgg(column='distance', aggfunc='mean'))
        .reset_index()
    )

    unique_customers = (
        orders_customers_df
        .groupby('storeid')
        .agg(unique_customers=pd.NamedAgg(column='customerid', aggfunc='nunique'))
        .reset_index()
    )

    avg_spending_per_customer = (
        orders_customers_df
        .groupby(['storeid', 'customerid'])
        .agg(total_spending=pd.NamedAgg(column='total', aggfunc='sum'))
        .groupby('storeid')
        .agg(avg_spending_per_customer=pd.NamedAgg(column='total_spending', aggfunc='mean'))
        .reset_index()
    )

    store_features = pd.merge(stores_df, avg_distance, on='storeid')
    store_features = pd.merge(store_features, unique_customers, on='storeid')
    store_features = pd.merge(store_features, avg_spending_per_customer, on='storeid')

    ibapah_store_id = stores_df[stores_df['city'] == 'Ibapah']['storeid'].values[0]

    return store_features, ibapah_store_id, stores_df

store_features, ibapah_store_id, stores_df = load_store_metrics_comparison_data()

def plot_ibapah_comparison():
    orders_query = "SELECT * FROM orders"
    stores_query = "SELECT * FROM stores"
    customers_query = "SELECT * FROM customers"
    orderitems_query = "SELECT * FROM orders_items"

    orders_df = pd.read_sql(orders_query, engine)
    stores_df = pd.read_sql(stores_query, engine)
    customers_df = pd.read_sql(customers_query, engine)
    orderitems_df = pd.read_sql(orderitems_query, engine)

    orders_df['orderdate'] = pd.to_datetime(orders_df['orderdate'], utc=True)

    orders_customers_df = pd.merge(orders_df, customers_df, on='customerid')
    orders_customers_df = pd.merge(orders_customers_df, stores_df, on='storeid')

    ibapah_store_id = stores_df[stores_df['city'] == 'Ibapah']['storeid'].values[0]

    ibapah_data = orders_customers_df[orders_customers_df['storeid'] == ibapah_store_id]

    ibapah_avg_distance = ibapah_data['distance'].mean()
    ibapah_unique_customers = ibapah_data['customerid'].nunique()
    ibapah_avg_spending_per_customer = ibapah_data.groupby('customerid')['total'].sum().mean()

    other_stores_data = orders_customers_df[orders_customers_df['storeid'] != ibapah_store_id]

    other_avg_distance = other_stores_data.groupby('storeid')['distance'].mean().mean()
    other_unique_customers = other_stores_data.groupby('storeid')['customerid'].nunique().mean()
    other_avg_spending_per_customer = other_stores_data.groupby(['storeid', 'customerid'])['total'].sum().groupby(
        'storeid').mean().mean()

    comparison_df = pd.DataFrame({
        'Metric': ['Average Distance to Store', 'Unique Customers', 'Average Spending per Customer'],
        'Ibapah': [ibapah_avg_distance, ibapah_unique_customers, ibapah_avg_spending_per_customer],
        'Other Stores Average': [other_avg_distance, other_unique_customers, other_avg_spending_per_customer]
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(['Ibapah', 'Other Stores Average'], [ibapah_avg_distance, other_avg_distance], color=['blue', 'gray'])
    axes[0].set_title('Average Distance to Store')
    axes[0].set_ylabel('Distance')

    axes[1].bar(['Ibapah', 'Other Stores Average'], [ibapah_unique_customers, other_unique_customers],
                color=['blue', 'gray'])
    axes[1].set_title('Unique Customers')
    axes[1].set_ylabel('Number of Unique Customers')

    axes[2].bar(['Ibapah', 'Other Stores Average'], [ibapah_avg_spending_per_customer, other_avg_spending_per_customer],
                color=['blue', 'gray'])
    axes[2].set_title('Average Spending per Customer')
    axes[2].set_ylabel('Spending')

    plt.suptitle('Comparison of Ibapah Store Metrics to Other Stores')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.callback(
    Output('ibapah-comparison-plot', 'src'),
    Input('tabs', 'active_tab')
)
def update_ibapah_comparison_plot(active_tab):
    if active_tab == "tab-comparison":
        img_base64 = plot_ibapah_comparison()
        return f"data:image/png;base64,{img_base64}"
    return dash.no_update

def load_animation_data():
    orderitems_df = pd.read_sql("SELECT * FROM orders_items", engine)
    orders_df = pd.read_sql("SELECT * FROM orders", engine)
    products_df = pd.read_sql("SELECT * FROM products", engine)
    stores_df = pd.read_sql("SELECT * FROM stores", engine)

    orders_df['orderdate'] = pd.to_datetime(orders_df['orderdate'], utc=True).dt.tz_convert(None)

    orders_df['year_month'] = orders_df['orderdate'].dt.to_period('M')

    product_sales_df = (
        orderitems_df
        .merge(orders_df, on='orderid')
        .merge(products_df, on='sku')
        .groupby(['sku', 'name', 'year_month'])
        .agg(total_sales=pd.NamedAgg(column='total', aggfunc='sum'))
        .reset_index()
    )

    store_sales_df = (
        orders_df
        .merge(stores_df, on='storeid')
        .groupby(['storeid', 'city', 'year_month'])
        .agg(total_sales=pd.NamedAgg(column='total', aggfunc='sum'))
        .reset_index()
    )

    return product_sales_df, store_sales_df

product_sales_df, store_sales_df = load_animation_data()

def load_market_basket_data():
    orderitems = pd.read_sql('SELECT * FROM orders_items', engine)
    orders = pd.read_sql('SELECT * FROM orders', engine)
    products = pd.read_sql('SELECT * FROM products', engine)

    order_product_data = pd.merge(orderitems, orders, on='orderid')
    order_product_data = pd.merge(order_product_data, products, on='sku')
    order_product_data = order_product_data[['orderid', 'name']]

    basket = (order_product_data.groupby(['orderid', 'name'])['name']
              .count().unstack().reset_index().fillna(0)
              .set_index('orderid'))
    basket = basket.astype(bool)

    frequent_itemsets = apriori(basket, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets_1 = frequent_itemsets[frequent_itemsets['length'] == 1]
    frequent_itemsets_2 = frequent_itemsets[frequent_itemsets['length'] == 2]
    frequent_itemsets_3 = frequent_itemsets[frequent_itemsets['length'] == 3]

    return frequent_itemsets_1, frequent_itemsets_2, frequent_itemsets_3, rules

frequent_itemsets_1, frequent_itemsets_2, frequent_itemsets_3, rules = load_market_basket_data()

def load_additional_data():
    orderitems = pd.read_sql('SELECT * FROM orders_items', engine)
    orders = pd.read_sql('SELECT * FROM orders', engine)
    customers = pd.read_sql('SELECT * FROM customers', engine)
    stores = pd.read_sql('SELECT * FROM stores', engine)
    products = pd.read_sql('SELECT * FROM products', engine)

    order_product_data = pd.merge(orderitems, orders, on='orderid')
    order_product_data = pd.merge(order_product_data, products, on='sku')
    order_product_data = pd.merge(order_product_data, customers, on='customerid')
    order_product_data = pd.merge(order_product_data, stores, on='storeid', suffixes=('_customer', '_store'))

    return order_product_data

order_product_data = load_additional_data()

def load_veggie_pizza_data():
    orderitems = pd.read_sql('SELECT * FROM orders_items', engine)
    orders = pd.read_sql('SELECT * FROM orders', engine)
    products = pd.read_sql('SELECT * FROM products', engine)
    stores = pd.read_sql('SELECT * FROM stores', engine)

    veggie_skus = products[products['name'].str.contains('Veggie Pizza', case=False, na=False)]['sku']

    veggie_orderitems = orderitems[orderitems['sku'].isin(veggie_skus)]

    merged_df = veggie_orderitems.merge(orders, on='orderid').merge(products, on='sku')

    veggie_sales_per_store = merged_df.groupby('storeid')['total'].sum().reset_index()
    veggie_sales_per_store.columns = ['storeid', 'veggie_sales']

    total_sales_per_store = orders.groupby('storeid')['total'].sum().reset_index()
    total_sales_per_store.columns = ['storeid', 'total_sales']

    sales_per_store = total_sales_per_store.merge(veggie_sales_per_store, on='storeid', how='left').fillna(0)
    sales_per_store = sales_per_store.merge(stores[['storeid', 'city']], on='storeid', how='left')

    sales_per_store['veggie_sales_percentage'] = (sales_per_store['veggie_sales'] / sales_per_store[
        'total_sales']) * 100

    return sales_per_store

sales_per_store = load_veggie_pizza_data()

color_sequence = pc.qualitative.Plotly

store_ids = get_store_locations()['storeid'].unique()

color_map = {store_id: color_sequence[i % len(color_sequence)] for i, store_id in enumerate(store_ids)}

navbar = dbc.NavbarSimple(
    brand="Pizza Store Dashboard",
    brand_href="#",
    color="primary",
    dark=True,
)

app.layout = dbc.Container([
    navbar,
    dcc.Store(id='selected-stores-data', data=[]),
    dbc.Row([
        dbc.Col(
            dbc.Tabs(
                [
                    dbc.Tab(label="Sales Analysis", tab_id="tab-sales"),
                    dbc.Tab(label="Customer Segmentation", tab_id="tab-customer"),
                    dbc.Tab(label="Forecasting Analysis", tab_id="tab-forecasting"),
                    dbc.Tab(label="Store Metrics Comparison", tab_id="tab-comparison"),
                    dbc.Tab(label="Live Animations", tab_id="tab-animations"),
                    dbc.Tab(label="Market Basket Analysis", tab_id="tab-market-basket")
                ],
                id="tabs",
                active_tab="tab-sales",
                className='mb-3'
            ), width=12
        )
    ]),
    html.Div(id="content")
], fluid=True)

sales_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H1("Pizza Store Sales Dashboard", className='text-center my-4 text-light'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.P("Select stores by clicking on the map. Selected stores will be displayed below.",
                       className='text-center mb-4 text-light'), width=12)
    ]),
    dbc.Row([

        dbc.Col(
            dcc.Graph(id='store-map', style={'height': '500px'}),
            width=6
        ),

        dbc.Col(
            dcc.Loading(
                id="loading-1",
                type="default",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='graph', style={'height': '500px'}),
                        ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s',
                                  'border-radius': '12px'})
                    )
                ]
            ), width=6
        )
    ]),
    dbc.Row([

        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Total Sales", className="card-title"),
                    html.H2(id="total-sales", className="card-text")
                ]), className="mb-3"
            ), width=3
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Total Orders", className="card-title"),
                    html.H2(id="total-orders", className="card-text")
                ]), className="mb-3"
            ), width=3
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Avg Order Value", className="card-title"),
                    html.H2(id="avg-order-value", className="card-text")
                ]), className="mb-3"
            ), width=3
        ),

        dbc.Col(
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=min_date,
                end_date=max_date,
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                display_format='YYYY-MM-DD',
                className='mb-3',
                style={'color': '#000'}
            ), width=3
        ),
    ]),
    dbc.Row([

        dbc.Col(
            dbc.Button('Select All Stores', id='select-all-stores-btn', color='primary', className='mb-2 btn-block',
                       style={'border-radius': '12px'}),
            width=2
        ),
        dbc.Col(
            dbc.Button('Deselect All Stores', id='deselect-all-stores-btn', color='primary', className='mb-2 btn-block',
                       style={'border-radius': '12px'}),
            width=2
        ),
        dbc.Col(
            dbc.Button('Load Data', id='load-data-btn', color='primary', className='mb-2 btn-block',
                       style={'border-radius': '12px'}),
            width=2
        ),
    ], justify="start"),

    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-2",
                type="default",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='heatmap', style={'height': '400px', 'width': '100%'}),
                        ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s',
                                  'border-radius': '12px'})
                    )
                ]
            ), width=6
        ),
        dbc.Col(
            dcc.Loading(
                id="loading-3",
                type="default",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='store-comparison', style={'height': '400px', 'width': '100%'}),
                        ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s',
                                  'border-radius': '12px'})
                    )
                ]
            ), width=6
        )
    ]),
])

customer_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H2("Customer and Product Segment Analysis", className='text-center my-4 text-light'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.P("Cluster analysis to identify customer segments based on purchase behavior.",
                       className='text-center mb-4 text-light'), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='cluster-dropdown',
                options=[
                    {'label': 'Cluster A', 'value': 'A'},
                    {'label': 'Cluster B', 'value': 'B'},
                    {'label': 'Cluster C', 'value': 'C'},
                    {'label': 'All Clusters', 'value': 'all'}
                ],
                value='all',
                placeholder="Select a cluster",
                className='mb-3',
                style={'color': '#000'}
            ), width=6
        ),
        dbc.Col(
            dcc.RangeSlider(
                id='date-slider',
                min=orders['orderdate'].dt.year.min(),
                max=orders['orderdate'].dt.year.max(),
                value=[orders['orderdate'].dt.year.min(), orders['orderdate'].dt.year.max()],
                marks={str(year): str(year) for year in
                       range(orders['orderdate'].dt.year.min(), orders['orderdate'].dt.year.max() + 1)},
                step=None
            ), width=6
        )
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    dcc.Graph(id='cluster-graph', style={'height': '400px', 'width': '100%'}),
                ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s', 'border-radius': '12px'})
            ), width=12
        )
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    dcc.Graph(id='expenses-graph', style={'height': '400px', 'width': '100%'}),
                ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s', 'border-radius': '12px'})
            ), width=12
        )
    ]),

])

forecasting_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H2("Sales Forecasting", className='text-center my-4 text-light'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.P("Use historical data to forecast future sales trends.", className='text-center mb-4 text-light'),
                width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='forecast-store-dropdown',
                options=[{'label': f"Store {store_id} - {row['city']}", 'value': store_id} for store_id, row in
                         get_store_locations().set_index('storeid').iterrows()],
                value=orders['storeid'].unique()[0],
                placeholder="Select a store for forecasting",
                className='mb-3',
                style={'color': '#000'}
            ), width=6
        ),
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-forecast",
                type="default",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='forecast-graph', style={'height': '400px', 'width': '100%'}),
                        ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s',
                                  'border-radius': '12px'})
                    )
                ]
            ), width=12
        )
    ]),

])

comparison_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H2("Comparison of Store Metrics", className='text-center my-4 text-light'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Label("Select a Store to Compare with Ibapah:", className='text-light'), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='store-dropdown',
                options=[{'label': city, 'value': storeid} for city, storeid in
                         zip(stores_df['city'], stores_df['storeid']) if storeid != ibapah_store_id],
                value=None,
                placeholder="Select a store",
                style={'color': '#000'}
            ), width=6
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='comparison-graph'),
            width=12
        )
    ]),

    dbc.Row([
        dbc.Col(
            html.Img(id='ibapah-comparison-plot', src='', style={'width': '100%', 'height': 'auto'}),
            width=12
        )
    ]),
])

animations_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H2("Live Sales Animations", className='text-center my-4 text-light'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.P("Animations for product and store sales over time.", className='text-center mb-4 text-light'),
                width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-animation-product",
                type="default",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='product-animation', style={'height': '300px', 'width': '100%'}),
                        ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s',
                                  'border-radius': '12px'})
                    )
                ]
            ), width=6
        ),
        dbc.Col(
            dcc.Loading(
                id="loading-animation-store",
                type="default",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='store-animation', style={'height': '300px', 'width': '100%'}),
                        ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s',
                                  'border-radius': '12px'})
                    )
                ]
            ), width=6
        )
    ]),
])

market_basket_layout = html.Div([
    dbc.Row([
        dbc.Col(html.H2("Market Basket Analysis", className='text-center my-4 text-light'), width=12)
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='itemset-size',
                options=[
                    {'label': 'Single Products', 'value': 1},
                    {'label': 'Product Pairs', 'value': 2},
                    {'label': 'Product Triplets', 'value': 3},
                ],
                value=1,
                className='mb-3',
                style={'color': '#000'}
            ), width=6
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-market-basket",
                type="default",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='itemset-graph', style={'height': '400px', 'width': '100%'}),
                        ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s',
                                  'border-radius': '12px'})
                    )
                ]
            ), width=12
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-veggie-sales",
                type="default",
                children=[
                    dcc.Graph(id='veggie-sales-bar-graph', style={'height': '400px', 'width': '100%'})
                ]
            ), width=6
        ),
        dbc.Col(
            dcc.Loading(
                id="loading-veggie-sales-pie",
                type="default",
                children=[
                    dcc.Graph(id='veggie-sales-pie-chart', style={'height': '400px', 'width': '100%'})
                ]
            ), width=6
        ),
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Loading(
                id="loading-pizza-count",
                type="default",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='pizza-count-graph', style={'height': '400px', 'width': '100%'})
                        ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s',
                                  'border-radius': '12px'})
                    )
                ]
            ), width=6
        ),
        dbc.Col(
            dcc.Loading(
                id="loading-geographic-sales",
                type="default",
                children=[
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(id='geographic-sales-graph', style={'height': '400px', 'width': '100%'})
                        ], style={'box-shadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'transition': '0.3s',
                                  'border-radius': '12px'})
                    )
                ]
            ), width=6
        )
    ]),
])

@app.callback(
    Output("content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "tab-sales":
        return sales_layout
    elif active_tab == "tab-customer":
        return customer_layout
    elif active_tab == "tab-forecasting":
        return forecasting_layout
    elif active_tab == "tab-comparison":
        return comparison_layout
    elif active_tab == "tab-animations":
        return animations_layout
    elif active_tab == "tab-market-basket":
        return market_basket_layout
    else:
        return "No content available"

@app.callback(
    Output('selected-stores-data', 'data'),
    [Input('store-map', 'clickData')],
    State('selected-stores-data', 'data')
)
def update_selected_stores(clickData, selected_stores):
    if not clickData:
        raise PreventUpdate

    store_id = clickData['points'][0]['customdata'][0]

    if selected_stores is None:
        selected_stores = []

    if store_id in selected_stores:
        selected_stores.remove(store_id)
    else:
        selected_stores.append(store_id)

    return selected_stores

@app.callback(
    Output('selected-stores-data', 'data', allow_duplicate=True),
    Input('select-all-stores-btn', 'n_clicks'),
    prevent_initial_call=True
)
def select_all_stores(n_clicks_select_all):
    if not n_clicks_select_all:
        raise PreventUpdate

    df = get_store_locations()
    return df['storeid'].tolist()

@app.callback(
    Output('selected-stores-data', 'data', allow_duplicate=True),
    Input('deselect-all-stores-btn', 'n_clicks'),
    prevent_initial_call=True
)
def deselect_all_stores(n_clicks_deselect_all):
    if not n_clicks_deselect_all:
        raise PreventUpdate

    return []

@app.callback(
    Output('total-sales', 'children'),
    Output('total-orders', 'children'),
    Output('avg-order-value', 'children'),
    [Input('selected-stores-data', 'data'),
     Input('load-data-btn', 'n_clicks'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_kpis(selected_stores, n_clicks, start_date, end_date):
    if not n_clicks:
        raise PreventUpdate

    store_ids = selected_stores if selected_stores else None

    df = load_data(store_ids, start_date, end_date)
    if df.empty:
        return "No data", "No data", "No data"

    total_sales = df['total'].sum()
    total_orders = df['total'].count()
    avg_order_value = total_sales / total_orders if total_orders else 0

    return f"${total_sales:,.2f}", f"{total_orders:,}", f"${avg_order_value:,.2f}"

@app.callback(
    Output('store-map', 'figure'),
    Input('tabs', 'active_tab'),
    Input('selected-stores-data', 'data')
)
def update_store_map(active_tab, selected_stores):
    if active_tab != "tab-sales":
        raise PreventUpdate

    df = get_store_locations()

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        text="storeid",
        hover_name="city",
        custom_data=["storeid"],
        zoom=4.5,
        height=500,
        title="Store Locations"
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=37.0, lon=-116),
            zoom=4.5
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    fig.update_traces(marker=dict(size=12))

    fig.for_each_trace(
        lambda trace: trace.update(
            marker=dict(
                color=[color_map[store_id.item()] for store_id in trace.customdata]
            )
        )
    )

    return fig

@app.callback(
    Output('graph', 'figure'),
    [Input('selected-stores-data', 'data'),
     Input('load-data-btn', 'n_clicks'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_data(selected_stores, n_clicks, start_date, end_date):
    if not n_clicks:
        raise PreventUpdate

    store_ids = selected_stores if selected_stores else None

    df = load_data(store_ids, start_date, end_date)
    if df.empty:
        fig = px.line(title="No data available")
        return fig

    df['orderdate'] = pd.to_datetime(df['orderdate']).dt.tz_convert('Europe/Berlin')
    df = df[df['total'] > 0]
    df['orderdate'] = df['orderdate'].dt.to_period('M').dt.to_timestamp()
    df_line = df.groupby(['storeid', 'orderdate'])['total'].sum().reset_index()

    fig = px.line(
        df_line,
        x='orderdate',
        y='total',
        color='storeid',
        title='Total Sales by Store',
        labels={'orderdate': 'Order Date', 'total': 'Total Sales', 'storeid': 'Store ID'}
    )

    for store_id in df_line['storeid'].unique():
        fig.for_each_trace(
            lambda trace: trace.update(
                line=dict(color=color_map[store_id])
            ) if trace.name == str(store_id) else ()
        )

    fig.update_layout(
        xaxis_title='Order Date',
        yaxis_title='Total Sales',
        xaxis=dict(
            tickformat='%Y-%m',
            tickmode='array',
            tickvals=df_line['orderdate'].unique(),
            ticktext=[d.strftime('%Y-%m') for d in df_line['orderdate'].unique()],
            tickangle=45
        ),
        yaxis=dict(range=[0, df_line['total'].max() + 10]),
        height=500,
        margin=dict(l=40, r=20, t=40, b=100),
        template='plotly_white'
    )

    return fig

@app.callback(
    Output('heatmap', 'figure'),
    [Input('selected-stores-data', 'data'),
     Input('load-data-btn', 'n_clicks'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_heatmap(selected_stores, n_clicks, start_date, end_date):
    if not n_clicks:
        raise PreventUpdate

    store_ids = selected_stores if selected_stores else None

    df = load_data(store_ids, start_date, end_date)
    if df.empty:
        fig = px.imshow([[0]], x=["No Data"], y=["No Data"], title="No data available")
        return fig

    df['orderdate'] = pd.to_datetime(df['orderdate']).dt.tz_convert('America/Los_Angeles')
    df['weekday'] = df['orderdate'].dt.day_name()
    df['hour'] = df['orderdate'].dt.hour

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['weekday'] = pd.Categorical(df['weekday'], categories=days_order, ordered=True)

    sales_pivot = df.pivot_table(index='weekday', columns='hour', values='total', aggfunc='sum')
    fig = px.imshow(sales_pivot, title='Sales Heatmap',
                    labels={'x': 'Hour of Day (Pacific Time)', 'y': 'Day of Week', 'color': 'Sales'})

    return fig

@app.callback(
    Output('store-comparison', 'figure'),
    [Input('selected-stores-data', 'data'),
     Input('load-data-btn', 'n_clicks'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_store_comparison(selected_stores, n_clicks, start_date, end_date):
    if not n_clicks:
        raise PreventUpdate

    store_ids = selected_stores if selected_stores else None

    df = load_data(store_ids, start_date, end_date)
    if df.empty:
        fig = px.bar(title="No data available")
        return fig

    df['orderdate'] = pd.to_datetime(df['orderdate']).dt.tz_convert('Europe/Berlin')
    df = df[df['total'] > 0]
    df_summary = df.groupby('storeid').agg({'total': 'sum'}).reset_index()

    fig = px.bar(df_summary, x='storeid', y='total', title='Sales Comparison by Store',
                 labels={'storeid': 'Store ID', 'total': 'Total Sales'})

    return fig

@app.callback(
    Output('cluster-graph', 'figure'),
    Output('expenses-graph', 'figure'),
    [Input('cluster-dropdown', 'value'),
     Input('date-slider', 'value')]
)
def update_cluster_graphs(selected_cluster, date_range):
    filtered_orders = orders[
        (orders['orderdate'].dt.year >= date_range[0]) & (orders['orderdate'].dt.year <= date_range[1])]

    if selected_cluster != 'all':
        filtered_customers = customers[customers['cluster'] == selected_cluster].copy()
        filtered_orders = filtered_orders[filtered_orders['cluster'] == selected_cluster].copy()
    else:
        filtered_customers = customers.copy()

    filtered_customers['cluster'] = pd.Categorical(filtered_customers['cluster'], categories=['A', 'B', 'C'],
                                                   ordered=True)
    filtered_orders['cluster'] = pd.Categorical(filtered_orders['cluster'], categories=['A', 'B', 'C'], ordered=True)

    total_spending_per_customer = filtered_orders.groupby('customerid')['total'].sum().reset_index()
    total_spending_per_customer.columns = ['customerid', 'total_spending']
    filtered_customers = filtered_customers.merge(total_spending_per_customer, on='customerid', how='left')
    filtered_customers['total_spending'] = filtered_customers['total_spending'].fillna(0)

    cluster_colors = {'A': '#636EFA', 'B': 'rgba(239, 85, 59, 0.5)',
                      'C': '#00CC96'}

    fig_cluster = px.scatter_mapbox(
        filtered_customers, lat='latitude', lon='longitude', color='cluster',
        title='Customer Segments based on Geographic Data',
        mapbox_style="open-street-map", zoom=5, height=500,
        color_discrete_map=cluster_colors,
        size='total_spending',
        size_max=15,
        opacity=0.7
    )

    fig_cluster.update_layout(
        mapbox=dict(center=dict(lat=filtered_customers['latitude'].mean(), lon=filtered_customers['longitude'].mean())),
        legend=dict(traceorder='normal')
    )

    segment_expenses = filtered_orders.groupby(['cluster', 'category'], observed=True).agg(
        {'total': 'sum'}).reset_index()
    fig_expenses = px.bar(
        segment_expenses, x='category', y='total', color='cluster',
        title='Expenses by Customer Segment and Product Category',
        color_discrete_map=cluster_colors
    )

    return fig_cluster, fig_expenses

@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('forecast-store-dropdown', 'value')]
)
def forecast_sales(store_id):
    if not store_id:
        raise PreventUpdate

    df = load_data([store_id], start_date='1970-01-01', end_date='2100-01-01')
    if df.empty:
        fig = px.line(title="No data available")
        return fig

    df = df.set_index('orderdate')
    df = df.resample('M').sum()

    try:
        model = ARIMA(df['total'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
        model_fit = model.fit()
    except Exception as e:
        fig = px.line(title='Forecasting failed due to insufficient data or other issues.')
        return fig

    forecast = model_fit.forecast(steps=12)
    forecast_dates = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='M')

    forecast_df = pd.DataFrame({'orderdate': forecast_dates, 'forecast': forecast})
    forecast_df.set_index('orderdate', inplace=True)

    combined_df = pd.concat([df, forecast_df])

    connection_df = pd.DataFrame({
        'orderdate': [df.index[-1], forecast_df.index[0]],
        'sales': [df['total'].iloc[-1], forecast_df['forecast'].iloc[0]],
        'type': ['connection', 'connection']
    })

    combined_df.reset_index(inplace=True)
    combined_df['type'] = 'total'
    forecast_df.reset_index(inplace=True)
    forecast_df['type'] = 'forecast'

    combined_df = combined_df.rename(columns={'total': 'sales'})
    forecast_df = forecast_df.rename(columns={'forecast': 'sales'})

    melted_df = pd.concat(
        [combined_df[['orderdate', 'sales', 'type']], forecast_df[['orderdate', 'sales', 'type']], connection_df])

    city = get_store_locations().set_index('storeid').loc[store_id, 'city']
    fig = px.line(melted_df, x='orderdate', y='sales', color='type', labels={'sales': 'Sales', 'orderdate': 'Date'},
                  title=f'Sales Forecast for Store {store_id} ({city}) for the Next 12 Months')

    fig.update_layout(
        xaxis=dict(
            tickformat='%Y-%m',
            tickmode='array',
            tickvals=melted_df['orderdate'],
            ticktext=melted_df['orderdate'].dt.strftime('%Y-%m'),
            tickangle=45
        ),
        height=500,
        margin=dict(l=40, r=20, t=40, b=100),
        template='plotly_white',
        showlegend=True
    )

    for trace in fig.data:
        if trace.name == 'forecast':
            forecast_color = trace.line.color
        if trace.name == 'connection':
            trace.update(showlegend=False)
            trace.update(line=dict(color=forecast_color))

    return fig

@app.callback(
    Output('comparison-graph', 'figure'),
    [Input('store-dropdown', 'value')]
)
def update_graph(selected_store_id):
    if selected_store_id is None:
        return {}

    selected_store_data = store_features[store_features['storeid'] == selected_store_id]

    comparison_df = pd.DataFrame({
        'Metric': ['Average Distance to Store', 'Unique Customers', 'Average Spending per Customer'],
        'Ibapah': [
            store_features[store_features['storeid'] == ibapah_store_id]['avg_distance'].values[0],
            store_features[store_features['storeid'] == ibapah_store_id]['unique_customers'].values[0],
            store_features[store_features['storeid'] == ibapah_store_id]['avg_spending_per_customer'].values[0]
        ],
        'Selected Store': [
            selected_store_data['avg_distance'].values[0],
            selected_store_data['unique_customers'].values[0],
            selected_store_data['avg_spending_per_customer'].values[0]
        ]
    })

    fig = px.bar(
        comparison_df,
        x='Metric',
        y=['Ibapah', 'Selected Store'],
        barmode='group',
        title=f'Comparison of Metrics: Ibapah vs. Selected Store',
        labels={'value': 'Value', 'Metric': 'Metric'},
        height=600,
        width=1000
    )

    return fig

@app.callback(
    Output('product-animation', 'figure'),
    Output('store-animation', 'figure'),
    Input('tabs', 'active_tab')
)
def update_animations(active_tab):
    if active_tab != 'tab-animations':
        raise PreventUpdate

    fig_product = px.bar(
        product_sales_df,
        x='total_sales',
        y='name',
        color='name',
        animation_frame=product_sales_df['year_month'].astype(str),
        title='Product Sales Over Time',
        labels={'total_sales': 'Total Sales', 'name': 'Product'},
        orientation='h',
        height=400,
        width=600
    )

    fig_product.update_layout(yaxis={'categoryorder': 'total ascending'})

    fig_store = px.bar(
        store_sales_df,
        x='total_sales',
        y='city',
        color='city',
        animation_frame=store_sales_df['year_month'].astype(str),
        title='Store Sales Over Time',
        labels={'total_sales': 'Total Sales', 'city': 'Store'},
        orientation='h',
        height=400,
        width=600
    )

    fig_store.update_layout(yaxis={'categoryorder': 'total ascending'})

    return fig_product, fig_store

@app.callback(
    Output('itemset-graph', 'figure'),
    [Input('itemset-size', 'value')]
)
def update_graph(selected_size):
    if selected_size == 1:
        itemsets = frequent_itemsets_1
        title = 'Top 10 Frequently Purchased Single Products'
        color = 'skyblue'
    elif selected_size == 2:
        itemsets = frequent_itemsets_2
        title = 'Top 10 Frequently Purchased Product Pairs'
        color = 'lightgreen'
    elif selected_size == 3:
        itemsets = frequent_itemsets_3
        title = 'Top 10 Frequently Purchased Product Triplets'
        color = 'lightcoral'
    else:
        return {}

    if len(itemsets) == 0:
        return {}

    top_10_itemsets = itemsets.nlargest(10, 'support')
    fig = px.bar(
        top_10_itemsets,
        x='support',
        y=top_10_itemsets['itemsets'].apply(lambda x: ', '.join(list(x))),
        orientation='h',
        title=title,
        labels={'support': 'Support (Proportion of Transactions)', 'y': 'Itemsets'},
        color_discrete_sequence=[color]
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

@app.callback(
    Output('pizza-count-graph', 'figure'),
    [Input('itemset-size', 'value')]
)
def update_pizza_count_graph(selected_size):
    orderitems = pd.read_sql('SELECT * FROM orders_items', engine)
    orders = pd.read_sql('SELECT * FROM orders', engine)
    order_data = pd.merge(orderitems, orders, on='orderid')

    customer_pizza_counts = order_data.groupby('customerid').size().reset_index(name='pizza_count')

    single_pizza_customers = customer_pizza_counts[customer_pizza_counts['pizza_count'] == 1].shape[0]
    multiple_pizza_customers = customer_pizza_counts[customer_pizza_counts['pizza_count'] > 1].shape[0]

    pizza_purchase_data = pd.DataFrame({
        'Pizza Count': ['1 Pizza', '>1 Pizza'],
        'Number of Customers': [single_pizza_customers, multiple_pizza_customers]
    })

    fig = px.bar(pizza_purchase_data,
                 x='Pizza Count',
                 y='Number of Customers',
                 title='Number of Customers Buying One Pizza vs. More Than One Pizza',
                 labels={'Pizza Count': 'Number of Pizzas Purchased', 'Number of Customers': 'Number of Customers'},
                 color='Pizza Count')

    return fig

@app.callback(
    Output('veggie-sales-bar-graph', 'figure'),
    Output('veggie-sales-pie-chart', 'figure'),
    Output('geographic-sales-graph', 'figure'),
    [Input('itemset-size', 'value')]
)
def update_veggie_sales_graph(selected_size):
    sales_per_store = load_veggie_pizza_data()

    avg_veggie_sales_percentage = sales_per_store['veggie_sales_percentage'].mean()
    avg_veggie_sales_percentage = round(avg_veggie_sales_percentage, 2)

    bar_fig = px.bar(
        sales_per_store,
        x='city',
        y=['veggie_sales', 'total_sales'],
        barmode='group',
        labels={'value': 'Sales', 'variable': 'Type', 'city': 'Store'},
        title='Veggie Pizza Sales vs Total Sales per Store',
        color_discrete_map={'veggie_sales': 'green', 'total_sales': 'blue'}
    )

    pie_fig = px.pie(
        names=['Veggie Sales', 'Other Sales'],
        values=[avg_veggie_sales_percentage, 100 - avg_veggie_sales_percentage],
        title='Average Share of Veggie Sales in Total Sales'
    )

    geographic_sales = sales_per_store.groupby('city')['total_sales'].sum().reset_index()

    geo_fig = px.bar(
        geographic_sales,
        x='city',
        y='total_sales',
        labels={'city': 'City', 'total_sales': 'Total Sales'},
        title='Geographic Distribution of Sales'
    )

    return bar_fig, pie_fig, geo_fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
