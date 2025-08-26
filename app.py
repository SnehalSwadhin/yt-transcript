import streamlit as st
import pandas as pd
import libsql
import os
import dotenv
from database import initialize_db

dotenv.load_dotenv()

DB_URL = os.environ.get("TURSO_DATABASE_URL")
DB_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")

initialize_db()

# --- Page Configuration ---
st.set_page_config(
    page_title="Bangalore Second Hand Cars",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
# Use st.cache_data to load data only once and store it in cache
@st.cache_data
def load_data(db_path):
    """Loads car data from the SQLite database."""
    try:
        # Use a SQL query to select all data from the 'cars' table
        conn = libsql.connect("local.db", sync_url=DB_URL, auth_token=DB_TOKEN)
        query = "SELECT * FROM cars WHERE published_at > datetime('now', '-30 days')"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # --- Data Cleaning and Preparation ---
        # # Convert price to numeric, coercing errors to NaN
        # df['price'] = pd.to_numeric(df['price'], errors='coerce')
        # # Drop rows where price is not available, as it's a key feature
        # df.dropna(subset=['price'], inplace=True)
        # # Convert price to integer for cleaner display
        # df['price'] = df['price'].astype(int)
        
        # Convert published_date to datetime objects for sorting
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
        # Fill missing values for categorical columns to avoid errors in filters
        for col in ['car_oem', 'model', 'variant', 'colour', 'dealer_location']:
            df[col] = df[col].fillna('N/A')
            
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()
        return None

# Load the data
df = load_data('car_data.db')
df = df.replace(r'^\s*(n/a|nan|null|na|NA|NaN|Null|N/a|N/A)\s*$', 'N/A', regex=True)
df = df[df['car_oem'] != 'N/A']  # Filter out rows where car_oem is 'N/A'

# --- Sidebar Filters ---
st.sidebar.header("Filter Cars 🕵️")

# Price Range Slider
min_price, max_price = int(df['price_numeric'].min()), int(df['price_numeric'].max())
selected_price_range = st.sidebar.slider(
    'Price Range (in Lakhs)',
    min_value=min_price // 100000,
    max_value=(max_price // 100000) + 1,
    value=(min_price // 100000, (max_price // 100000) + 1),
    key='price_slider'
)
# Convert back to actual price for filtering
min_selected_price = selected_price_range[0] * 100000
max_selected_price = selected_price_range[1] * 100000

# Brand (OEM) Multiselect
sorted_oems = sorted(df['car_oem'].unique())
selected_oems = st.sidebar.multiselect(
    'Car Brand (OEM)',
    options=sorted_oems,
    default=[],
    key='oem_select'
)

# Model Multiselect (Dynamic based on selected OEM)
if selected_oems:
    # Filter models based on selected brands
    available_models = df[df['car_oem'].isin(selected_oems)]['model'].unique()
else:
    # Show all models if no brand is selected
    available_models = df['model'].unique()

sorted_models = sorted(available_models)
selected_models = st.sidebar.multiselect(
    'Model',
    options=sorted_models,
    default=[],
    key='model_select'
)

# Location Multiselect
sorted_locations = sorted(df['dealer_location'].unique())
selected_locations = st.sidebar.multiselect(
    'Dealer Location',
    options=sorted_locations,
    default=[],
    key='location_select'
)

#--- Filtering Logic ---
filtered_df = df[
    (df['price_numeric'] >= min_selected_price) & (df['price_numeric'] <= max_selected_price)
]

if selected_oems:
    filtered_df = filtered_df[filtered_df['car_oem'].isin(selected_oems)]

if selected_models:
    filtered_df = filtered_df[filtered_df['model'].isin(selected_models)]

if selected_locations:
    filtered_df = filtered_df[filtered_df['dealer_location'].isin(selected_locations)]

# --- Main Page Display ---
st.title("🚗 Second Hand Car Listings - Bangalore")
st.markdown("Data scraped from daily YouTube video listings.")

# --- Sorting Options ---
sort_option = st.selectbox(
    "Sort results by:",
    ("Newest First", "Price: Low to High", "Price: High to Low"),
    key='sort'
)

if sort_option == "Newest First":
    # Sort by the 'published_date' column, with newest (latest) dates first
    filtered_df = filtered_df.sort_values(by='published_at', ascending=False)
elif sort_option == "Price: Low to High":
    filtered_df = filtered_df.sort_values(by='price', ascending=True)
else: # "Price: High to Low"
    filtered_df = filtered_df.sort_values(by='price', ascending=False)

# Get a list of unique video IDs, preserving the sort order from the step above
unique_video_ids = filtered_df['video_id'].unique()

# Display a summary of the results
st.header(f"Found {len(filtered_df)} cars across {len(unique_video_ids)} videos")
st.markdown("---")

# --- Display Results as Cards ---
if filtered_df.empty:
    st.warning("No cars found. Try adjusting the filters!")
else:
    # for video_id in unique_video_ids:
    #     # Get all rows (cars) that belong to the current video
    #     video_cars_df = filtered_df[filtered_df['video_id'] == video_id].copy()
        
    #     # Get common information from the first car in the group
    #     # (assuming dealer and date are the same for all cars in one video)
    #     first_car = video_cars_df.iloc[0]
    #     dealer_name = first_car['dealer_name']
    #     dealer_location = first_car['dealer_location']
    #     dealer_contact = first_car['dealer_contact']
    #     dealer_email = first_car['dealer_email']
    #     dealer_website = first_car['dealer_email']

    #     video_cars_df['price_lakhs'] = video_cars_df['price_numeric'] / 100000.0
    #     video_cars_df['price_lakhs'] = video_cars_df['price_lakhs'].apply(lambda x: f"₹{x:.2f} Lakhs" if x >= 0.1 else "N/A")
    #     video_cars_df = video_cars_df[['car_oem', 'model', 'variant', 'price_lakhs', 'colour', 'odometer', 'year', 'service_record', 'frame_type', 'transmission_type', 'fuel_type', 'num_owners', 'rto', 'city', 'engine_details', 'feature_details', 'rating', 'start_timestamp', 'video_link', 'price']]
    #     video_cars_df = video_cars_df.dropna(axis=1, how='all')
        
    #     # Create a container for each video's section
    #     with st.container(border=True):
    #         # --- Video Header ---
    #         col1, col2 = st.columns([1, 2])
    #         with col1:
    #             base_video_url = f"https://www.youtube.com/watch?v={video_id}"
    #             st.video(base_video_url)
            
    #         with col2:
    #             st.subheader(first_car['video_title'])
    #             st.markdown(f"**Channel:** {first_car['channel_title']}")
    #             if pd.notna(first_car['published_at']):
    #                 st.markdown(f"**Published:** {first_car['published_at'].strftime('%d %B, %Y')}")
                
    #             st.markdown(f"""**Dealer:** {dealer_name} ({dealer_location})
    #                         {'📞' + dealer_contact if dealer_contact is not None else ''}
    #                         {'📧' + dealer_email if dealer_email is not None else ''}
    #                         {'[🌐' + dealer_website + '](' + dealer_website + ')' if dealer_website is not None else ''}""")
            
    #         st.markdown("##### Cars Featured in This Video:")
            
    #         st.dataframe(
    #             video_cars_df,
    #             column_config={
    #                 'car_oem': st.column_config.TextColumn("Brand"),
    #                 'model': st.column_config.TextColumn("Model"),
    #                 'variant': st.column_config.TextColumn("Variant"),
    #                 'price_lakhs': st.column_config.TextColumn("Price (lakhs)"),
    #                 'colour': st.column_config.TextColumn("Colour"),
    #                 'price': st.column_config.TextColumn("Price (raw)"),
    #                 'odometer': st.column_config.TextColumn("Odometer (km)"),
    #                 'year': st.column_config.TextColumn("Year"),
    #                 'service_record': st.column_config.TextColumn("Service Record"),
    #                 'frame_type': st.column_config.TextColumn("Frame Type"),
    #                 'transmission_type': st.column_config.TextColumn("Transmission Type"),
    #                 'fuel_type': st.column_config.TextColumn("Fuel Type"),
    #                 'num_owners': st.column_config.TextColumn("Number of Owners"),
    #                 'rto': st.column_config.TextColumn("RTO"),
    #                 'city': st.column_config.TextColumn("City"),
    #                 'engine_details': st.column_config.TextColumn("Engine Details"),
    #                 'feature_details': st.column_config.TextColumn("Feature Details"),
    #                 'rating': st.column_config.TextColumn("Rating"),
    #                 'start_timestamp': st.column_config.TextColumn("Start Timestamp"),
    #                 'video_link': st.column_config.LinkColumn("Video Link", help="Link to the YouTube video"),
    #             },
    #             hide_index=True,
    #             use_container_width=True,
    #         )
    display_df = filtered_df.copy()
    display_df['price_lakhs'] = display_df['price_numeric'] / 100000.0
    display_df['price_lakhs'] = display_df['price_lakhs'].apply(lambda x: f"₹{x:.2f} Lakhs" if x >= 0.1 else "N/A")
    
    # Select and order columns for display
    columns_to_display = [
        'car_oem', 'model', 'variant', 'price_lakhs', 'colour', 'odometer', 
        'year', 'service_record', 'frame_type', 'transmission_type', 'fuel_type', 
        'num_owners', 'rto', 'city', 'engine_details', 'feature_details', 
        'rating', 'dealer_name', 'dealer_location', 'dealer_contact',
        'video_link', 'published_at'
    ]
    
    display_df = display_df[columns_to_display].dropna(axis=1, how='all')
    
    st.dataframe(
        display_df,
        column_config={
            'car_oem': st.column_config.TextColumn("Brand"),
            'model': st.column_config.TextColumn("Model"),
            'variant': st.column_config.TextColumn("Variant"),
            'price_lakhs': st.column_config.TextColumn("Price (lakhs)"),
            'colour': st.column_config.TextColumn("Colour"),
            'odometer': st.column_config.TextColumn("Odometer (km)"),
            'year': st.column_config.TextColumn("Year"),
            'service_record': st.column_config.TextColumn("Service Record"),
            'frame_type': st.column_config.TextColumn("Frame Type"),
            'transmission_type': st.column_config.TextColumn("Transmission Type"),
            'fuel_type': st.column_config.TextColumn("Fuel Type"),
            'num_owners': st.column_config.TextColumn("Number of Owners"),
            'rto': st.column_config.TextColumn("RTO"),
            'city': st.column_config.TextColumn("City"),
            'engine_details': st.column_config.TextColumn("Engine Details"),
            'feature_details': st.column_config.TextColumn("Feature Details"),
            'rating': st.column_config.TextColumn("Rating"),
            'dealer_name': st.column_config.TextColumn("Dealer Name"),
            'dealer_location': st.column_config.TextColumn("Dealer Location"),
            'dealer_contact': st.column_config.TextColumn("Dealer Contact"),
            'video_link': st.column_config.LinkColumn("Video Link"),
            'published_at': st.column_config.DateColumn("Published Date"),
        },
        hide_index=True,
        use_container_width=True,
    )

