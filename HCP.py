import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from category_encoders import BinaryEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV,cross_validate,StratifiedKFold
from datetime import date
from groq import Groq
from streamlit_float import *
st.set_page_config(
    page_title="Hotel Booking Prediction",
    layout="wide"
)

float_init()
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    position: relative;
    height: 75vh;
    border-radius: 28px;
    overflow: hidden;
    background-image:
        linear-gradient(rgba(15,23,42,0.55), rgba(15,23,42,0.55)),
        url("https://images.unsplash.com/photo-1566073771259-6a8506099945");
    background-size: cover;
    background-position: center;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.25);
}

.hero-content {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 70px;
    color: white;
}

.hero h1 {
    font-size: 58px;
    font-weight: 800;
    margin-bottom: 16px;
    color: white;
}

.hero p {
    font-size: 21px;
    max-width: 760px;
    line-height: 1.6;
    color: #e5e7eb;
}

.badge {
    background: rgba(255,255,255,0.18);
    padding: 10px 18px;
    border-radius: 30px;
    width: fit-content;
    margin-bottom: 18px;
    font-weight: 600;
}

.custom-card {
    background: rgba(255,255,255,0.95);
    padding: 24px;
    border-radius: 22px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    border: 1px solid #e2e8f0;
    margin-top: 25px;
}

.card-title {
    color: #0f172a;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 10px;
}

.card-text {
    color: #475569;
    font-size: 16px;
    line-height: 1.7;
}

.kpi-card {
    background: white;
    padding: 22px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    border: 1px solid #e2e8f0;
}

.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: #2563eb;
}

.kpi-title {
    font-size: 15px;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    HBCP = pd.read_csv("hotel_bookings.csv")
    

    if HBCP["is_canceled"].dtype == object:
        HBCP["is_canceled"] = HBCP["is_canceled"].replace({
            "Canceled": 1,
            "Not Canceled": 0,
            "1": 1,
            "0": 0
        })

    HBCP["is_canceled"] = pd.to_numeric(
        HBCP["is_canceled"],
        errors="coerce"
    )

    return HBCP
HBCP = load_data()
@st.cache_resource
def load_model():
    return joblib.load("HBCP.pkl")

model = load_model()

with open("Project_summary_HCP.md", "r", encoding="utf-8") as f:
    SUMMARY = f.read()

SYSTEM_PROMPT = f"""
You are a Hotel Booking Management Assistant for hotel managers.

Answer questions about the project using only the documentation below.
Focus on business insights, cancellation analysis, booking behavior, and management decisions.
If the answer is not available in the documentation, say that clearly.

{SUMMARY}
"""

st.markdown("""
<style>
.chatbot-box {
    position: fixed;
    right: 20px;
    bottom: 80px;
    width: 360px;
    max-height: 620px;
    background: white;
    border-radius: 18px;
    box-shadow: 0 10px 35px rgba(15, 23, 42, 0.25);
    border: 1px solid #e2e8f0;
    padding: 18px;
    z-index: 9999;
}
.chatbot-title {
    font-size: 20px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
}
.chatbot-text {
    font-size: 14px;
    color: #475569;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Analysis", "Prediction"]
)

main_col = st.container()
if page == "Home":
        st.markdown("""
        <div class="hero">
            <div class="hero-content">
                <div class="badge">Machine Learning Project</div>
                <h1>Hotel Booking Analytics & Prediction</h1>
                <p>
                    Smart hotel booking analysis and cancellation prediction powered by machine learning.
                    This project helps hotels understand booking behavior, identify cancellation patterns,
                    and improve business decision-making.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Project Description</div>
            <div class="card-text">
                This project analyzes hotel booking data and builds a machine learning model to predict
                whether a customer booking will be canceled or not.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{HBCP.shape[0]:,}</div>
                <div class="kpi-title">Total Bookings</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{HBCP.shape[1]}</div>
                <div class="kpi-title">Dataset Features</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            cancel_rate = HBCP["is_canceled"].mean() * 100
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{cancel_rate:.1f}%</div>
                <div class="kpi-title">Cancellation Rate</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Problem Statement</div>
            <div class="card-text">
                Hotel cancellations can cause revenue loss and poor resource planning.
                The goal of this project is to predict booking cancellations early using customer,
                reservation, and booking-related features.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Dataset Overview</div>
            <div class="card-text">
                The dataset contains hotel booking records from city and resort hotels.
                It includes features such as lead time, arrival date, number of guests,
                meal type, market segment, room type, deposit type, customer type,
                special requests, and cancellation status.
                The target variable is <b>is_canceled</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View Sample Data"):
            st.dataframe(HBCP.head(), use_container_width=True)

elif page == "Analysis":

        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Hotel Booking Analysis</div>
            <div class="card-text">
                Explore cancellation patterns by hotel type, customer behavior, market segment,
                deposit type, guest type, country, and booking time.
            </div>
        </div>
        """, unsafe_allow_html=True)

        HBCP = HBCP.copy()

        HBCP["is_canceled_label"] = HBCP["is_canceled"].map({
            0: "Not Canceled",
            1: "Canceled"
        })

        st.sidebar.header("Analysis Filters")

        year_filter = st.sidebar.slider(
            "Arrival Year",
            min_value=int(HBCP["arrival_date_year"].min()),
            max_value=int(HBCP["arrival_date_year"].max()),
            value=(
                int(HBCP["arrival_date_year"].min()),
                int(HBCP["arrival_date_year"].max())
            )
        )

        hotel_filter = st.sidebar.multiselect(
            "Hotel",
            options=HBCP["hotel"].unique(),
            default=HBCP["hotel"].unique()
        )

        HBCP = HBCP[
            (HBCP["arrival_date_year"].between(year_filter[0], year_filter[1])) &
            (HBCP["hotel"].isin(hotel_filter))
        ]

        def beautify_fig(fig):
            fig.update_layout(
                template="plotly_white",
                title=dict(font=dict(size=22, color="#0f172a")),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial", size=13, color="#334155"),
                margin=dict(l=20, r=20, t=60, b=20)
            )
            fig.update_xaxes(showgrid=False, linecolor="#cbd5e1")
            fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", linecolor="#cbd5e1")
            return fig

        tab1, tab2, tab3, tab4 = st.tabs([
            "Cancellation Overview",
            "Customer Behavior",
            "Booking Channels",
            "Time & Price Analysis"
        ])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    HBCP,
                    x="is_canceled_label",
                    text_auto=True,
                    title="Booking Cancellation Distribution"
                )
                st.plotly_chart(beautify_fig(fig), use_container_width=True)

            with col2:
                highest_hotel_cancel = (
                    HBCP[HBCP["is_canceled_label"] == "Canceled"]
                    .groupby("hotel")["is_canceled_label"]
                    .count()
                    .sort_values(ascending=False)
                    .reset_index(name="cancel_count")
                )

                fig1 = px.bar(
                    highest_hotel_cancel,
                    x="hotel",
                    y="cancel_count",
                    text="cancel_count",
                    title="Highest Hotel Cancellation Count"
                )
                st.plotly_chart(beautify_fig(fig1), use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                cancel_rate_deposit = (
                    HBCP.groupby("deposit_type")["is_canceled_label"]
                    .apply(lambda x: round((x == "Canceled").mean() * 100, 3))
                    .reset_index(name="Cancellation_Rate")
                )

                fig4 = px.bar(
                    cancel_rate_deposit,
                    x="deposit_type",
                    y="Cancellation_Rate",
                    text="Cancellation_Rate",
                    title="Deposit Type Effect on Cancellation"
                )
                st.plotly_chart(beautify_fig(fig4), use_container_width=True)

            with col4:
                if "reservation_status" in HBCP.columns:
                    fig_reservation = px.pie(
                        HBCP,
                        names="reservation_status",
                        title="Reservation Status Distribution"
                    )
                    st.plotly_chart(
                        beautify_fig(fig_reservation),
                        use_container_width=True
                    )
                else:
                    st.warning("Column 'reservation_status' not found in the dataset.")

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                cancel_rate_repeated = (
                    HBCP.groupby("is_repeated_guest")["is_canceled_label"]
                    .apply(lambda x: round((x == "Canceled").mean() * 100, 3))
                    .reset_index(name="Cancellation_Rate")
                )

                fig3 = px.bar(
                    cancel_rate_repeated,
                    x="is_repeated_guest",
                    y="Cancellation_Rate",
                    text="Cancellation_Rate",
                    title="Cancellation Rate: Repeated vs New Guests"
                )
                st.plotly_chart(beautify_fig(fig3), use_container_width=True)

            with col2:
                cancel_rate_requests = (
                    HBCP.groupby("total_of_special_requests")["is_canceled_label"]
                    .apply(lambda x: round((x == "Canceled").mean() * 100, 3))
                    .reset_index(name="Cancellation_Rate")
                )

                fig5 = px.bar(
                    cancel_rate_requests,
                    x="total_of_special_requests",
                    y="Cancellation_Rate",
                    text="Cancellation_Rate",
                    title="Special Requests Effect on Cancellation"
                )
                st.plotly_chart(beautify_fig(fig5), use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                fig_customer = px.pie(
                    HBCP,
                    names="customer_type",
                    title="Customer Type Distribution"
                )
                st.plotly_chart(beautify_fig(fig_customer), use_container_width=True)

            with col4:
                cancel_rate_customer = (
                    HBCP.groupby("customer_type")["is_canceled_label"]
                    .apply(lambda x: round((x == "Canceled").mean() * 100, 3))
                    .reset_index(name="Cancellation_Rate")
                )

                fig6 = px.bar(
                    cancel_rate_customer,
                    x="customer_type",
                    y="Cancellation_Rate",
                    text="Cancellation_Rate",
                    title="Cancellation Rate by Customer Type"
                )
                st.plotly_chart(beautify_fig(fig6), use_container_width=True)

            HBCP["guest_type"] = np.select(
                [
                    (HBCP["children"] == 0) & (HBCP["babies"] == 0),
                    (HBCP["children"] > 0) & (HBCP["babies"] == 0),
                    (HBCP["children"] == 0) & (HBCP["babies"] > 0),
                    (HBCP["children"] > 0) & (HBCP["babies"] > 0)
                ],
                [
                    "Adults only",
                    "Adults with children",
                    "Adults with babies",
                    "Adults with children and babies"
                ],
                default="Other"
            )

            col5, col6 = st.columns(2)

            with col5:
                cancel_rate_guest = (
                    HBCP.groupby("guest_type")["is_canceled_label"]
                    .apply(lambda x: round((x == "Canceled").mean() * 100, 3))
                    .reset_index(name="Cancellation_Rate")
                )

                fig_guest = px.bar(
                    cancel_rate_guest,
                    x="guest_type",
                    y="Cancellation_Rate",
                    text="Cancellation_Rate",
                    title="Cancellation Rate by Guest Type"
                )
                st.plotly_chart(beautify_fig(fig_guest), use_container_width=True)

            with col6:
                lead_time_guest = (
                    HBCP.groupby("guest_type")["lead_time"]
                    .mean()
                    .round(2)
                    .reset_index()
                )

                fig_lead = px.bar(
                    lead_time_guest,
                    x="guest_type",
                    y="lead_time",
                    text="lead_time",
                    title="Average Lead Time by Guest Type"
                )
                st.plotly_chart(beautify_fig(fig_lead), use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                highest_market_segment_cancel = (
                    HBCP[HBCP["is_canceled_label"] == "Canceled"]
                    .groupby("market_segment")["is_canceled_label"]
                    .count()
                    .sort_values(ascending=False)
                    .reset_index(name="Cancellation")
                )

                fig_market = px.bar(
                    highest_market_segment_cancel,
                    x="market_segment",
                    y="Cancellation",
                    text="Cancellation",
                    title="Market Segment with Most Cancellations"
                )
                st.plotly_chart(beautify_fig(fig_market), use_container_width=True)

            with col2:
                highest_distribution_channel_cancel = (
                    HBCP[HBCP["is_canceled_label"] == "Canceled"]
                    .groupby("distribution_channel")["is_canceled_label"]
                    .count()
                    .sort_values(ascending=False)
                    .reset_index(name="Cancellation")
                )

                fig_channel_cancel = px.bar(
                    highest_distribution_channel_cancel,
                    x="distribution_channel",
                    y="Cancellation",
                    text="Cancellation",
                    title="Distribution Channel Cancellation Count"
                )
                st.plotly_chart(beautify_fig(fig_channel_cancel), use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                fig_channel = px.pie(
                    HBCP,
                    names="distribution_channel",
                    title="Distribution Channel Distribution"
                )
                st.plotly_chart(beautify_fig(fig_channel), use_container_width=True)

            with col4:
                country_cancel = (
                    HBCP.groupby("country")["is_canceled_label"]
                    .apply(lambda x: round((x == "Canceled").mean() * 100, 3))
                    .reset_index(name="Cancellation_Rate")
                    .sort_values("Cancellation_Rate", ascending=False)
                    .head(15)
                )

                fig_country = px.bar(
                    country_cancel,
                    x="country",
                    y="Cancellation_Rate",
                    text="Cancellation_Rate",
                    title="Top 15 Countries by Cancellation Rate"
                )
                st.plotly_chart(beautify_fig(fig_country), use_container_width=True)

            fig_room = px.histogram(
                HBCP,
                x="reserved_room_type",
                text_auto=True,
                title="Reserved Room Type Distribution"
            )
            st.plotly_chart(beautify_fig(fig_room), use_container_width=True)

        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                cancel_rate_year = (
                    HBCP.groupby("arrival_date_year")["is_canceled_label"]
                    .apply(lambda x: round((x == "Canceled").mean() * 100, 3))
                    .reset_index(name="Cancellation_Rate")
                )

                cancel_rate_year["arrival_date_year"] = cancel_rate_year["arrival_date_year"].astype(str)

                fig_year = px.line(
                    cancel_rate_year,
                    x="arrival_date_year",
                    y="Cancellation_Rate",
                    text="Cancellation_Rate",
                    title="Cancellation Rate by Year",
                    markers=True
                )
                st.plotly_chart(beautify_fig(fig_year), use_container_width=True)

            with col2:
                month_order = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]

                cancel_rate_month = (
                    HBCP.groupby("arrival_date_month")["is_canceled_label"]
                    .apply(lambda x: round((x == "Canceled").mean() * 100, 3))
                    .reset_index(name="Cancellation_Rate")
                )

                cancel_rate_month["arrival_date_month"] = pd.Categorical(
                    cancel_rate_month["arrival_date_month"],
                    categories=month_order,
                    ordered=True
                )

                cancel_rate_month = cancel_rate_month.sort_values("arrival_date_month")

                fig_month = px.line(
                    cancel_rate_month,
                    x="arrival_date_month",
                    y="Cancellation_Rate",
                    text="Cancellation_Rate",
                    title="Cancellation Rate by Month",
                    markers=True
                )
                st.plotly_chart(beautify_fig(fig_month), use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                fig_adr_box = px.box(
                    HBCP,
                    x="adr",
                    y="is_canceled_label",
                    title="ADR by Cancellation Status"
                )
                st.plotly_chart(beautify_fig(fig_adr_box), use_container_width=True)

            with col4:
                adr_cancel = (
                    HBCP.groupby("is_canceled_label")["adr"]
                    .mean()
                    .reset_index()
                    .round(2)
                )

                fig_adr_bar = px.bar(
                    adr_cancel,
                    x="is_canceled_label",
                    y="adr",
                    text="adr",
                    title="Average ADR by Cancellation Status"
                )
                st.plotly_chart(beautify_fig(fig_adr_bar), use_container_width=True)

        with st.expander("View Filtered Data"):
            st.dataframe(HBCP.head(100), use_container_width=True)

elif page == "Prediction":

        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Booking Cancellation Prediction</div>
            <div class="card-text">
                Enter the booking information below to predict whether the reservation will be canceled.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            hotel = st.selectbox("Hotel", HBCP["hotel"].unique())
            market_segment = st.selectbox("Market Segment", HBCP["market_segment"].unique())
            distribution_channel = st.selectbox("Distribution Channel", HBCP["distribution_channel"].unique())
            meal = st.selectbox("Meal", HBCP["meal"].unique())
            customer_type = st.selectbox("Customer Type", HBCP["customer_type"].unique())
            deposit_type = st.selectbox("Deposit Type", HBCP["deposit_type"].unique())
            country = st.selectbox("Country", sorted(HBCP["country"].dropna().unique()))
            reserved_room_type = st.selectbox("Reserved Room Type", HBCP["reserved_room_type"].unique())
            assigned_room_type = st.selectbox("Assigned Room Type", HBCP["assigned_room_type"].unique())

        with col2:
            is_repeated_guest = st.selectbox("Repeated Guest", [0, 1])
            min_year = int(HBCP["arrival_date_year"].min())
            max_year = int(HBCP["arrival_date_year"].max())
            
            arrival_date = st.date_input(
                "Arrival Date",
                value=date(min_year, 1, 1),
                min_value=date(min_year, 1, 1),
                max_value=date(max_year, 12, 31)
            )
            
            arrival_date_month = arrival_date.strftime("%B")
            arrival_date_year = arrival_date.year
            arrival_date_day_of_month = arrival_date.day
            arrival_date_week_number = arrival_date.isocalendar()[1]

            adults = st.number_input("Adults", min_value=0, value=2)
            children = st.number_input("Children", min_value=0, value=0)
            babies = st.number_input("Babies", min_value=0, value=0)
            stays_in_weekend_nights = st.number_input("Stays In Weekend Nights", min_value=0, value=0)
            stays_in_week_nights = st.number_input("Stays In Week Nights", min_value=0, value=1)
            adr = st.number_input("Average Daily Rate", min_value=0.0, value=100.0)

        st.markdown("### More Booking Details")

        col3, col4 = st.columns(2)

        with col3:
            lead_time = st.number_input("Lead Time", min_value=0, value=30)
            previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
            previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
            booking_changes = st.number_input("Booking Changes", min_value=0, value=0)

        with col4:
            days_in_waiting_list = st.number_input("Days In Waiting List", min_value=0, value=0)
            required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, value=0)
            total_of_special_requests = st.number_input("Total Of Special Requests", min_value=0, value=0)

        if st.button("Predict Cancellation"):
            input_HBCP = pd.DataFrame({
                "hotel": [hotel],
                "lead_time": [lead_time],
                "arrival_date_year": [arrival_date_year],
                "arrival_date_month": [arrival_date_month],
                "arrival_date_week_number": [arrival_date_week_number],
                "arrival_date_day_of_month": [arrival_date_day_of_month],
                "stays_in_weekend_nights": [stays_in_weekend_nights],
                "stays_in_week_nights": [stays_in_week_nights],
                "adults": [adults],
                "children": [children],
                "babies": [babies],
                "meal": [meal],
                "country": [country],
                "market_segment": [market_segment],
                "distribution_channel": [distribution_channel],
                "is_repeated_guest": [is_repeated_guest],
                "previous_cancellations": [previous_cancellations],
                "previous_bookings_not_canceled": [previous_bookings_not_canceled],
                "reserved_room_type": [reserved_room_type],
                "assigned_room_type": [assigned_room_type],
                "booking_changes": [booking_changes],
                "deposit_type": [deposit_type],
                "days_in_waiting_list": [days_in_waiting_list],
                "customer_type": [customer_type],
                "adr": [adr],
                "required_car_parking_spaces": [required_car_parking_spaces],
                "total_of_special_requests": [total_of_special_requests],
                "agent": [0],
                "company": [0]
            })
            st.dataframe(input_HBCP, use_container_width=True)

            if hasattr(model, "feature_names_in_"):
                model_columns = list(model.feature_names_in_)

                for col in model_columns:
                    if col not in input_HBCP.columns:
                        if col in HBCP.columns:
                            if pd.api.types.is_numeric_dtype(HBCP[col]):
                                input_HBCP[col] = int(HBCP[col].dropna().mode()[0])
                            else:
                                input_HBCP[col] = HBCP[col].mode()[0]
                        else:
                            input_HBCP[col] = 0

                input_HBCP = input_HBCP[model_columns]
                for col in input_HBCP.columns:
                    if col in HBCP.columns and pd.api.types.is_integer_dtype(HBCP[col]):
                        input_HBCP[col] = input_HBCP[col].astype("int64")

            try:
                int_cols = input_HBCP.select_dtypes('integer').columns
                input_HBCP[int_cols] = input_HBCP[int_cols].astype('float64')
                prediction = model.predict(input_HBCP)[0]

                if prediction == 1:
                    st.error("Booking Will Be Canceled")
                else:
                    st.success("Booking Will Not Be Canceled")

            except Exception as e:
                st.error("Prediction error")
                st.exception(e)
                st.write("Input columns:")
                st.write(input_HBCP.columns.tolist())
                st.write("Input dtypes:")
                st.write(input_HBCP.dtypes)


st.markdown("""
<style>
.floating-chat {
    position: fixed;
    right: 28px;
    bottom: 28px;
    width: 360px;
    height: 560px;
    background: #ffffff;
    border-radius: 24px;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.25);
    border: 1px solid #e5e7eb;
    z-index: 9999;
    padding: 18px;
}

.chat-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 800;
    font-size: 18px;
    color: #111827;
    margin-bottom: 14px;
}

.chat-icon {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #ef4444, #9333ea);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
}

.chat-welcome {
    background: #f3f4f6;
    padding: 14px;
    border-radius: 18px;
    font-size: 14px;
    color: #374151;
    line-height: 1.5;
    margin-bottom: 12px;
}

.chat-input-box input {
    border-radius: 18px !important;
}

div[data-testid="stChatMessage"] {
    background: #f9fafb;
    border-radius: 16px;
    padding: 8px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stChatMessage"] {
    background: #f9fafb;
    border-radius: 15px;
    padding: 8px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    st.markdown("### Hotel Assistant")

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.api_key:
        key = st.text_input(
            "Enter Groq API Key",
            type="password",
            key="groq_api_key_input"
        )

        if st.button("Connect", key="connect_groq"):
            if key:
                st.session_state.api_key = key
                st.rerun()
            else:
                st.warning("Please enter your Groq API key.")

    else:
        client = Groq(api_key=st.session_state.api_key)

        for msg in st.session_state.messages[-4:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        prompt = st.chat_input("Ask about hotel bookings...")

        if prompt:
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT}
                ] + st.session_state.messages,
                temperature=0.2
            )

            answer = response.choices[0].message.content

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.rerun()

    chat_container.float("""
        bottom: 20px;
        right: 20px;
        width: 380px;
        max-height: 650px;
        background-color: white;
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
        z-index: 9999;
    """)
# if st.button('predict Is Canceled'):
#     new_data=pd.DataFrame(columns=HBCP.columns.drop('is_canceled','reservation_status_date','reservation_status'),data=[[model,year,transmission,mileage,fueltype,tax,mpg,enginesize]])
#     st.write('predicted price:',model.predict(new_data).round(2)[0])


