"""Interactive historical flight delay risk demo."""
import calendar

import altair as alt
import pandas as pd
import streamlit as st

from model import DATA, FEATURES, load_flights, train

st.set_page_config(page_title="Flight Delay Risk Predictor", page_icon="✈️", layout="wide")
st.caption("PORTFOLIO DEMO  /  PREDICTIVE ANALYTICS")
st.title("Flight Delay Risk Predictor")
st.markdown("Explore how the scheduled route, airline, day and time relate to arrival delays.")
st.info("Historical study of Washington, DC area departures to New York in January 2004. "
        "Use it to explore the model and dataset.")

if not DATA.is_file():
    st.error("Download FlightDelays.csv using the data link in the README and place it in data/.")
    st.stop()


@st.cache_resource
def fit_project():
    return train(load_flights())


model, history, summary = fit_project()
explore, methods = st.tabs(["Explore a flight", "Model evaluation"])

with explore:
    first, second, third = st.columns(3)
    origin = first.selectbox("Origin airport", sorted(history["origin"].unique()))
    origin_rows = history[history["origin"].eq(origin)]
    destination = second.selectbox("Destination airport", sorted(origin_rows["destination"].unique()))
    route = origin_rows[origin_rows["destination"].eq(destination)]
    carrier = third.selectbox("Carrier code", sorted(route["carrier"].unique()))
    schedule = route[route["carrier"].eq(carrier)]
    weekday_col, hour_col = st.columns(2)
    weekday = weekday_col.selectbox("Day of week", sorted(schedule["weekday"].unique()),
                                   format_func=lambda day: calendar.day_name[day])
    matching_day = schedule[schedule["weekday"].eq(weekday)]
    hours = sorted(matching_day["hour"].unique())
    hour = hour_col.selectbox("Scheduled departure hour (local)", hours,
                             format_func=lambda value: f"{value:02d}:00")
    scenarios = pd.DataFrame({
        "origin": origin, "destination": destination, "carrier": carrier,
        "weekday": weekday, "hour": hours,
    })[FEATURES]
    scenarios["risk"] = model.predict_proba(scenarios)[:, 1]
    selected = scenarios.loc[scenarios["hour"].eq(hour), "risk"].iloc[0]
    risk, route_rate, records = st.columns(3)
    risk.metric("Model-estimated delay risk", f"{selected:.1%}")
    route_rate.metric("Historical route delay rate", f"{route['delayed'].mean():.1%}")
    records.metric("Historical flights on this route", f"{len(route):,}")
    st.caption("The route benchmark includes all carriers and departure times through January 24.")
    st.subheader("How the estimate changes by departure hour")
    st.write("Same route, carrier and weekday. The chart uses hours observed for that combination.")
    scenarios["Local hour"] = scenarios["hour"].map(lambda value: f"{value:02d}:00")
    chart = alt.Chart(scenarios).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Local hour:O", sort=None),
        y=alt.Y("risk:Q", title="Estimated delay probability",
                axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(alt.datum.hour == int(hour), alt.value("#2563EB"), alt.value("#94A3B8")),
        tooltip=["Local hour:O", alt.Tooltip("risk:Q", format=".1%", title="Delay probability")],
    )
    st.altair_chart(chart, width="stretch")

with methods:
    st.subheader("Chronological holdout")
    st.write("Train on January 1–18, choose the classifier on January 19–24 by lowest Brier score, "
             "then refit through January 24 and evaluate on January 25–31.")
    st.markdown(f"**Selected classifier** · {summary['selected']}")
    st.dataframe(pd.DataFrame(summary["test"]).T.rename(columns={
        "roc_auc": "ROC AUC", "average_precision": "Average precision", "brier": "Brier score",
    }).round(3), width="stretch")
    st.caption(f"Test flights · {summary['rows']['test']:,}   |   "
               f"Observed delay rate · {summary['test_delay_rate']:.1%}")
    st.subheader("Which inputs helped on the holdout?")
    st.write("Average increase in Brier loss when each feature was shuffled. "
             "Positive values indicate that shuffling worsened probability accuracy.")
    importance = pd.Series(summary["importance"], name="Increase in Brier loss").sort_values()
    st.bar_chart(importance, horizontal=True)
