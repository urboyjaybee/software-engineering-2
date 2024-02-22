# Prototype Project for Software Engineering 1
# pip install
import streamlit as st
import mysql.connector
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings

# Establish connection with the database
mydb = mysql.connector.connect(
    host="localhost", user="root", password="", database="project"
)

mycursor = mydb.cursor()
print("Connected to Database")


# Main function
def main():
    login_status = st.session_state.get("login_status", False)

    if not login_status:
        # Display login and registration forms
        st.subheader("Login")
        username_login = st.text_input("Username:")
        password_login = st.text_input("Password:", type="password")

        if st.button("Login"):
            # Check if the username and password match an existing record in the database
            sql = "SELECT * FROM users WHERE username = %s AND password = %s"
            val = (username_login, password_login)
            mycursor.execute(sql, val)
            result = mycursor.fetchone()

            if result:
                st.session_state["login_status"] = True
                st.success("Login Successful!")
                st.rerun()  # Rerun the app to refresh the page
            else:
                st.error("Login Failed. Please check your credentials.")

        # Register form
        st.subheader("Register")
        username_reg = st.text_input("Enter Username")
        password_reg = st.text_input("Enter Password", type="password")

        if st.button("Create"):
            sql = "INSERT INTO users (username, password) VALUES (%s, %s)"
            val = (username_reg, password_reg)
            mycursor.execute(sql, val)
            mydb.commit()
            st.success("Account Created Successfully!!!")

    else:
        # Add a logout button in the sidebar
        if st.sidebar.button("Logout"):
            st.session_state["login_status"] = False
            st.success("Logout Successful!")
            st.rerun()

        # Heading
        warnings.filterwarnings("ignore")
        st.title(" 📊 Customer Segmentation for Optimizing Business Outcomes")
        st.markdown(
            "<style>div.block-container{padding-top:1rem;}</style>",
            unsafe_allow_html=True,
        )

        # File uploader
        fl = st.file_uploader(
            ":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"])
        )

        # Flag to control visibility
        file_uploaded = False

        if fl is not None:
            filename = fl.name
            st.write(filename)
            df = pd.read_csv(fl, encoding="ISO-8859-1")
            file_uploaded = True
        else:
            os.chdir(r"C:\Users\vennp\OneDrive\Desktop\SOFTWARE ENGINEERING\demo2")
            df = pd.read_csv("Superstore.csv", encoding="ISO-8859-1")

        # Show visualizations only if file is uploaded

        if file_uploaded:
            col1, col2 = st.columns((2))
            df["Order Date"] = pd.to_datetime(df["Order Date"])
            startDate = pd.to_datetime(df["Order Date"]).min()
            endDate = pd.to_datetime(df["Order Date"]).max()

            with col1:
                date1 = pd.to_datetime(st.date_input("Start Date", startDate))

            with col2:
                date2 = pd.to_datetime(st.date_input("End Date", endDate))

            df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

            st.sidebar.header("Choose filter: ")

            # Pick Region
            region = st.sidebar.multiselect("Pick the Region", df["Region"].unique())
            if not region:
                df2 = df.copy()
            else:
                df2 = df[df["Region"].isin(region)]

            # Pick State
            state = st.sidebar.multiselect("Pick the State", df2["State"].unique())
            if not state:
                df3 = df2.copy()
            else:
                df3 = df2[df2["State"].isin(state)]

            # Pick City
            city = st.sidebar.multiselect("Pick the City", df3["City"].unique())

            # Filter the data based on Region, State and City
            if not region and not state and not city:
                filtered_df = df
            elif not state and not city:
                filtered_df = df[df["Region"].isin(region)]
            elif not region and not city:
                filtered_df = df[df["State"].isin(state)]
            elif state and city:
                filtered_df = df3[df["State"].isin(state) & df3["City"].isin(city)]
            elif region and city:
                filtered_df = df3[df["Region"].isin(region) & df3["City"].isin(city)]
            elif region and state:
                filtered_df = df3[df["Region"].isin(region) & df3["State"].isin(state)]
            elif city:
                filtered_df = df3[df3["City"].isin(city)]
            else:
                filtered_df = df3[
                    df3["Region"].isin(region)
                    & df3["State"].isin(state)
                    & df3["City"].isin(city)
                ]

            # Customer Segment sales proportion
            # Users can visualize the proportion of sales attributed to each segment.
            chart1, chart2 = st.columns((2))
            with chart1:
                st.subheader("Customer Segment Sales")
                fig = px.pie(
                    filtered_df, values="Sales", names="Segment", template="plotly_dark"
                )
                fig.update_traces(text=filtered_df["Segment"], textposition="inside")
                st.plotly_chart(fig, use_container_width=True)

            # Users can see how sales are distributed across different product categories, providing insights into
            # which categories contribute the most to overall sales
            with chart2:
                st.subheader("Product Category Sales")
                fig = px.pie(
                    filtered_df, values="Sales", names="Category", template="gridon"
                )
                fig.update_traces(text=filtered_df["Category"], textposition="inside")
                st.plotly_chart(fig, use_container_width=True)

            # Product Category Bar Chart
            category_df = filtered_df.groupby(by=["Category"], as_index=False)[
                "Sales"
            ].sum()

            with col1:
                st.subheader("Product Category wise Sales")
                fig = px.bar(
                    category_df,
                    x="Category",
                    y="Sales",
                    text=["${:,.2f}".format(x) for x in category_df["Sales"]],
                    template="seaborn",
                )
                st.plotly_chart(fig, use_container_width=True, height=200)

            # Region Sales Pie Chart
            with col2:
                st.subheader("Region wise Sales")
                fig = px.pie(filtered_df, values="Sales", names="Region", hole=0.5)
                fig.update_traces(text=filtered_df["Region"], textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            # Category Sales Summary
            cl1, cl2 = st.columns((2))
            with cl1:
                with st.expander("Category Sales Summary"):
                    st.write(category_df.style.background_gradient(cmap="Blues"))
                    csv = category_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Data",
                        data=csv,
                        file_name="Category.csv",
                        mime="text/csv",
                        help="Click here to download the data as a CSV file",
                    )

            # Region Sales Summary
            with cl2:
                with st.expander("Region Sales Summary"):
                    region = filtered_df.groupby(by="Region", as_index=False)[
                        "Sales"
                    ].sum()
                    st.write(region.style.background_gradient(cmap="Oranges"))
                    csv = region.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Data",
                        data=csv,
                        file_name="Region.csv",
                        mime="text/csv",
                        help="Click here to download the data as a CSV file",
                    )

            # Times Series Analysis of Sales
            filtered_df["Month/Year"] = filtered_df["Order Date"].dt.to_period("M")
            st.subheader("Time Series Analysis")

            linechart = pd.DataFrame(
                filtered_df.groupby(filtered_df["Month/Year"].dt.strftime("%Y : %b"))[
                    "Sales"
                ].sum()
            ).reset_index()
            fig2 = px.line(
                linechart,
                x="Month/Year",
                y="Sales",
                labels={"Sales": "Amount"},
                height=500,
                width=1000,
                template="gridon",
            )
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Data Summary of Time Series:"):
                st.write(linechart.T.style.background_gradient(cmap="Blues"))
                csv = linechart.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Data",
                    data=csv,
                    file_name="TimeSeries.csv",
                    mime="text/csv",
                )

            # Sub-Category Sales Summary
            import plotly.figure_factory as ff

            st.subheader("Month wise Sub-Category Sales Summary")
            with st.expander("Summary Table"):
                df_sample = df[0:4][
                    [
                        "Region",
                        "State",
                        "City",
                        "Category",
                        "Sales",
                        "Profit",
                        "Quantity",
                    ]
                ]
                fig = ff.create_table(df_sample, colorscale="Cividis")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("Month wise Sub-Category Sales Table")
                filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
                sub_category_Year = pd.pivot_table(
                    data=filtered_df,
                    values="Sales",
                    index=["Sub-Category"],
                    columns="month",
                )
                st.write(sub_category_Year.style.background_gradient(cmap="Blues"))

            # Create a scatter plot
            data1 = px.scatter(filtered_df, x="Sales", y="Profit", size="Quantity")
            data1["layout"].update(
                title="Relationship between Sales and Profits using Scatter Plot.",
                titlefont=dict(size=20),
                xaxis=dict(title="Sales", titlefont=dict(size=19)),
                yaxis=dict(title="Profit", titlefont=dict(size=19)),
            )
            st.plotly_chart(data1, use_container_width=True)

            with st.expander("View Data"):
                st.write(
                    filtered_df.iloc[:500, 1:20:2].style.background_gradient(
                        cmap="Oranges"
                    )
                )

            # Download orginal DataSet
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Data", data=csv, file_name="Data.csv", mime="text/csv"
            )

            # Display the Streamlit app
            st.title("K-means Clustering Analysis")

            # Read the dataset
            df = pd.read_csv("Superstore.csv", encoding="ISO-8859-1")

            # Select relevant features
            features = df[["Sales", "Profit", "Quantity", "Discount"]]

            # Preprocess data
            features.fillna(0, inplace=True)  # Handle missing values

            # Standardize/Normalize numerical features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Sidebar for user input
            st.sidebar.title("K-means Clustering")
            num_clusters = st.sidebar.slider("Select the number of clusters:", 2, 10, 4)

            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            df["Cluster"] = kmeans.fit_predict(scaled_features)

            # The point at which the rate of decrease in WCSS slows down and the curve starts to flatten is
            # considered the optimal number of clusters.
            # Display Elbow Plot
            st.subheader("Elbow Method for Optimal k")
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=42)
                kmeans.fit(scaled_features)
                wcss.append(kmeans.inertia_)

            elbow_data = pd.DataFrame(
                {"Number of Clusters": range(1, 11), "WCSS": wcss}
            )
            st.line_chart(elbow_data.set_index("Number of Clusters"))

            # Display Scatter Plot
            st.subheader("Scatter Plot - Sales vs. Profit")
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                df["Sales"],
                df["Profit"],
                c=df["Cluster"],
                cmap="viridis",
                alpha=0.6,
                edgecolors="w",
            )
            ax.set_xlabel("Sales")
            ax.set_ylabel("Profit")
            ax.set_title("K-means Clustering - Sales vs. Profit")

            # Add legend
            legend_labels = [f"Cluster {i}" for i in range(num_clusters)]
            legend = ax.legend(legend_labels, title="Clusters")
            legend.get_title().set_fontsize("12")

            st.pyplot(fig)

            # Display Cluster Distribution
            st.subheader("Cluster Distribution")
            cluster_counts = df["Cluster"].value_counts()
            st.bar_chart(cluster_counts)

            # Display Cluster Summaries
            st.subheader("Cluster Summaries")
            for cluster_num in range(num_clusters):
                cluster_data = df[df["Cluster"] == cluster_num]
                st.write(f"\n**Cluster {cluster_num} Summary:**")
                # Exclude "Row ID" and "Postal Code" columns
                cluster_summary = cluster_data.drop(
                    ["Row ID", "Postal Code"], axis=1
                ).describe()
                st.write(cluster_summary)

            if "my_input" not in st.session_state:
                st.session_state["my_input"] = ""

            my_input = st.session_state["my_input"]


if __name__ == "__main__":
    main()
