import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import altair as alt
import base64
from statistics import mean
import copy
import scipy

# Streamlit configuration
st.set_page_config(
    page_title="Network Analysis App",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_nodes(n, competencies):
    nodes = []
    for i in range(n):
        node = {competency: random.randint(0, 10) for competency in competencies}
        nodes.append(node)
    return nodes

def create_similarity_df(nodes, competencies):
    num_nodes = len(nodes)
    similarity_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            a = [nodes[i][competency] for competency in competencies]
            b = [nodes[j][competency] for competency in competencies]
            similarity_matrix[i][j] = cosine_similarity(a, b)

    # Create a DataFrame
    df = pd.DataFrame(similarity_matrix, columns=[f"Node {i}" for i in range(num_nodes)],
                      index=[f"Node {i}" for i in range(num_nodes)])

    # Melt the DataFrame
    df_melt = df.reset_index().melt(id_vars="index", var_name="column", value_name="similarity")
    df_melt.columns = ["row", "column", "similarity"]
    return df_melt

def calculate_potential_growth(nodes, best_fit_candidate):
    team_competencies = pd.DataFrame(nodes).mean()
    candidate_competencies = pd.DataFrame(best_fit_candidate, index=[0])

    growth = candidate_competencies.subtract(team_competencies, axis=1).mean().abs()

    return growth

# Sidebar for inputs
st.sidebar.title("Input Parameters")

# Step 0: Define competencies
with st.sidebar.expander("Specify Competencies"):
    num_competencies = int(st.number_input("Number of competencies", min_value=1, max_value=10, value=4, step=1, key="num_competencies"))
    competencies = []
    for i in range(num_competencies):
        competency = st.text_input(f"Competency {i + 1}", value=f"Competency {i + 1}", key=f"competency_{i + 1}")
        competencies.append(competency)
        
# Step 1: Input number of team members and new candidates
with st.sidebar.expander("Provide Team and Candidate Counts"):
    num_team_members = int(st.number_input("Number of team members", min_value=1, max_value=20, value=10, step=1, key="num_team_members"))
    num_candidates = int(st.number_input("Number of new candidates", min_value=1, max_value=20, value=4, step=1, key="num_candidates"))

# Step 2: Input data for team members
with st.sidebar.expander("Input Team Member Data"):
    data_input_option_team = st.selectbox("Choose input method for team members", ["Manually input data", "Upload a CSV or Excel file"])

    if data_input_option_team == "Manually input data":
        nodes = []
        for i in range(num_team_members):
            node = {}
            st.markdown(f"**Team Member {i + 1}**")
            for competency in competencies:
                node[competency] = st.number_input(f"Competency {competency}", min_value=0, max_value=10, value=0, step=1, key=f"competency_{competency}_{i}")
            nodes.append(node)

    else:  # "Upload a CSV or Excel file"
        team_file = st.file_uploader("Upload a file for team members data", type=['csv', 'xlsx'])
        if team_file is not None:
            try:
                if '.csv' in team_file.name:
                    nodes = pd.read_csv(team_file)
                else:
                    nodes = pd.read_excel(team_file)
            except Exception as e:
                st.write("There was an error loading the file: ", e)

# Step 3: Input data for new candidates
with st.sidebar.expander("Input Candidate Data"):
    data_input_option_candidates = st.selectbox("Choose input method for new candidates", ["Manually input data", "Upload a CSV or Excel file"])

    if data_input_option_candidates == "Manually input data":
        new_nodes = []
        for i in range(num_candidates):
            node = {}
            st.markdown(f"**Candidate {i + 1}**")
            for competency in competencies:
                node[competency] = st.number_input(f"Candidate's Competency {competency}", min_value=0, max_value=10, value=0, step=1, key=f"cand_competency_{competency}_{i}")
            new_nodes.append(node)

    else:  # "Upload a CSV or Excel file"
        candidate_file = st.file_uploader("Upload a file for new candidates data", type=['csv', 'xlsx'])
        if candidate_file is not None:
            try:
                if '.csv' in candidate_file.name:
                    new_nodes = pd.read_csv(candidate_file)
                else:
                    new_nodes = pd.read_excel(candidate_file)
            except Exception as e:
                st.write("There was an error loading the file: ", e)

# Step 4: Input threshold value
with st.sidebar.expander("Input threshold for establishing connections"):
    threshold = st.number_input("Set Threshold", min_value=0.0, max_value=1.0, value=0.90, step=0.01, key="threshold")

with st.sidebar.expander("Compare Candidates"):
    candidates_to_compare = st.multiselect("Select candidates to compare", [i for i in range(num_candidates)], key="candidates_to_compare")

# Main app
st.title("Competency-Based Team Fit Analysis")

st.write("This tool helps you identify the best fit candidate for your team based on competencies.")

if st.sidebar.button("Perform Network Analysis to Find the Best Fit"):

    with st.spinner('Processing...'):
        nodes = generate_nodes(num_team_members, competencies)
        new_nodes = generate_nodes(num_candidates, competencies)

        # Show team members and candidates data in tables
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Team Members Data:")
            st.dataframe(pd.DataFrame(nodes))  # Changed to st.dataframe
        with col2:
            st.subheader("Candidates Data:")
            st.dataframe(pd.DataFrame(new_nodes))  # Changed to st.dataframe

        nodes = generate_nodes(num_team_members, competencies)
        new_nodes = generate_nodes(num_candidates, competencies)


    with st.expander("Visual Representation of Competency Scores", expanded=True):
        # Converting data to a pandas DataFrame
        df = pd.DataFrame(nodes + new_nodes)
        df['Type'] = ['Team Member' if i < num_team_members else 'Candidate' for i in range(num_team_members + num_candidates)]

        # Melting data to long format for Altair
        df = df.melt(id_vars='Type', var_name='Competency', value_name='Score')

        # Creating Altair charts
        bar_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Competency:N', title='Competency'),
            y=alt.Y('Score:Q', title='Score'),
            color='Type:N',
            tooltip=['Type:N', 'Competency:N', 'Score:Q']
        ).properties(
            width=600,
            height=500
        ).interactive()

        histogram = alt.Chart(df).mark_bar().encode(
            alt.X("Score:Q", bin=True),
            alt.Y('count()'),
            color='Type:N',
            tooltip=['count()']
        ).properties(
            width=600,
            height=500
        ).interactive()

        boxplot = alt.Chart(df).mark_boxplot().encode(
            x='Competency:N',
            y='Score:Q',
            color='Type:N',
            tooltip=['Type:N', 'Competency:N', 'Score:Q']
        ).properties(
            width=600,
            height=500
        ).interactive()

        heatmap = alt.Chart(df).mark_rect().encode(
            x='Competency:N',
            y='Type:N',
            color=alt.Color('mean(Score):Q', legend=alt.Legend(title="Mean Score")),
            tooltip=['Type:N', 'Competency:N', 'mean(Score):Q']
        ).properties(
            width=600,
            height=500
        ).interactive()

        # Displaying charts in a 2 column layout
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(bar_chart, use_container_width=True)
            st.altair_chart(histogram, use_container_width=True)
        with col2:
            st.altair_chart(boxplot, use_container_width=True)
            st.altair_chart(heatmap, use_container_width=True)

    with st.expander("Compare Selected Candidates", expanded=True):
        compare_df = pd.DataFrame(new_nodes[i] for i in candidates_to_compare)
        compare_df.index = ["Candidate " + str(i) for i in candidates_to_compare]

        # Show candidates comparison data in a table
        st.subheader("Candidates Comparison:")
        st.dataframe(compare_df)

        # Show a graph comparing the candidates
        compare_df = compare_df.reset_index().melt('index')

        bar_chart_1 = alt.Chart(compare_df).mark_bar().encode(
            x='index:N',
            y='value:Q',
            color='index:N',
            column='variable:N',
            tooltip=['index:N', 'variable:N', 'value:Q']
        ).interactive().properties(
            width=50,
            height=150,
            )

        line_chart_1 = alt.Chart(compare_df).mark_line().encode(
            x='variable:N',
            y='value:Q',
            color='index:N',
            tooltip=['index:N', 'variable:N', 'value:Q']
        ).interactive()

        area_chart_1 = alt.Chart(compare_df).mark_area(opacity=0.4).encode(
            x='variable:N',
            y=alt.Y('value:Q', stack=None),
            color='index:N',
            tooltip=['index:N', 'variable:N', 'value:Q']
        ).interactive()

        scatter_plot_1 = alt.Chart(compare_df).mark_circle().encode(
            x='variable:N',
            y='value:Q',
            color='index:N',
            size='value:Q',
            tooltip=['index:N', 'variable:N', 'value:Q']
        ).interactive()

        # 2-column layout
        col3, col4 = st.columns(2)

        # First column
        with col3:
            st.altair_chart(bar_chart_1)
            st.altair_chart(area_chart_1, use_container_width=True)

        # Second column
        with col4:
            st.altair_chart(line_chart_1, use_container_width=True)
            st.altair_chart(scatter_plot_1, use_container_width=True)


    # Step 2: Display graph before and after introducing new nodes
    with st.expander("Network Analysis Visualization", expanded=True):

        similarity_matrix = np.zeros((num_team_members, num_team_members))
        for i in range(num_team_members):
            for j in range(num_team_members):
                a = [nodes[i][competency] for competency in competencies]
                b = [nodes[j][competency] for competency in competencies]
                similarity_matrix[i][j] = cosine_similarity(a, b)

        # Creating a networkx graph and adding edges based on similarity matrix
        G = nx.Graph()
        for i in range(num_candidates):
            for j in range(len(nodes)):
                a = [nodes[i][competency] for competency in competencies]
                b = [nodes[j][competency] for competency in competencies]
                similarity = cosine_similarity(a, b)
                if i != j and similarity > threshold:
                    G.add_edge(i, j, weight=similarity)

        col5, col6 = st.columns(2)

        # First column
        with col5:
            # Plotting the graph before introducing new nodes
            nx.draw(G, with_labels=True)
            plt.title("Network before introducing new nodes")
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.clf()

        # Second column
        with col6:
            # Add new candidates to the graph
            for i in range(num_candidates):
                node_attributes = {competency: new_nodes[i][competency] for competency in competencies}
                G.add_node(num_team_members + i, **node_attributes)

            # Connect the new nodes to the original network
            for i in range(num_candidates):
                for j in range(len(nodes)):
                    a = [new_nodes[i][competency] for competency in competencies]
                    b = [nodes[j][competency] for competency in competencies]
                    similarity = cosine_similarity(a, b)
                    if i != j and similarity > threshold:
                        G.add_edge(num_team_members + i, j, weight=similarity)

            # Plotting the graph after connecting the new nodes
            colors = ["blue" if i >= num_team_members else "red" for i in G.nodes()]
            nx.draw(G, with_labels=True, node_color=colors)
            plt.title("Network after connecting the new nodes")
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.clf()

    with st.expander("Network Analysis Results", expanded=True):

        eigen_centrality = nx.eigenvector_centrality_numpy(G)

        # Finding the new node with the highest eigen vector centrality
        max_centrality = 0
        max_node = None
        for node in G.nodes():
            if node >= num_team_members:
                centrality = eigen_centrality[node]
                if centrality > max_centrality:
                    max_centrality = centrality
                    max_node = node

        # Calculating the quality of connections to the old nodes in the network
        total_similarity = 0
        num_connections = 0
        for node in G.nodes():
            if node < num_team_members:
                similarity = G.get_edge_data(max_node, node, default={'weight': 0})['weight']
                if similarity > 0:
                    total_similarity += similarity
                    num_connections += 1

        avg_similarity = total_similarity / num_connections if num_connections else 0
        confidence = avg_similarity * 100

        # Creating a DataFrame for the analysis results
        analysis_df = pd.DataFrame({
            'Best Fit': [max_node],
            'Confidence of the suggestion': [confidence]
        })

        col7, col8 = st.columns(2)

        with col7:
            # Displaying the analysis results
            st.write(analysis_df)

        with col8:
            centrality = nx.eigenvector_centrality(G)

            # Finding the new node with highest eigenvector centrality
            new_node_ids = [i for i in range(num_team_members, num_team_members + num_candidates)]
            new_node_centrality = [centrality[i] for i in new_node_ids]
            best_fitted_node = new_node_ids[np.argmax(new_node_centrality)]


            # Calculating the quality of connections to old nodes
            quality = {}
            for j in range(num_team_members):
                a = [new_nodes[best_fitted_node - num_team_members][competency] for competency in competencies]
                b = [nodes[j][competency] for competency in competencies]
                similarity = cosine_similarity(a, b)
                if similarity > threshold:
                    quality[j] = similarity * 100

            # Creating a DataFrame for the quality of connections
            quality_df = pd.DataFrame({
                'Node': list(quality.keys()),
                'Quality of connection (%)': list(quality.values())
            })

            # Displaying the quality of connections
            st.write(quality_df)        

    with st.expander("Network Visualization with Best Fit Candidate", expanded=True):
        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        # Plot in the first column: Network with all candidates and best candidate highlighted
        with col1:
            best_fit_G = G.copy()
            node_colors = ["blue" if i == max_node else "red" if i < num_team_members else "green" for i in best_fit_G.nodes()]
            nx.draw(best_fit_G, with_labels=True, node_color=node_colors)
            plt.title("Network with All Candidates (Best Fit Highlighted)")
            st.pyplot(plt.gcf())
            plt.clf()

        # Plot in the second column: Network with only best fit candidate
        with col2:
            best_fit_G_only = G.copy()
            nodes_to_remove = [i for i in range(num_team_members, num_team_members + num_candidates) if i != max_node]
            best_fit_G_only.remove_nodes_from(nodes_to_remove)
            node_colors = ["blue" if i == max_node else "red" for i in best_fit_G_only.nodes()]
            nx.draw(best_fit_G_only, with_labels=True, node_color=node_colors)
            plt.title("Network with Only Best Fit Candidate")
            st.pyplot(plt.gcf())
            plt.clf()

    with st.expander("Potential Team Growth with Best Fit Candidate", expanded=True):
        growth = calculate_potential_growth(nodes, new_nodes[max_node - num_team_members])

        # Reset index and convert column names to string
        growth = growth.reset_index()
        growth.columns = growth.columns.astype(str)

        # Create two columns
        col1, col2 = st.columns(2)

        # Use the first column to display the results
        with col1:
            st.write(growth)

        # Use the second column to display the chart
        with col2:
            growth_chart = alt.Chart(growth).mark_bar().encode(
                x='index:O',
                y='0',
            ).properties(title='Potential Growth in Each Competency')
            st.altair_chart(growth_chart, use_container_width=True)
