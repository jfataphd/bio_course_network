import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import networkx as nx
import re

# Load the CSV file
bio_data = pd.read_csv('Bio_Req.csv')

# Create the graph
G = nx.Graph()

# List of special nodes
# special_nodes = ["M4 Flag", "C6 Flag", "Res 51", "BD Flag", "Department Permission", "Graduate Only"]

# Add nodes with course names and store 'Description_Short' data
bio_data['Course Name'] = bio_data['Subject'] + bio_data['Catalog'].astype(str)
description_short_dict = {}
for _, row in bio_data.iterrows():
    course_name = row['Course Name']
    G.add_node(course_name)
    description_short_dict[course_name] = row['Description_Short']

# Add special nodes and their descriptions
# special_nodes = ["M4 Flag", "C6 Flag", "Res 51", "BD Flag", "Department Permission", "Graduate Only"]
# special_nodes_descriptions = {
#     "M4 Flag": "M4 Flag",
#     "C6 Flag": "C6 Flag",
#     "Res 51": "Res 51",
#     "BD Flag": "BD Flag",
#     "Department Permission": "Dept Perm",
#     "Graduate Only": "GRADUATE STUDENTS ONLY"
#     # Add descriptions for the rest of the special nodes
# }

# for node in special_nodes:
#     G.add_node(node)
#     description_short_dict[node] = special_nodes_descriptions[node]

# Function to extract level from course name
def get_course_level(course_name):
    match = re.search(r'\d+', course_name)
    if match:
        return int(match.group()) // 100 * 100
    return None

# Function to get node color based on course level
def get_node_color(course_name):
    level = get_course_level(course_name)
    if level is not None:
        if 100 <= level < 500:
            pastel_colors = {100: 'lightblue', 200: 'lightgreen', 300: 'mediumpurple', 400: 'lightpink'}
            return pastel_colors.get(level, 'grey')  # Default to grey if level not in pastel_colors
        else:
            return 'black'  # Black for 500 and above
    return 'grey'  # Default color


# Function to parse and classify requirements, including special cases
def classify_requirements(description, target_course, subject_code):
    edges = []

    # Handle course pairs like '170/171'
    slash_pairs = re.findall(r'(\d{3})\/(\d{3})', description)
    for pair in slash_pairs:
        for course_number in pair:
            course_code = f"{subject_code}{course_number}"
            edges.append((course_code, target_course))

    # Handle ampersand pairs like '106&107'
    ampersand_pairs = re.findall(r'(\d{3})&(\d{3})', description)
    for pair in ampersand_pairs:
        for course_number in pair:
            course_code = f"{subject_code}{course_number}"
            edges.append((course_code, target_course))

    # Standard course prerequisites and corequisites
    standard_courses = [course.replace(" ", "") for course in re.findall(r'[A-Z]{3}\s?\d{3}', description)]
    edges.extend([(req, target_course) for req in standard_courses if req in bio_data['Course Name'].values])

    # Special cases
    # if "M4" in description:
    #     edges.append(("M4 Flag", target_course))
    # if "C6" in description:
    #     edges.append(("C6 Flag", target_course))
    # if "Res" in description:
    #     edges.append(("Res 51", target_course))
    # if "BD" in description:
    #     edges.append(("BD Flag", target_course))
    # if "Dept" in description:
    #     edges.append(("Department Permission", target_course))
    # if "GRADUATE" in description:
    #     edges.append(("Graduate Only", target_course))

    # Subject-specific cases with slashes and ampersands, potentially multiple pairs
    complex_patterns = re.findall(r'([A-Z]{3})\s?(\d{3})[&/](\d{3})', description)
    for subject, num1, num2 in complex_patterns:
        edges.append((f"{subject}{num1}", target_course))
        edges.append((f"{subject}{num2}", target_course))

    return edges

# Parse each row's 'Description_Short' to add edges
for _, row in bio_data.iterrows():
    subject_code = row['Subject']  # Extract the subject code from the row
    G.add_edges_from(classify_requirements(row['Description_Short'], row['Course Name'], subject_code))


# Function to determine node size based on its degree
def get_node_size(node, base_size=8):
    degree = G.degree(node)
    # Use a smaller multiplier to reduce the size increment
    return base_size + max(degree * 1.5, 2)

# Streamlit app
st.title('Course Network Graph')

# Input for search functionality
user_input = st.text_input("Enter a course name (e.g., BIO170):").strip().upper()

# Only proceed if the user input is valid
if user_input:
    # Check if the user's input is in the graph
    if user_input in G:
        # Generate positions for each node using a 3D layout algorithm
        pos = nx.spring_layout(G, dim=3, k=0.6, iterations=100, seed=60)

        # When unpacking positions for max_y and min_y calculations
        max_y = max([y for _, y, _ in pos.values()])  # Correctly handle 3D coordinates
        min_y = min([y for _, y, _ in pos.values()])

        # When unpacking positions in other parts of the script (like edge trace creation)
        for edge in G.edges():
            x0, y0, _ = pos[edge[0]]  # Correctly handle 3D coordinates
            x1, y1, _ = pos[edge[1]]

        for node in G.nodes():
            level = get_course_level(node)
            if level is not None:
                # Normalize the level to be between 0 and 1 and then scale it to the y-coordinate range
                normalized_level = (level - 100) / 300  # Assuming levels are between 100 and 400
                x, y, z = pos[node]  # Unpack all three coordinates
                pos[node] = (x, normalized_level * (max_y - min_y) + min_y, z)  # Adjust y keeping x and z same

        # Create edges for 3D plot
        edge_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )

        # Add edges to edge trace for 3D plot
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
            edge_trace['z'] += tuple([z0, z1, None])

        # Prepare lists for node attributes for 3D plot
        node_positions_x, node_positions_y, node_positions_z = [], [], []
        node_sizes, node_colors, node_texts = [], [], []

        # Iterate over nodes for 3D plot
        for node in G.nodes():
            if G.degree(node) > 0:
                x, y, z = pos[node]
                node_positions_x.append(x)
                node_positions_y.append(y)
                node_positions_z.append(z)
                node_sizes.append(get_node_size(node))
                node_colors.append(get_node_color(node) if node != user_input else 'red')
                hover_text = f"<b>{node}</b><br>{description_short_dict.get(node, '')}"
                if node == user_input:
                    hover_text = f"<b>{node} (Searched)</b><br>{description_short_dict.get(node, '')}"
                node_texts.append(hover_text)

        # Create nodes with updated attributes for 3D plot
        node_trace = go.Scatter3d(
            x=node_positions_x,
            y=node_positions_y,
            z=node_positions_z,
            text=node_texts,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2)
            ),
            showlegend=False
        )

        # Create the 3D figure
        fig = go.Figure(data=[edge_trace, node_trace])

        # Define the color scheme for the legend including the searched node
        color_scheme = {
            '100 Level': 'lightblue',
            '200 Level': 'lightgreen',
            '300 Level': 'mediumpurple',
            '400 Level': 'lightpink',
            # '500+ Level': 'black',
            # 'Special Nodes': 'yellow',
            'Searched Course': 'red'
        }

        # Add dummy traces for the legend
        for level, color in color_scheme.items():
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10 if color != 'red' else 15, color=color),
                name=f"{level}{' - ' + user_input if color == 'red' else ''}"
            ))

        # Update layout to include custom legend and adjust other settings as needed
        layout_config = dict(
            title='Network Graph made with Python',
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',  # Set hovermode to closest to enable hover information for nodes
            margin=dict(b=20, l=5, r=5, t=40),
            autosize=True,  # Enable responsive layout
            height=800,  # You can adjust this value as needed

            # Configure the 3D scene
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showspikes=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showspikes=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showspikes=False),
            )
        )

        print("Layout Configuration:", layout_config)  # Print the layout configuration
        fig.update_layout(layout_config)

        # Display the 3D figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
