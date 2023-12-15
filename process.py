import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from read_data import read_soccer_data

# Load data using the data loader function
file_path = "/Users/tomnguyen/Documents/DePauw/Junior/Data Mining/final_project/archive/male_players.csv"
soccer_data = read_soccer_data(file_path)

# Load pre-trained GloVe vectors
# Replace with the path to your GloVe file
glove_path = '/Users/tomnguyen/Documents/DePauw/Junior/Data Mining/final_project/glove.6B/glove.6B.50d.txt'
word_vectors = {}

with open(glove_path, 'r', encoding='utf-8') as glove_file:
    for line in glove_file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        word_vectors[word] = vector

# Extract relevant textual features from your DataFrame
text_features = ['fifa_version', 'player_positions', 'overall', 'potential', 'age', 'height_cm', 'weight_kg', 'club_team_id', 'league_id', 'league_level', 'nationality_id', 'weak_foot', 'skill_moves', 'international_reputation', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power',
                 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed', 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk']
text_corpus = soccer_data[text_features].astype(
    str).apply(lambda row: ' '.join(row), axis=1)

# Extract textual features using GloVe embeddings
text_matrix = np.array([np.mean([word_vectors.get(word, np.zeros(50))
                                for word in text.split()], axis=0) for text in text_corpus])

# Normalize textual features
text_matrix = normalize(text_matrix, norm='l2', axis=0)


def get_player_vector(player_name):
    player_row = soccer_data[soccer_data['short_name'] == player_name]
    if not player_row.empty:
        text = ' '.join(player_row[text_features].astype(str).values[0])
        vector = np.mean([word_vectors.get(word, np.zeros(50))
                         for word in text.split()], axis=0)
        return vector
    else:
        return None


def calculate_similarities(input_vector):
    if input_vector is not None:
        similarities = cosine_similarity([input_vector], text_matrix)[0]
        return similarities
    else:
        return None


def get_top_similar_players(input_player_name, top_n=100, consider_positions=False):
    input_vector = get_player_vector(input_player_name)
    similarities = cosine_similarity([input_vector], text_matrix)[0]

    if input_vector is not None:
        # Get indices of top N most similar players
        top_indices = np.argsort(similarities)[::-1][:top_n]

        if consider_positions:
            # Filter top players based on having at least one common player_positions
            input_positions = set(
                soccer_data.loc[soccer_data['short_name'] == input_player_name, 'player_positions'].values[0].split(','))
            top_players = [(soccer_data.iloc[i]['short_name'],
                            soccer_data.iloc[i]['player_positions'],
                            similarities[i])
                           for i in top_indices
                           if any(pos in input_positions for pos in soccer_data.iloc[i]['player_positions'].split(','))]

        else:
            # Include all top players
            top_players = [(soccer_data.iloc[i]['short_name'], similarities[i])
                           for i in top_indices]

        return top_players
    else:
        return None


# Example usage for considering players with at least one common player_positions
input_player_name = 'K. De Bruyne'
top_similar_players = get_top_similar_players(
    input_player_name, consider_positions=False)
top_similar_players_positions = get_top_similar_players(
    input_player_name, consider_positions=True)


def print_top_players(players, title):
    if players:
        print(f"{title}:")
        for player in players[:5]:
            if len(player) == 2:
                print(f"{player[0]}: Similarity = {player[1]}")
            elif len(player) == 3:
                print(
                    f"{player[0]}: Position = {player[1]}, Similarity = {player[2]}")
    else:
        print(
            f"No information found for {input_player_name} or no players with at least one common position.")


print_top_players(top_similar_players,
                  f"Top 5 players with the most similar with {input_player_name}")
print_top_players(top_similar_players_positions,
                  f"Top 5 players with at least one common position as {input_player_name}")

# Fit K-Means clustering on the data
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(text_matrix)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(text_matrix)

# Add cluster labels to the soccer_data DataFrame
soccer_data['cluster_label'] = cluster_labels

# Visualize the clusters
plt.scatter(pca_result[:, 0], pca_result[:, 1],
            c=cluster_labels, cmap='viridis')
plt.title('Player Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
