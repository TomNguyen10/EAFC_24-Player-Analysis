# player_similarity_analysis.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from read_data import read_soccer_data  # Import the data loading function

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
text_features = ['fifa_version', 'overall', 'potential', 'age', 'height_cm', 'weight_kg', 'club_team_id', 'league_id', 'league_level', 'nationality_id', 'weak_foot', 'skill_moves', 'international_reputation', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power',
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


def get_top_similar_players(input_player_name, top_n=10):
    input_vector = get_player_vector(input_player_name)
    if input_vector is not None:
        similarities = cosine_similarity([input_vector], text_matrix)[0]
        # Get indices of top N most similar players
        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_players = [(soccer_data.iloc[i]['short_name'], similarities[i])
                       for i in top_indices]
        return top_players
    else:
        return None


# Example usage
input_player_name = 'J. Koundé'
top_similar_players = get_top_similar_players(input_player_name)

if top_similar_players:
    print(f"Top 10 players similar to {input_player_name}:")
    for player, similarity in top_similar_players:
        print(f"{player}: Similarity = {similarity}")
else:
    print(f"No information found for {input_player_name}.")
