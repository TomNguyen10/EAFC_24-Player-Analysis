import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from read_data import read_soccer_data


def load_glove_vectors(glove_path):
    word_vectors = {}
    with open(glove_path, 'r', encoding='utf-8') as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors


def process_text_features(soccer_data, text_features, word_vectors):
    text_corpus = soccer_data[text_features].astype(
        str).apply(lambda row: ' '.join(row), axis=1)
    text_matrix = np.array([np.mean([word_vectors.get(word, np.zeros(50))
                                     for word in text.split()], axis=0) for text in text_corpus])
    text_matrix = normalize(text_matrix, norm='l2', axis=0)
    return text_matrix


def get_top_similar_players(input_player_name, text_matrix, soccer_data, word_vectors, top_n=100, consider_positions=False):
    input_vector = get_player_vector(
        input_player_name, soccer_data, word_vectors)
    similarities = cosine_similarity([input_vector], text_matrix)[0]

    if input_vector is not None:
        top_indices = np.argsort(similarities)[::-1][:top_n]

        if consider_positions:
            input_positions = set(
                soccer_data.loc[soccer_data['short_name'] == input_player_name, 'player_positions'].values[0].split(','))
            top_players = [(soccer_data.iloc[i]['short_name'],
                            soccer_data.iloc[i]['player_positions'],
                            similarities[i])
                           for i in top_indices
                           if any(pos in input_positions for pos in soccer_data.iloc[i]['player_positions'].split(','))]

        else:
            top_players = [(soccer_data.iloc[i]['short_name'], similarities[i])
                           for i in top_indices]

        return top_players
    else:
        return None


def get_player_vector(player_name, soccer_data, word_vectors):
    player_row = soccer_data[soccer_data['short_name'] == player_name]
    if not player_row.empty:
        text = ' '.join(player_row[soccer_data.columns].astype(str).values[0])
        vector = np.mean([word_vectors.get(word, np.zeros(50))
                         for word in text.split()], axis=0)
        return vector
    else:
        return None


def cluster_players(text_matrix, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(text_matrix)
    return cluster_assignments


def visualize_clusters(text_matrix, cluster_assignments, soccer_data, num_clusters, num_players):
    pca_2d = PCA(n_components=2)
    player_embeddings_2d = pca_2d.fit_transform(text_matrix)

    label_features = ['overall', 'potential']

    plt.figure(figsize=(10, 8))
    for cluster_idx in range(num_clusters):
        cluster_points = player_embeddings_2d[cluster_assignments == cluster_idx]
        labels = soccer_data.iloc[:num_players][label_features].astype(str).apply(
            lambda row: ', '.join(row), axis=1).values[cluster_assignments == cluster_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:,
                    1], label=f'Cluster {cluster_idx + 1}')

        for label, x, y in zip(labels, cluster_points[:, 0], cluster_points[:, 1]):
            plt.annotate(label, (x, y), textcoords="offset points",
                         xytext=(0, 5), ha='center')

    plt.title('Player Clusters based on Textual Features (2D PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()


def print_top_players(players, title, input_player_name):
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


def calculate_player_potential(short_name, soccer_df):
    base_potential = 0.2
    age_modifier = 0.05
    league_level_modifier = 0.02
    skill_moves_modifier = 0.01
    international_rep_modifier = 0.005
    stats_modifier = 0.002

    selected_player = soccer_df[soccer_df['short_name'] == short_name].iloc[0]

    age = selected_player["age"]
    league_level = selected_player["league_level"]
    skill_moves = selected_player["skill_moves"]
    international_reputation = selected_player["international_reputation"]
    stats_sum = sum(selected_player[stat] for stat in (
        "pace", "shooting", "passing", "dribbling", "defending", "physic"))

    potential = base_potential + (
        age_modifier * (20 - age) +
        league_level_modifier * league_level +
        skill_moves_modifier * skill_moves +
        international_rep_modifier * international_reputation +
        stats_modifier * stats_sum)

    potential = max(1, min(potential, 1.15))
    potential = potential * selected_player['overall']

    return int(potential), selected_player['potential']


def train_linear_regression_model(features, labels):
    model_default = LinearRegression()
    model_default.fit(features, labels)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model_scaled = LinearRegression()
    model_scaled.fit(features_scaled, labels)

    models = {'default': model_default, 'scaled': model_scaled}

    return models


def evaluate_linear_regression_model(models, features, labels):
    predictions = {}
    mses = {}

    for config, model in models.items():
        predictions[config] = cross_val_predict(model, features, labels, cv=5)
        mses[config] = np.abs(np.mean(cross_val_score(
            model, features, labels, scoring='neg_mean_squared_error', cv=5)))

        print(f'Mean Squared Error ({config}): {mses[config]}')

    return mses, predictions


def plot_predictions_vs_real(labels, predictions, title):
    plt.scatter(labels, predictions)
    plt.xlabel('Real Potential')
    plt.ylabel('Predicted Potential')
    plt.title(title)
    plt.show()


def main():
    file_path = "archive/male_players.csv"
    soccer_data = read_soccer_data(file_path)

    glove_path = 'glove.6B/glove.6B.50d.txt'
    word_vectors = load_glove_vectors(glove_path)

    text_features = [col for col in soccer_data.columns]
    text_matrix = process_text_features(
        soccer_data, text_features, word_vectors)

    input_player_name = 'S. Amrabat'
    top_similar_players = get_top_similar_players(
        input_player_name, text_matrix, soccer_data, word_vectors, consider_positions=False)
    top_similar_players_positions = get_top_similar_players(
        input_player_name, text_matrix, soccer_data, word_vectors, consider_positions=True)

    num_players = 1000
    num_clusters = 5
    cluster_assignments = cluster_players(
        text_matrix[:num_players], num_clusters)

    print_top_players(top_similar_players,
                      f"Top 5 players with the most similarity with {input_player_name}", input_player_name)
    print_top_players(top_similar_players_positions,
                      f"Top 5 players with at least one common position as {input_player_name}", input_player_name)

    visualize_clusters(text_matrix[:num_players],
                       cluster_assignments, soccer_data, num_clusters, num_players)

    predict, real_val = calculate_player_potential(
        input_player_name, soccer_data)

    print(
        f"Player: {input_player_name}, Predicted Potential: {predict}, Real Potential: {real_val}")

    features = text_matrix[:num_players]
    _, real_potential = zip(*[calculate_player_potential(player['short_name'], soccer_data)
                              for _, player in soccer_data.head(num_players).iterrows()])

    trained_models = train_linear_regression_model(features, real_potential)

    evaluation_mses, predictions = evaluate_linear_regression_model(
        trained_models, features, real_potential)

    for config, mse in evaluation_mses.items():
        print(f'Mean Squared Error ({config}): {mse}')

    for config, prediction in predictions.items():
        plot_predictions_vs_real(
            real_potential, prediction, f'Linear Regression ({config})')


if __name__ == "__main__":
    main()
