from datetime import date
import os
import string
from collections import Counter

# Data cleaning and modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

DIRNAME = os.path.dirname(os.path.abspath("__file__"))
PATH_TO_CSV = os.path.join(DIRNAME, 'daylio_export.csv')

ORDERED_MOODS = ['awful', 'bad', 'meh', 'good',
                 'rad']  # Daylio moods ordered from worst to best
SCALE = 10  # The scale for your mood score. This means scores will be between 0 and range(SCALE)


class Journal:
    def __init__(self,
                 path_to_csv=PATH_TO_CSV,
                 ordered_moods=ORDERED_MOODS,
                 scale=SCALE):

        self.ordered_moods = ordered_moods
        self.scale = scale
        self.raw_data = pd.read_csv(path_to_csv, index_col=False)
        self.data, self.activities = self.clean_df()
        self.dot_plot = self.gen_dot_plot()

    def clean_df(self):

        df = self.raw_data
        ordered_moods = self.ordered_moods
        scale = self.scale

        df.full_date = pd.to_datetime(df.full_date)
        df_with_one_hot_encoding, all_activities = self.convert_activities_to_categorical(
            df)
        df_with_mood_scores = self.mood_to_score(df_with_one_hot_encoding)

        return (df_with_mood_scores, all_activities)

    def gen_hist(self):

        moods = self.data[['mood']]

        sns.set()
        sns.countplot(moods['mood'], order=ORDERED_MOODS, color='gray')
        plt.title("mood distribution")

        return (plt)

    def gen_dot_plot(self, matrix_length=30):
        '''
        Generates a matrix dot plot visualization representing all my moods over time.

        Input: Journal DataFrame, list of ordered moods, int of matrix length
        Output: PyPlot figure
        '''
        df = self.data

        mood_categories = {}
        category = 0
        for mood in self.ordered_moods:
            mood_categories[mood] = category
            category += 1

        mood_list = []
        for mood in list(df.mood):
            mood_list.append(mood_categories[mood])

        next_round_value = matrix_length * (len(mood_list) // matrix_length +
                                            1)
        matrix_height = int(next_round_value / matrix_length)
        mood_list_with_remainder = mood_list + [category] * (next_round_value -
                                                             len(mood_list))

        all_mood_array = []

        for x in range(matrix_height):
            new_row = mood_list_with_remainder[(x *
                                                matrix_length):((x + 1) *
                                                                matrix_length)]
            all_mood_array.append(new_row)

        all_mood_array = np.array(all_mood_array)

        plt.figure(figsize=(5, 5))
        colormap = colors.ListedColormap([
            "darkgreen", "forestgreen", "limegreen", "yellowgreen",
            "greenyellow", "darkgrey"
        ])
        plt.imshow(all_mood_array, cmap=colormap)
        plt.axis('off')

        return (plt)

    def convert_activities_to_categorical(self, df):
        '''
        Input:
            DataFrame of journal entries
        Output:
            DataFrame of journal entries with a categorical variable per Daylio activity
        '''

        all_activities = []

        for index, row in df.iterrows():
            if type(row['activities']) == str:
                activities_list = row['activities'].split(" | ")
                for activity in activities_list:
                    if activity not in all_activities:
                        all_activities.append(activity)

        categorical_activity_matrix = []

        for index, row in df.iterrows():
            activity_list_binary = []
            if type(row['activities']) != str:
                activity_list_binary = [False] * len(all_activities)
            else:
                for activity in all_activities:
                    if activity in row['activities']:
                        activity_list_binary.append(True)
                    else:
                        activity_list_binary.append(False)
            categorical_activity_matrix.append(activity_list_binary)

        categorical_df = pd.DataFrame(categorical_activity_matrix,
                                      columns=all_activities)

        full_df = pd.concat([df, categorical_df], axis=1)

        return (full_df, all_activities)

    def mood_to_score(self, df):
        '''
        Input:
            DataFrame of journal entries
            List of default or custom moods ordered from worst to best
            Scale for normalized values
        Output:
            DataFrame of journal entries with an additional `mood_score` column, which normalizes in the mood.
        '''

        original_metric = {}
        num = 1
        for mood in self.ordered_moods:
            original_metric[mood] = num
            num += 1

        old_min = min(original_metric.values())
        old_max = max(original_metric.values())

        ordered_mood_scores = {}
        for mood in original_metric.keys():
            value = original_metric[mood]
            weighted_score = self.scale / (old_max - old_min) * (
                value - old_max) + self.scale
            ordered_mood_scores[mood] = weighted_score

        df['mood_score'] = df['mood'].map(ordered_mood_scores)

        return (df)


# def gen_linear_model(df, all_activities):

#     # including comment abt duplicately named tables
#     y = df[['mood_score']].to_numpy()
#     df = df.select_dtypes(include='bool')
#     X = df.astype(float).to_numpy()

#     X_train, X_test, y_train, y_test = train_test_split(X,
#                                                         y,
#                                                         test_size=0.33,
#                                                         random_state=42)

#     # generate random forest
#     model = RandomForestRegressor(n_estimators=100,
#                                   min_samples_leaf=8,
#                                   random_state=42)
#     fit = model.fit(X_train, y_train)
#     predictions = model.predict(X_test)

#     # get importance for all features
#     features = all_activities
#     importances = model.feature_importances_

#     feature_importance = {}

#     for x in range(len(features)):
#         feature_importance[features[x]] = importances[x]

#     k = Counter(feature_importance)
#     high = k.most_common(10)

#     top_feature_importances = high

#     # idx = (-importances).argsort()
#     # desc_features = [all_activities[i] for i in idx]
#     # top_features = desc_features[:10]

#     # top_importances = []
#     # for feature in top_features:
#     #     for i in range(len(features)):
#     #         if features[i] == feature:
#     #             top_importances.append(importances[i])

#     # top_importances = np.array(top_importances)
#     # new_indices = [x for x in range(10)]

#     # plt.figure(1)
#     # plt.title('feature importance of activities')
#     # plt.barh(range(len(new_indices)),
#     #          top_importances[new_indices],
#     #          color='b',
#     #          align='center')
#     # plt.yticks(range(len(new_indices)), top_features[new_indices])
#     # plt.xlabel('relative importance')

#     # reg = LinearRegression().fit(X_train, y_train)
#     # reg.score(X, y)
#     # prediction = reg.predict(X_test)

#     # return (prediction, Y_test)

#     # plt.show()

#     pass

# def gen_entries_over_time_hist(df):

#     df.date = pd.to_datetime(df.full_date).dt.to_period('M').dt.to_timestamp()

#     earliest_entry = min(df.date)
#     start_year = earliest_entry.year
#     start_month = earliest_entry.month

#     latest_entry = max(df.date)
#     end_year = latest_entry.year
#     end_month = latest_entry.month

#     all_months = [
#         date(m // 12, m % 12 + 1, 1)
#         for m in range(start_year * 12 + start_month - 1, end_year * 12 +
#                        end_month)
#     ]
#     num_entries = []

#     for month in all_months:
#         num_entries.append(len(df[df._month == month]))

#     ax = plt.subplot(111)
#     ax.bar(all_months, num_entries, width=25, color="darkorange")
#     ax.xaxis_date()
#     plt.title("# journal entries written, by month")

#     return (plt)

# def gen_wordcloud(df, ordered_moods=ORDERED_MOODS):

#     all_words = ''
#     stopwords = set(STOPWORDS)

#     words_by_mood = {}

#     for mood in ordered_moods:

#         words_by_mood[mood] = ''

#     for index, row in df.iterrows():

#         mood = row[['mood']].mood
#         note = row[['note']].note
#         if type(note) == str:
#             clean_note = note.translate(
#                 str.maketrans('', '', string.punctuation)).lower()
#         words_by_mood[mood] += clean_note

#     for mood, words in words_by_mood.items():
#         wordcloud = WordCloud(width=800,
#                               height=800,
#                               background_color='white',
#                               stopwords=stopwords,
#                               min_font_size=10,
#                               max_words=100).generate(words)
#         plt.figure(figsize=(8, 8), facecolor=None)
#         plt.imshow(wordcloud)
#         plt.axis("off")
#         plt.tight_layout(pad=0)

#         plt.savefig(mood + '.png')

#     pass

# if __name__ == '__main__':
#     print(gen_baseline_metrics())
