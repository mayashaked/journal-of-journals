import pandas as pd 
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

DIRNAME = os.path.dirname(os.path.abspath("__file__"))
PATH_TO_CSV = os.path.join(DIRNAME, 'daylio_export.csv')


def gen_baseline_metrics(path_to_csv = PATH_TO_CSV):

    ugly_df = pd.read_csv(path_to_csv, index_col=False)
    df, all_activities = clean_df(ugly_df)
    wordcloud = gen_wordcloud(df)

    # entries_over_time = gen_entries_over_time_hist(df)
    # wordcloud = gen_wordcloud(df)

    linear_model = gen_linear_model(df, all_activities)

    return(df)


def clean_df(df):

    df.full_date = pd.to_datetime(df.full_date)
    df_with_one_hot_encoding, all_activities = convert_activities_to_categorical(df)
    df_with_mood_scores = mood_to_score(df_with_one_hot_encoding)
    
    return(df_with_mood_scores, all_activities)

def gen_dot_plot(df):

    import matplotlib.patches as p
    import matplotlib.pyplot as plt
    import math

    fig = plt.figure(1, figsize=(4,4))
    ax = plt.subplot(111, aspect='equal')
    colors=['red', 'orange', 'yellow', 'green', 'blue']

    ps = pd.Series([i for i in df.mood])
    counts = ps.value_counts()

    x=0.05
    y=0.9
    m=0
    cols=1
    for i in counts:
            for j in range(i):
                    p1 = p.Circle((x, y), 0.05, fc=colors[m],
    edgecolor='white')
                    x=x+0.1
                    ax.add_patch(p1)
                    cols=cols+1
                    if (cols>10):
                            cols=1
                            x=0.05
                            y=y-0.1
            m=m+1

    x=0.05
    y=0.3
    m=0
    for i in counts.index:
            p2 = p.Circle((x, y), 0.05, fc=colors[m], edgecolor='white')
            ax.add_patch(p2)
            plt.text(x+0.17, y-0.03, str(int(i))+" blah", ha="center",
    family='sans-serif', size=11, color='b' )
            #y=y-0.13
            x=x+0.35
            m=m+1

    ax.axis('off')
    plt.title("Dot Matrix by Mood", color='b', family='sans-serif', size=14)
    plt.show()


def gen_linear_model(df, all_activities):

    # including comment abt duplicately named tables
    y = df[['mood_score']].to_numpy()
    df = df.select_dtypes(include='bool')
    X = df.astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_sate=42) 

    reg = LinearRegression().fit(X_train, y_train)
    reg.sore(X, y)
    prediction = reg.predict(X_test)

    return(prediction, Y_test)


def mood_to_score(df, ordered_moods = ['awful', 'bad', 'meh', 'good', 'rad'], scale = 10):
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
    for mood in ordered_moods:
        original_metric[mood] = num
        num += 1

    old_min = min(original_metric.values())
    old_max = max(original_metric.values())

    ordered_mood_scores = {}
    for mood in original_metric.keys():
        value = original_metric[mood]
        weighted_score = scale / (old_max - old_min) * (value - old_max) +  scale
        ordered_mood_scores[mood] = weighted_score

    df['mood_score'] = df['mood'].map(ordered_mood_scores)

    return(df)


def convert_activities_to_categorical(df):
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


    categorical_df = pd.DataFrame(categorical_activity_matrix, columns = all_activities)

    full_df = pd.concat([df, categorical_df], axis=1)

    return(full_df, all_activities)


def gen_entries_over_time_hist(df):

    df.date = pd.to_datetime(df.full_date).dt.to_period('M').dt.to_timestamp()

    earliest_entry = min(df.date)
    start_year = earliest_entry.year
    start_month = earliest_entry.month

    latest_entry = max(df.date)
    end_year = latest_entry.year
    end_month = latest_entry.month

    all_months = [date(m//12, m%12+1, 1) for m in range(start_year*12+start_month-1, end_year*12+end_month)]
    num_entries = []

    for month in all_months:
        num_entries.append(len(df[df._month == month]))

    ax = plt.subplot(111)
    ax.bar(all_months, num_entries, width = 25, color = "darkorange")
    ax.xaxis_date()
    plt.title("# journal entries written, by month")

    return(plt)

def gen_wordcloud(df, ordered_moods = ['awful', 'bad', 'meh', 'good', 'rad']):

    all_words = ''
    stopwords = set(STOPWORDS)

    words_by_mood = {}
    for mood in ordered_moods:
        words_by_mood[mood] = ''

    for index, row in df.iterrows():
        mood = row[['mood']]
        note = row[['note']]
        clean_note = " ".join(note) + " "
        words_by_mood[mood.mood] += clean_note


    for mood, words in words_by_mood.items():

        wordcloud = WordCloud(width = 800, height = 800, 
            background_color ='white', 
            stopwords = stopwords, 
            min_font_size = 10).generate(words)

        plt.figure(figsize = (8,8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)

        plt.savefig(mood + '.png')

if __name__ == '__main__':
    print(gen_baseline_metrics())
