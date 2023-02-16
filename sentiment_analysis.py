# pip install --user pandas emoji nltk xgboost
import numpy as np
import pandas as pd
import re
import emoji

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.arlstem import ARLSTem

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


positive_tweets__file = "Data/PositiveTweets.tsv"
negative_tweets__file = "Data/NegativeTweets.tsv"
bad_words__file = "Data/bad_words.txt"


def main():
    df = read_data_files()
    print("Data read from files")

    add_features(df)
    print("Features added")

    build_models(df)

def read_data_files():
    neg_df = pd.read_csv(negative_tweets__file, sep='\t',
                         header=None, names=["sentiment", "content"], encoding='utf-8')

    pos_df = pd.read_csv(positive_tweets__file, sep='\t',
                         header=None, names=["sentiment", "content"], encoding='utf-8')

    return pd.concat([neg_df, pos_df])


def add_features(df):
    add_features_before_preprocessing(df)
    add_features_after_preprocessing(df)

def add_features_before_preprocessing(df):
    df["content_length_before"] = df["content"].apply(len)
    df["tokens_count_before"] = df["content"].apply(lambda x: len(word_tokenize(x)))
    df["sentences_count_before"] = df["content"].apply(lambda x: len(sent_tokenize(x)))

    df["hashtags_count"] = df["content"].apply(count_hashtags)
    df["bad_words_count"] = df["content"].apply(count_bad_words)
   
    df["emojis_count"] = df["content"].apply(count_emojis)
    add_emojis_features(df)
    add_words_features(df)

def add_features_after_preprocessing(df):
    df["normalized_content"] = df["content"].apply(normalize_text)
    df["content_length_after"] = df["normalized_content"].apply(len)
    df["tokens_count_after"] = df["normalized_content"].apply(lambda x: len(word_tokenize(x)))


def count_links(text):
    url_pattern = re.compile(r'(https?://[^\s]+)')
    urls = re.findall(url_pattern, text)
    return len(urls)

def count_hashtags(text):
    return len(re.findall(r'#', text))

def count_bad_words(text):
    bad_words = read_bad_words_from_the_file()
    words = word_tokenize(text)
    words = [word for word in words if word in bad_words]
    return len(words)

def count_emojis(text):
    return emoji.emoji_count(text)

def add_emojis_features(df):
    love_emojis = ['❣', '💍', '🤎', '💌', '🧡', '💙', '💛', '🤍', '💗', '💓', '💋', '💝', '💚', '🖤', '💟'
                   '🔥', '\u200d', '🩹', '💘', '💜', '❤', '💟', '💞', '💕', '💖', '😍', '😘', '😗', '😙', '♥️'
                   '😚', '😽', '😇', '🫶', '😊', '☺️', '🤗', '🤩', '🥰', '😻', '🙈', '🙊', '🫦', '👄', '🫀', '💏', '💑', '💋']
    broken_heart_emoji = ['💔']
    happy_emojis = ['😄', '😹', '😃', '😆', '😅',
                    '😀', '🤣', '😂', '😁', '😝', '😸', '😇', '🤪', '😎', '🕺', '💃']
    sad_emojis = ['\U0001fae4', '🥲', '😟', '😩', '😢', '😓', '😥', '🙃', '💅'
                  '😞', '😔', '🙁', '☹️', '😿', '😖', '️💔', '😰', '😭', '😣', '☹', '😕', '🥵', '🥴', '🤕', '🤒']
    smile_emojis = ['🙂', '🙃']
    thinking_emojis = ['🤔', '🤨', '🧐', '🤓']
    flowers_emojis = ['💐', '🌸', '🏵️', '🌹', '🌺', '🌻', '🌼', '🌷', '🥀', '☘️', '🍁']
    moon_and_sun_emojis = ['🌝', '🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗',
                           '🌘', '🌙', '🌚', '🌛', '🌜', '☀️', '🌞', '⭐', '🌟', '🌠', '✨']
    hand_emojis = ['👉', '✊', '👌', '\U0001faf5', '🤘', '🤞', '🤝', '\U0001faf1', '👐', '👎', '\U0001faf0', '🤟', '🤜', '🤲', '👋', '👈', '🤚', '🙌',
                   '🙏', '🤌', '🤏', '\U0001faf2', '👆', '🖐', '💪', '✌', '🤙', '✋', '👊', '\U0001faf4', '🖖', '👍', '☝', '\U0001faf3', '👏', '🤛', '👇']
    surprising_emojis = ['🤨', '😐', '😑', '😶', '😮', '😯', '😲', '😧', '😦', '😨', '😱', '🤯', '😵', '😵‍💫', '🧐']
    angry_emojis = ['😑', '😐', '😤', '😮‍💨', '🤬', '😡', '😠', '🤢', '🤮', '👿', '😈']
    prohibited_emojis = ['🚫', '🔇', '🔕', '🛑',
                         '🆘', '⛔', '🛑', '📛', '❌', '⭕', '🔞', '☢️']


    df['love_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in love_emojis)
    )
    df['broken_heart'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in broken_heart_emoji)
    )
    df['happy_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in happy_emojis)
    )
    df['sad_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in sad_emojis)
    )
    df['smile_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in smile_emojis)
    )
    df['thinking_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in thinking_emojis)
    )
    df['flowers_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in flowers_emojis)
    )
    df['moon_and_sun_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in moon_and_sun_emojis)
    )
    df['hand_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in hand_emojis)
    )
    df['surprising_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in surprising_emojis)
    )
    df['angry_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in angry_emojis)
    )
    df['prohibited_emojis'] = df['content'].apply(
        lambda x: sum(emoji in x for emoji in prohibited_emojis)
    )


def add_words_features(df):
    df['يارب'] = df['content'].apply(lambda x: count_substrings(x, ['يارب', 'يا رب']))
    df['لله'] = df['content'].str.count('لله')
    df['الحمد'] = df['content'].str.count('الحمد')
    
    good_words =['جميل', 'جمال', 'حب', 'خير', 'صباح']
    df['good_words'] = df['content'].apply(lambda x: count_substrings(x, good_words))

    bad_words = ['عيب', 'غلط', 'تعب', 'كئيب', 'قرف', 'مرض', 'موت', 'سيء', 'مشكل', 'خرا', 'زفت', 'ظلم', 'كذب']
    df['bad_words'] = df['content'].apply(lambda x: count_substrings(x, bad_words))




def count_substrings(row, substrings):
    count = 0
    for substring in substrings:
        count += row.count(substring)
    return count


def normalize_text(text):
    text = remove_stop_words(text)
    text = remove_non_arabic(text)
    text = remove_consecutive_redundant_characters(text, 3)
    text = stem_words(text)
    text = remove_stop_words(text)

    return text

def remove_stop_words(text):
    stop_words = set(stopwords.words("arabic"))
    words = word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words]
    return " ".join(words)
    
def remove_non_arabic(text):
    return re.sub(r'[^\u0621-\u064A\s]', '', text)

def remove_consecutive_redundant_characters(text, number_of_consecutive_characters):
    text += "\0"
    result = ""

    count = 1
    prev_char = text[0]

    for i in range(1, len(text)):
        current_char = text[i]
        if current_char == prev_char:
            count += 1
        else:
            if count > number_of_consecutive_characters:
                result += prev_char
            else:
                result += prev_char * count

            count = 1
            prev_char = current_char

    return result


def stem_words(text):
    stemmer = ARLSTem()
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


def read_bad_words_from_the_file():
    stop_words = set()
    with open(bad_words__file, "r", encoding="utf-8") as file:
        for line in file:
            stop_words.add(line.strip())

    return stop_words







def build_models(df):
    X = df.drop(columns=['sentiment', 'content', 'normalized_content'])
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)


    build_decision_tree(X_train, X_test, y_train, y_test)
    build_xgboost_model(X_train, X_test, y_train, y_test)
    build_neural_networks_model(X_train, X_test, y_train, y_test)
    build_knn_model(X_train, X_test, y_train, y_test)
    build_naive_base_model(X_train, X_test, y_train, y_test)


def build_decision_tree(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=123)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n\nDecision Tree Model:")
    print_measures(y_test, y_pred)
    # print_features_importances(clf, X_train)

def build_xgboost_model(X_train, X_test, y_train, y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    clf = xgb.XGBClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nXGBoost Model:")
    print_measures(y_test, y_pred)


def build_neural_networks_model(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nNeural Networks Model:")
    print_measures(y_test, y_pred)

def build_knn_model(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nKNN Model:")
    print_measures(y_test, y_pred)


def build_naive_base_model(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nNaive Base Model:")
    print_measures(y_test, y_pred)




def print_measures(y_test, y_pred):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))
    print("F-measure:", metrics.f1_score(y_test, y_pred, average='weighted'))
    print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))


def print_features_importances(clf, X_train):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X_train.columns

    print("Feature importances:")
    for f in range(X_train.shape[1]):
        print(f"{features[indices[f]]}: {importances[indices[f]]}")

if __name__ == "__main__":
    main()
    