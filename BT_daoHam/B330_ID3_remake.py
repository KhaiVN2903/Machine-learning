import pandas as pd
import numpy as np

from math import log2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# 1. TẠO DATASET
# =====================================================

data = {
    "amount": [100, 200, 150, 5000, 7000,
               6000, 300, 8000, 250, 5500],

    "international": [
        "No", "No", "No", "Yes", "Yes",
        "Yes", "No", "Yes", "No", "Yes"
    ],

    "night": [
        "No", "No", "Yes", "Yes", "Yes",
        "No", "Yes", "Yes", "No", "Yes"
    ],

    "fraud": [
        0, 0, 0, 1, 1,
        1, 0, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# =====================================================
# 2. CHUYỂN amount THÀNH 2 NHÓM
# =====================================================

df["amount_group"] = np.where(
    df["amount"] >= 1000,
    "High",
    "Low"
)

df = df.drop(columns=["amount"])


# =====================================================
# 3. ENTROPY
# =====================================================

def entropy(y):

    values, counts = np.unique(
        y,
        return_counts=True
    )

    total = len(y)

    result = 0

    for count in counts:

        p = count / total

        result -= p * log2(p)

    return result


# =====================================================
# 4. INFORMATION GAIN
# =====================================================

def information_gain(df, feature, target):

    parent_entropy = entropy(df[target])

    values = df[feature].unique()

    weighted_entropy = 0

    for value in values:

        subset = df[df[feature] == value]

        weight = len(subset) / len(df)

        weighted_entropy += (
            weight * entropy(subset[target])
        )

    gain = parent_entropy - weighted_entropy

    return gain


# =====================================================
# 5. TÌM FEATURE TỐT NHẤT
# =====================================================

def best_feature(df, features, target):

    gains = {}

    for feature in features:

        gains[feature] = information_gain(
            df,
            feature,
            target
        )

    best = max(
        gains,
        key=gains.get
    )

    return best


# =====================================================
# 6. XÂY DỰNG CÂY ID3
# =====================================================

def id3(df, features, target):

    # Nếu tất cả mẫu cùng một class
    if len(df[target].unique()) == 1:

        return df[target].iloc[0]

    # Không còn feature
    if len(features) == 0:

        return df[target].mode()[0]

    # Chọn feature có Information Gain lớn nhất
    best = best_feature(
        df,
        features,
        target
    )

    tree = {
        best: {}
    }

    remaining_features = [
        feature
        for feature in features
        if feature != best
    ]

    # Xây cây cho từng giá trị
    for value in df[best].unique():

        subset = df[
            df[best] == value
        ]

        if len(subset) == 0:

            tree[best][value] = (
                df[target].mode()[0]
            )

        else:

            tree[best][value] = id3(
                subset,
                remaining_features,
                target
            )

    return tree


# =====================================================
# 7. PREDICT
# =====================================================

def predict_one(tree, sample):

    # Nếu cây đã trả về class
    if not isinstance(tree, dict):

        return tree

    feature = next(iter(tree))

    value = sample[feature]

    subtree = tree[feature]

    if value not in subtree:

        return 0

    return predict_one(
        subtree[value],
        sample
    )


def predict(tree, X):

    predictions = []

    for _, row in X.iterrows():

        predictions.append(
            predict_one(tree, row)
        )

    return np.array(predictions)


# =====================================================
# 8. CHIA TRAIN / TEST
# =====================================================

X = df.drop(columns=["fraud"])

y = df["fraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# =====================================================
# 9. HUẤN LUYỆN ID3
# =====================================================

train_data = X_train.copy()

train_data["fraud"] = y_train.values

features = list(X_train.columns)

tree = id3(
    train_data,
    features,
    "fraud"
)


print("\nDecision Tree:")
print(tree)


# =====================================================
# 10. DỰ ĐOÁN
# =====================================================

y_pred = predict(
    tree,
    X_test
)


# =====================================================
# 11. ĐÁNH GIÁ
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n===== KẾT QUẢ =====")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)