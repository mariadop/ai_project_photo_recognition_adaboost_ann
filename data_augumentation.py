from scipy.ndimage import rotate, shift
import numpy as np
import pandas as pd

def augment_image_flat(x):
    img = x.reshape(24, 24)

    # mały obrót
    angle = np.random.uniform(-5, 5)
    img = rotate(img, angle, reshape=False, mode="nearest")

    # małe przesunięcie max 1-2 piksele
    dx = np.random.uniform(-1, 1)
    dy = np.random.uniform(-1, 1)
    img = shift(img, shift=(dy, dx), mode="nearest")

    return img.reshape(-1)


def augument_df(X_train, y_train):
    X_aug = [X_train]
    y_aug = [y_train]


    unique_classes, counts = np.unique(y_train, return_counts=True)
    target = counts.max()
    for y, count in zip(unique_classes,counts):
       X = X_train[y_train==y]
       to_fill = target-count
       new_img = []
       for c in range(to_fill):
            img = X[np.random.randint(len(X))]
            new_img.append(augment_image_flat(img))
       if len(new_img)>0:
            X_aug.append(np.array(new_img))
            y_aug.append(np.full(to_fill, y))         

    return np.vstack(X_aug), np.concatenate(y_aug)
    