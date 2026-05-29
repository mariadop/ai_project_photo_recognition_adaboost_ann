from scipy.ndimage import rotate, shift
import numpy as np

def augment_image_flat(x):
    img = x.reshape(24, 24)

    angle = np.random.uniform(-5, 5)
    img = rotate(img, angle, reshape=False, mode="nearest")

    dx = np.random.uniform(-1, 1)
    dy = np.random.uniform(-1, 1)
    img = shift(img, shift=(dy, dx), mode="nearest")

    return img.reshape(-1)

def augment_df(X_train, y_train, min_target=100, max_target=160):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()

    X_new = []
    y_new = []

    unique_classes, counts = np.unique(y_train, return_counts=True)

    for cls, count in zip(unique_classes, counts):
        X_cls = X_train[y_train == cls]
        n = len(X_cls)

        if n > max_target:
            idx = np.random.choice(n, size=max_target, replace=False)
            X_cls_final = X_cls[idx]
        elif n < min_target:
            to_fill = min_target - n
            idx = np.random.choice(n, size=to_fill, replace=True)
            aug = np.array([augment_image_flat(X_cls[i]) for i in idx])
            X_cls_final = np.vstack([X_cls, aug])
        else:
            X_cls_final = X_cls

        X_new.append(X_cls_final)
        y_new.append(np.full(len(X_cls_final), cls))


    return np.vstack(X_new), np.concatenate(y_new)