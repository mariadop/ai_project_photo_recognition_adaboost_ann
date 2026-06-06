import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean(filename):
    return np.loadtxt(filename, skiprows=2)


data_x = load_and_clean('x24x24.txt')
data_y = load_and_clean('y24x24.txt')
data_z = load_and_clean('z24x24.txt')
full_data = np.vstack([data_x, data_y, data_z])

X = full_data[:, :576]
y = full_data[:, 578]

X_buff, X_test, y_buff, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=67,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_buff, y_buff,
    test_size=0.2,
    random_state=67,
    stratify=y_buff
)


def rows_to_set(arr):
    return set(map(lambda r: r.tobytes(), arr))

s_test = rows_to_set(X_test)
s_train = rows_to_set(X_train)
s_val = rows_to_set(X_val)

overlap_train_test = len(s_train & s_test)
overlap_val_test = len(s_val & s_test)
overlap_train_val = len(s_train & s_val)

print('overlap_train_test=', overlap_train_test)
print('overlap_val_test=', overlap_val_test)
print('overlap_train_val=', overlap_train_val)

if overlap_train_test+overlap_val_test+overlap_train_val == 0:
    print('No overlap detected between train/val/test sets.')
    exit(0)
else:
    print('Overlap detected!')
    exit(2)
