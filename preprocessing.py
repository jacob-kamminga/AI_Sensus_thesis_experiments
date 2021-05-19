import numpy as np
import pandas as pd
import glob
from sklearn import preprocessing
from scipy import stats
from keras.utils import np_utils
from sklearn.utils import shuffle

# ------------------------------------------ #
#  Pre-Processing Constants:
# ------------------------------------------ #

# Label encoder used to get a numeric representation of a label
le = preprocessing.LabelEncoder()

# The activities
LABELS = ['grazing', 'running', 'trotting', 'standing', 'walking-natural', 'walking-rider']

# Add columns to drop from dataframe
REMOVE_COLUMNS = ['Mx', 'My', 'Mz', 'A3D', 'G3D', 'M3D']

# Add subjects you want to include
SUBJECTS = ['Galoway', 'Patron', 'Happy', 'Driekus']

# Amount of features (xyz acc / xyz gyr)
N_FEATURES = 6

# Name of the column used as output
OUTPUT_LABEL = 'ActivityEncoded'

# Sliding windows parameters
TIME_PERIODS = 200
STEP_DISTANCE = 100

# Datasets
PATH = '/content/drive/MyDrive/Bachelor GP/Let there be IMU data/datasets/JacobHorse/*'
FILES = sorted(glob.glob(PATH))


# ------------------------------------------ #
#  Helper functions:
# ------------------------------------------ #

input_shape = None
num_classes = None


def create_dataframe(files):
    """
    Simple function to set up dataframe and initial clean-up of the data
    files: path to files
    returns: dataframe
    """
    result = pd.DataFrame()
    # Pick only the files in SUBJECTS
    matching = [f for f in files if any(s in f for s in SUBJECTS)]

    for file in matching:
        csv = pd.read_csv(file)
        csv['filename'] = file
        result = result.append(csv)

    # remove redundant columns
    result.drop(REMOVE_COLUMNS, axis=1, inplace=True)
    result = relabel_activities(result)
    # create a new column with a unique integer value for each label
    result[OUTPUT_LABEL] = le.fit_transform(result['label'].values.ravel())

    return result


def relabel_activities(df):
    df['label'] = df['label'].replace(to_replace=['trotting-natural'], value='trotting')
    df['label'] = df['label'].replace(to_replace=['trotting-rider'], value='trotting')
    df['label'] = df['label'].replace(to_replace=['running-natural'], value='running')
    df['label'] = df['label'].replace(to_replace=['running-rider'], value='running')
    result = df[df['label'].isin(LABELS)]

    return result


def split_by_subject(df, name):
    test = df[df['filename'].str.contains(name)]
    train = df[~df['filename'].str.contains(name)]
    return train, test


def feature_scaling(df):
    train_x_max = df['Ax'].max()
    train_y_max = df['Ay'].max()
    train_z_max = df['Az'].max()

    train_gx_max = df['Gx'].max()
    train_gy_max = df['Gy'].max()
    train_gz_max = df['Gz'].max()

    pd.options.mode.chained_assignment = None

    # divide all 3 axis with the max value in the training set
    df['Ax'] = df['Ax'] / train_x_max
    df['Ay'] = df['Ay'] / train_y_max
    df['Az'] = df['Az'] / train_z_max

    df['Gx'] = df['Gx'] / train_gx_max
    df['Gy'] = df['Gy'] / train_gy_max
    df['Gz'] = df['Gz'] / train_gz_max

    return df


def create_windows(df, time_steps, step, label_name):
    windows = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        axs = df['Ax'].values[i: i + time_steps]
        ays = df['Ay'].values[i: i + time_steps]
        azs = df['Az'].values[i: i + time_steps]

        gxs = df['Gx'].values[i: i + time_steps]
        gys = df['Gy'].values[i: i + time_steps]
        gzs = df['Gz'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        windows.append([axs, ays, azs, gxs, gys, gzs])
        labels.append(label)
    # Bring the segments into a better shape
    reshaped_windows = np.asarray(windows, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_windows, labels


# Reshape input into a format compatible with the NN
def reshape_input(x, shape):
    result = x.reshape(x.shape[0], shape)
    return result


# Apply one hot coding to output
def encode_output(y, classes):
    result = np_utils.to_categorical(y, classes)
    return result


def preprocess_training(df):
    train = feature_scaling(df)
    x_train, y_train = create_windows(train, TIME_PERIODS, STEP_DISTANCE, OUTPUT_LABEL)
    x_train, y_train = shuffle(np.array(x_train), np.array(y_train))

    num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
    num_classes = le.classes_.size
    input_shape = (num_time_periods * num_sensors)

    x_train = reshape_input(x_train, input_shape)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    y_train = encode_output(y_train, num_classes)

    return x_train, y_train


def preprocess_test(df):
    test = feature_scaling(df)
    x_test, y_test = create_windows(test, TIME_PERIODS, STEP_DISTANCE, OUTPUT_LABEL)

    num_time_periods, num_sensors = x_test.shape[1], x_test.shape[2]
    num_classes = le.classes_.size
    input_shape = (num_time_periods * num_sensors)

    x_test = reshape_input(x_test, input_shape)
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    y_test = encode_output(y_test, num_classes)

    return x_test, y_test


def preprocess(df, test_subject):
    train, test = split_by_subject(df, test_subject)
    x_train, y_train = preprocess_training(train)
    x_test, y_test = preprocess_test(test)

    return x_train, y_train, x_test, y_test
