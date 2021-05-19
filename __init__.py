import keras
from db import DBParameters as dbp
from db import connect
from preprocessing import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape

# Hyper-parameters
BATCH_SIZE = 600
EPOCHS = 2

# Datasets
PATH = '/home/wannes/Downloads/JacobHorses/*'
FILES = sorted(glob.glob(PATH))

# Set to false to disable database support
db_enabled = True


def build_classifier():
    model_m = Sequential()
    model_m.add(Reshape((TIME_PERIODS, 6), input_shape=(input_shape,)))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Dense(100, activation='relu'))
    model_m.add(Flatten())
    model_m.add(Dense(num_classes, activation='softmax'))
    return model_m


def train_classifier(input, output):
    model = build_classifier()

    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='accuracy', patience=1)
    ]

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
    history = model.fit(input,
                        output,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks_list,
                        validation_split=0.2,
                        verbose=1)

    return history, model


def main():

    if db_enabled:
        connect(dbp.DB_NAME, dbp.USER, dbp.PASSWORD, dbp.HOST, dbp.PORT)

    df = create_dataframe(FILES)

    for s in SUBJECTS:
        xtrain, ytrain, xtest, ytest = preprocess(df, s)

        #debug
        print(xtrain)
        print(ytrain)

        # history, model = train_classifier(xtrain, ytrain)
        # test_prediction = model.predict(xtest)
        #
        # max_test_prediction = np.argmax(test_prediction, axis=1)
        # max_y_test = np.argmax(ytest, axis=1)
        #
        # print('Result on test set: ' + max_test_prediction + '\t' + max_y_test)


if __name__ == '__main__':
    main()
