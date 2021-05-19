from peewee import *

db = None


class DBParameters:
    DB_NAME = 'HorsingAround'
    USER = 'maartjehuveneers'
    PASSWORD = 'p4RwxJCchw7Ljqhv'
    HOST = 'www.jacobkamminga.nl'
    PORT = 3306


# Establish connection to database
def connect(db_name, username, password, host, port):
    db = MySQLDatabase(db_name, user=username, password=password, host=host, port=port)
    db.connect()
    print("Connected to: " + host + ":" + str(port))


def create_experiment(uid, name, horse, date, acc, balanced_acc, f_score_avg, mcc_score, recall, confusion, params, desc):
    Experiment.create(key=uid,
                      username=name,
                      test_horse=horse,
                      date=date,
                      accuracy_experiment=acc,
                      balanced_accuracy_experiment=balanced_acc,
                      fscore=f_score_avg,
                      mcc=mcc_score,
                      recall=recall,
                      confusion_matrix=confusion,
                      parameters=params,
                      description=desc)


def create_activity(uid, horse, activity, accuracy, recall, specificity, precision, tp, tn, fp, fn, index):
    Activity.create(key=uid,
                    test_horse=horse,
                    activity=activity,
                    accuracy_activity=accuracy,
                    recall_activity=recall[index],
                    specificity=specificity,
                    precision=precision[index],
                    TP=tp,
                    TN=tn,
                    FP=fp,
                    FN=fn)


class Experiment(Model):
    key = UUIDField()
    username = TextField()
    test_horse = TextField()
    date = DateField()
    accuracy_experiment = FloatField()
    balanced_accuracy_experiment = FloatField()
    fscore = FloatField()
    mcc = FloatField()
    recall = FloatField()
    confusion_matrix = BlobField()
    parameters = TextField()
    description = TextField()

    class Meta:
        database = db


class Activity(Model):
    key = UUIDField()
    test_horse = TextField()
    activity = TextField()
    accuracy_activity = FloatField()
    recall_activity = FloatField()
    specificity = FloatField()
    precision = FloatField()
    TP = IntegerField()
    TN = IntegerField()
    FP = IntegerField()
    FN = IntegerField()

    class Meta:
        database = db
