import os

class Config(object):
    DEBUG = True
    # flask-login uses sessions which require a secret Key
    SECRET_KEY = os.urandom(24)
    SQLALCHEMY_DATABASE_URI = "postgresql://xhan:azimuth2019@msds603.c25gop7kn8dl.us-west-2.rds.amazonaws.com:5432/azimuth"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    AWS_ACCESS_KEY_ID = 'AKIA4236MEX2CQJZIKEU'
    AWS_ACCESS_SECRET_KEY = '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'
    BUCKET_NAME = 'msds603-azimuth'
