import boto3
import base64

from config import Config
from werkzeug import secure_filename

config = Config()

s3_resource = boto3.resource(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_ACCESS_SECRET_KEY
)

s3_bucket = s3_resource.Bucket(config.BUCKET_NAME)


def create_key(username, shot_id, file, video=True):
    if video:
        return username + '/orig_video/' + str(shot_id) + '_' + \
               secure_filename(file.filename)
    else:
        # for gif, we pass in only filename instead of the file
        return username + '/gif/' + file


def upload_file(username, shot_id, file, video=True):
    """Upload file to the user's folder in S3 bucket"""
    # upload file to bucket
    if video:
        data = open('./instance/files/' + secure_filename(file.filename), 'rb')
        s3_bucket.put_object(Key=create_key(username, shot_id, file, video),
                             Body=data)
        return create_key(username, shot_id, file, video)
    else:
        data = open('./app/static/trajectory/'+file, 'rb')
        s3_bucket.put_object(Key=create_key(username, shot_id, file, video),
                             Body=data)
        return create_key(username, shot_id, file, video)
    # TODO: Error handle


def download_file(key):
    """Gets file from bucket with given key"""
    return s3_bucket.Object(key).get()


def get_imagedata(username, filename):
    tdta = download_file(username + '/gif/' + filename)['Body'].read()
    img64 = base64.b64encode(tdta).decode('utf8')

    return img64
