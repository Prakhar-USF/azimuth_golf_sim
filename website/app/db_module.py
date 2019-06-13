from app import application
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash


db = SQLAlchemy(application)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    subscription = db.Column(db.String(40), nullable=False)

    def __init__(self, username, email, password, subscription):
        self.username = username
        self.email = email
        self.set_password(password)
        self.subscription = subscription

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Shot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    ball_speed = db.Column(db.Float, nullable=False)
    launch_angle = db.Column(db.Float, nullable=False)
    azimuth = db.Column(db.Float, nullable=False)
    side_spin = db.Column(db.Integer, nullable=False)
    back_spin = db.Column(db.Integer, nullable=False)

    def __init__(self, user_id, ball_speed, launch_angle, azimuth,
                 side_spin, back_spin):
        self.user_id = user_id
        self.ball_speed = ball_speed
        self.launch_angle = launch_angle
        self.azimuth = azimuth
        self.side_spin = side_spin
        self.back_spin = back_spin


class Trajectory(db.Model):
    shot_id = db.Column(db.Integer, db.ForeignKey('shot.id'), primary_key=True)
    user_id = db.Column(db.Integer, primary_key=True)
    carry = db.Column(db.Float, nullable=False)
    offline = db.Column(db.Float, nullable=False)
    total_dist = db.Column(db.Float, nullable=False)
    gif_name = db.Column(db.String(40), nullable=False)

    def __init__(self, shot_id, user_id, carry, offline, total_dist, gif_name):
        self.shot_id = shot_id
        self.user_id = user_id
        self.carry = carry
        self.offline = offline
        self.total_dist = total_dist
        self.gif_name = gif_name


class Files(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    shot_id = db.Column(db.Integer, db.ForeignKey('shot.id'), nullable=False)
    video_key = db.Column(db.String(80), unique=True, nullable=False)
    gif_key = db.Column(db.String(80), unique=True, nullable=False)
    username = db.Column(db.String(80), nullable=False)
    date_upload = db.Column(db.String(30), nullable=False)

    def __init__(self, user_id, shot_id, video_key, gif_key,
                 username, date_upload):
        self.user_id = user_id
        self.shot_id = shot_id
        self.video_key = video_key
        self.gif_key = gif_key
        self.username = username
        self.date_upload = date_upload


class Contest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    description = db.Column(db.String(300), nullable=False)
    start_date = db.Column(db.String(30), nullable=False)
    end_date = db.Column(db.String(30), nullable=False)
    target_x = db.Column(db.Float, nullable=False)  # target total distance
    target_y = db.Column(db.Float, nullable=False)  # target offline

    def __init__(self, title, description, start_date, end_date,
                 target_x, target_y):
        self.title = title
        self.description = description
        self.start_date = start_date
        self.end_date = end_date
        self.target_x = target_x
        self.target_y = target_y


class UserContest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    shot_id = db.Column(db.Integer, db.ForeignKey('shot.id'), nullable=False)
    contest_id = db.Column(db.Integer, db.ForeignKey('contest.id'),
                           nullable=True)
    enter_date = db.Column(db.String(30), nullable=False)

    def __init__(self, user_id, shot_id, contest_id, enter_date):
        self.user_id = user_id
        self.shot_id = shot_id
        self.contest_id = contest_id
        self.enter_date = enter_date


db.create_all()
db.session.commit()
