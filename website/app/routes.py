import os
import re
import math
import arrow
import pickle
import numpy as np
from datetime import datetime, timedelta

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask import render_template, redirect, url_for, session, flash
from flask_login import current_user, login_user, login_required, logout_user

from werkzeug import secure_filename
from wtforms.validators import DataRequired, Email
from wtforms import SubmitField, StringField, PasswordField

from app.predict import Step2
from app.s3_functions import *
from app import application, make_traj, db_module, db


def extract_factors(video_path: str) -> dict:
    """
    Extract the five factors from the video for trajectory prediction.
    Five factors are: ball speed, launch angle, azimuth, side-spin, back-spin

    :param video_path: local file path of the original video.
    :return: five factors in a dictionary {factor_name: value}
    """
    # load prerequisites
    impact_finder = pickle.load(
        open('./app/static/models/impact_finder.sav', 'rb'))
    classify_bbox = pickle.load(
        open('./app/static/models/classify_bbox_new.sav', 'rb'))
    azimuth_regr = pickle.load(
        open('./app/static/models/azimuth_regr.sav', 'rb'))
    back_spin_regr = pickle.load(
        open('./app/static/models/back_spin_regr.sav', 'rb'))
    ball_speed_regr = pickle.load(
        open('./app/static/models/ball_speed_regr.sav', 'rb'))
    launch_angle_regr = pickle.load(
        open('./app/static/models/launch_angle_regr.sav', 'rb'))
    side_spin_regr = pickle.load(
        open('./app/static/models/side_spin_regr.sav', 'rb'))

    # hard-code ball_init for now
#     ball_init = [357, 781, 25, 22]

    # process raw video and extract five input features -> dict
    s2 = Step2()
    factors, f2 = s2.get_shot_factors(video_path, impact_finder, classify_bbox,
                                      azimuth_regr, back_spin_regr,
                                      ball_speed_regr, launch_angle_regr,
                                      side_spin_regr)

    if factors["ball_speed"] < 70:
        factors["ball_speed"] = 70.23
    if factors["ball_speed"] > 120:
        factors["ball_speed"] = 120.12

    if factors["launch_angle"] < 15:
        factors["launch_angle"] = 15.13
    if factors["launch_angle"] > 40:
        factors["launch_angle"] = 40.54

    if factors["azimuth"] < -15:
        factors["azimuth"] = -15.45
    if factors["azimuth"] > 15:
        factors["azimuth"] = 15.17

    if factors["side_spin"] < -2000:
        factors["side_spin"] = -1986.87
    if factors["side_spin"] > 2000:
        factors["side_spin"] = 2045.91

    if factors["back_spin"] < 1000:
        factors["back_spin"] = 1005.98
    if factors["back_spin"] > 7500:
        factors["back_spin"] = 7472.67

    return factors


def predict(factors: dict, gif_name: str, is_auth=False) -> tuple:
    """
    Make trajectory prediction.

    :param factors: five factors in dictionary format
    :param gif_name: the file name for output gif file
    :param is_auth: check if the user has logged in
    :return: carry(x1), offline(y) and total distance(x2) of the shot
    """
    simulation = make_traj.TrackSimulation()

    reg_carry = pickle.load(
        open('./app/static/models/reg_carry.pkl', 'rb'))
    reg_tot_dist = pickle.load(
        open('./app/static/models/reg_tot_dist.pkl', 'rb'))
    reg_offline_ratio = pickle.load(
        open('./app/static/models/reg_offline_ratio.pkl', 'rb'))
    reg_peak_height = pickle.load(
        open('./app/static/models/reg_peak_height.pkl', 'rb'))

    x1, y, x2, track = simulation.traj(factors, reg_carry, reg_tot_dist,
                                       reg_peak_height, reg_offline_ratio)

    if is_auth:
        # simulation.make_anime(track, gif_name)
        simulation.make_plotly(track, gif_name)

    return x1, y, x2


def delete_tmp_file(video, gif, plotly=True):
    os.remove(video)
    os.remove(video+'.wav')
    if not plotly:
        os.remove(gif)


class RegistrationForm(FlaskForm):
    """FlaskForm class for user registration."""
    username = StringField('', validators=[DataRequired()])
    email = StringField('', validators=[DataRequired(), Email()])
    password = PasswordField('', validators=[DataRequired()])
    password_confirmation = PasswordField('', validators=[DataRequired()])
    submit = SubmitField('Submit')


class LogInForm(FlaskForm):
    """FlaskForm class for user log-ins."""
    username = StringField('', validators=[DataRequired()])
    password = PasswordField('', validators=[DataRequired()])
    submit = SubmitField('Login')


class UploadFileForm(FlaskForm):
    """FlaskForm class for uploading file."""
    file_selector = FileField('', validators=[FileRequired()])
    submit = SubmitField('Upload')


@application.before_request
def keep_session():
    """Keep user session alive for 30 minutes."""
    session.permanent = True
    application.permanent_session_lifetime = timedelta(minutes=30)


@application.route('/register', methods=['GET', 'POST'])
def register():
    """
    Allow user to register, check for username/password validity,
    and store the user information in database.

    :return: No value returned, render the registration page only
    """
    reg_form = RegistrationForm()

    if reg_form.validate_on_submit():
        username = reg_form.username.data
        password = reg_form.password.data
        password_confirmation = reg_form.password_confirmation.data
        email = reg_form.email.data

        # check username, only allow numbers and alphabets
        match = re.search("[^a-zA-Z0-9]+", username)
        if match:
            flash('Error - please do not use special characters.', 'error')

        # check pwd and pwd_confirmation, they should be the same
        if password != password_confirmation:
            flash("Error - Password doesn't match", 'error')

        # check if username/email is existed
        user_cnt = db_module.User.query.filter_by(username=username).count() +\
            db_module.User.query.filter_by(email=email).count()
        if user_cnt > 0:
            flash(f'Error - Existing user: {username} or {email}', 'error')
        else:
            subscription = 'basic'  # default:basic on registration
            user = db_module.User(username, email, password, subscription)
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('login'))
    db.session.close()
    return render_template('register.html', form=reg_form)


@application.login_manager.user_loader
def load_user(id):
    return db_module.User.query.get(int(id))


@application.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login user and redirect to upload page.

    :return: No value returned, render the login page only
    """
    login_form = LogInForm()
    if login_form.validate_on_submit():
        username = login_form.username.data
        password = login_form.password.data

        # Look for this user in the database
        user = db_module.User.query.filter_by(username=username).first()

        if not user:
            flash("Username does not exist.", 'error')

        # Login and validate the user
        if user is not None:
            if user.check_password(password):
                login_user(user)
                keep_session()
                return redirect(url_for('upload'))
            else:
                flash("Username and password doesn't match", 'error')
    return render_template('login.html', form=login_form)


@application.route('/logout')
@login_required
def logout():
    """
    Logout user and redirect to home page.
    """
    logout_user()
    return redirect(url_for('index'))


@application.route('/upload',
                   defaults={'contest_id': None}, methods=['GET', 'POST'])
@application.route('/upload/<contest_id>', methods=['GET', 'POST'])
def upload(contest_id):
    """Allow user to upload files and start making prediction after uploading.

    :return: No value returned, render the upload page and redirect user
             to the result page after the prediction is completed
    """
    file_form = UploadFileForm()
    global temp_scores

    if file_form.validate_on_submit():
        f = file_form.file_selector.data
        filename = secure_filename(f.filename)

        try:
            # Save to local folder
            file_dir_path = os.path.join(application.instance_path, 'files')
            if not os.path.isdir(file_dir_path):
                os.makedirs(file_dir_path)
            file_path = os.path.join(file_dir_path, filename)
            if not os.path.isfile(file_path):
                f.save(file_path)
        except Exception as e:
            # print(e)
            flash('Error occured uploading file, please try again.', 'error')

        # Push factors to db for data record
        try:
            factors = extract_factors(file_path)

            # If logged in, upload video and write factors to db
            if current_user.is_authenticated:
                # write factors to db
                user = db_module.User.query.filter_by(
                    username=current_user.username).first()
                user_id = user.id
                shot_factors = db_module.Shot(user_id,
                                              factors['ball_speed'],
                                              factors['launch_angle'],
                                              factors['azimuth'],
                                              factors['side_spin'],
                                              factors['back_spin'])
                shot_id = shot_factors.id
                db.session.add(shot_factors)
                db.session.commit()
        except Exception as e:
            print(e)
            flash('Error - cannot extract factors, please try again.', 'error')

        # Calculate outputs, push to db, and display on the result page
        try:
            # Log in to see the trajectory
            if current_user.is_authenticated:
                shot_id = shot_factors.id

                # # plt gif version
                # gif_name = str(current_user.username) + '_' + str(
                #     shot_id) + '.gif'

                # plotly version
                gif_name = str(current_user.username) + '_' + str(
                    shot_id) + '.html'
                carry, offline, tot_dist = predict(factors, gif_name,
                                                   is_auth=True)
                # fly_dist = np.sqrt(carry ** 2 + offline ** 2)
            else:
                gif_name = ''
                carry, offline, tot_dist = predict(factors, gif_name,
                                                   is_auth=True)
                temp_scores = [carry, offline, tot_dist,
                               factors['ball_speed'],
                               factors['launch_angle'],
                               factors['azimuth'],
                               factors['side_spin'],
                               factors['back_spin']
                               ]

            # If logged in, write data to db
            if current_user.is_authenticated:
                # Get timestamp
                utc = arrow.utcnow()
                local = utc.to('US/Pacific')
                enter_date = local.format('YYYY-MM-DD')
                upload_timestamp = local.format('YYYY-MM-DD hh:mm:ss A')

                trajectory = db_module.Trajectory(shot_id, user_id, carry,
                                                  offline, tot_dist, gif_name)
                db.session.add(trajectory)
                db.session.commit()
                db.session.close()

                # add shot to contest if contest id is provided
                if contest_id:
                    current_contest = db_module.UserContest(user_id, shot_id,
                                                            contest_id,
                                                            enter_date)
                    db.session.add(current_contest)
                    db.session.commit()
                    db.session.close()

                # Upload video and gif to S3 bucket
                username = current_user.username
                try:
                    video_key = upload_file(username, shot_id,
                                            file=f, video=True)
                    gif_key = upload_file(username, shot_id,
                                          file=gif_name, video=False)
                    video_path = file_path
                    gif_path = './app/static/trajectory/' + str(gif_name)
                    delete_tmp_file(video_path, gif_path, plotly=True)
                except Exception as e:
                    print(e)
                    pass
                    # TODO: Error handle

                # write file info to database
                file = db_module.Files(user_id=user_id,
                                       shot_id=shot_id,
                                       video_key=video_key,
                                       gif_key=gif_key,
                                       username=username,
                                       date_upload=upload_timestamp)
                db.session.add(file)
                db.session.commit()
                db.session.close()

                return redirect(
                    url_for('result', user_id=user_id, shot_id=shot_id))
            else:
                return redirect(url_for('result_anonymous'))
        except Exception as e:
            print(e)
            flash('Error - cannot predict trajectory, please try again.',
                  'error')

    return render_template('upload.html', form=file_form)


@application.route('/contests/<contest_id>')
def individual_contest(contest_id):
    contest = db_module.Contest.query.filter_by(id=contest_id).first()
    exist_user_ids = db.session.query(db_module.UserContest.user_id)\
                               .filter(db_module.UserContest.
                                       contest_id == contest_id)\
                               .all()
    exist_user_ids = list(map(lambda x: x[0], exist_user_ids))
    all_users = db.session.query(db_module.User.username,
                                 db_module.Trajectory.total_dist,
                                 db_module.Trajectory.carry,
                                 db_module.Trajectory.offline,
                                 db_module.UserContest.enter_date)\
                          .join(db_module.UserContest)\
                          .filter(db_module.UserContest.contest_id ==
                                  contest_id,
                                  db_module.UserContest.user_id ==
                                  db_module.User.id,
                                  db_module.UserContest.shot_id ==
                                  db_module.Trajectory.shot_id) \
                          .limit(20).all()
    target_tot_dist = contest.target_x
    target_offline = contest.target_y
    diffs_to_target = list(map(lambda x: math.sqrt((x[1]-target_tot_dist)**2 +
                                                   (x[3]-target_offline)**2),
                               all_users))
    users_in_contest = sorted(map(lambda x: x[0] + (x[1],),
                              [*zip(all_users, diffs_to_target)]),
                              key=lambda x: x[-1], reverse=False)

    # print(exist_user_ids)
    return render_template('individual_contest.html',
                           contest=contest,
                           users_in_contest=users_in_contest,
                           exist_user_ids=exist_user_ids)


@application.route('/profile/<user_id>')
@login_required
def profile(user_id):
    """User Profile Page : Renders profile.html with user_id."""
    user = db_module.User.query.filter_by(id=user_id).first()
    all_shots = db.session.query(db_module.User.id,
                                 db_module.Trajectory.shot_id,
                                 db_module.Trajectory.carry,
                                 db_module.Trajectory.offline,
                                 db_module.Trajectory.total_dist,
                                 db_module.UserContest.contest_id,
                                 # db_module.Contest.title,
                                 db_module.Files.date_upload) \
                          .join(db_module.UserContest) \
                          .filter(db_module.UserContest.user_id == user_id,
                                  db_module.UserContest.user_id ==
                                  db_module.User.id,
                                  db_module.UserContest.shot_id ==
                                  db_module.Trajectory.shot_id,
                                  db_module.UserContest.shot_id ==
                                  db_module.Files.shot_id)\
                          .all()

    return render_template('profile.html',
                           user=user, shots=all_shots)


@application.route('/contests')
def contests():
    """All Contests Page : Renders contests.html."""
    all_contests = db_module.Contest.query.all()
    contest_list = []
    for contest in all_contests:
        contests = [contest.id, contest.title, contest.description,
                    contest.target_x, contest.target_y,
                    contest.start_date, contest.end_date]
        current_dt = datetime.today()
        end_dt = datetime.strptime(contest.end_date, '%Y-%m-%d')
        delta = end_dt - current_dt
        contest_list.append(contests + [delta.days])

    return render_template('contests.html', contests=contest_list)


@application.route('/')
def index():
    """Index Page : Renders index.html with author name."""
    return render_template('index.html')


@application.route('/about_us')
def about_us():
    """About us Page : Renders about_us.html."""
    return render_template('about_us.html')


@application.route('/team')
def team():
    """Team Page : Renders team.html."""
    return render_template('team.html')


@application.route('/pricing')
def pricing():
    """Pricing Page : Renders pricing.html."""
    return render_template('pricing.html')


@application.route('/payment')
def payment():
    """Payment Page : Renders payment.html."""
    if current_user.is_authenticated:
        return render_template('payment.html')
    else:
        return redirect(url_for('login'))


@application.route('/thank_you')
def thank_you():
    """Thank you Page : Renders thank_you.html."""
    return render_template('thank_you.html')


@application.route('/result/<user_id>/<shot_id>')
def result(user_id: int, shot_id: int):
    """Showing the result (sores and trajectory) for log-in users."""
    traj = db_module.Trajectory.query.filter_by(user_id=user_id,
                                                shot_id=shot_id).first_or_404()
    factors = db_module.Shot.query.filter_by(user_id=user_id,
                                             id=shot_id).first_or_404()
    scores = (traj.carry, traj.offline, traj.total_dist, factors.ball_speed,
              factors.launch_angle, factors.azimuth,
              factors.side_spin, factors.back_spin)
    gif_name = traj.gif_name

    # for gif animation only
    # username = current_user.username
    # image_data = get_imagedata(username, gif_name)

    return render_template('result_login.html',
                           scores=scores, gif=gif_name)


@application.route('/result/anonymous')
def result_anonymous():
    """Showing the scores only, no trajectory gif for anonymous users."""
    return render_template('result.html', scores=temp_scores)
