import pandas as pd
from step1 import *


def add_na_rows(shot_df, session_init):
    max_since_impact = shot_df.since_impact.max()
    num_rows = max_since_impact-shot_df.shape[0]+1
    impact_frame = shot_df.index.min()
    zerodf = pd.DataFrame([[-1 for j in range(shot_df.shape[1])]
                           for i in range(num_rows)], columns=shot_df.columns)
    zerodf.since_impact = [i for i in range(0, max_since_impact+1) if i not in shot_df.since_impact.values]
    zerodf.index = [i for i in range(impact_frame, impact_frame+max_since_impact+1)
                    if i not in shot_df.index.values]
    shot_df.width_max = shot_df.width_max/session_init[2]
    shot_df.leading_edge_x = shot_df.leading_edge_x/session_init[0]
    shot_df.leading_edge_y = shot_df.leading_edge_y/session_init[1]
    combined = pd.concat([shot_df, zerodf])
    return combined.sort_values('since_impact')


def make_df(frame_nos, ball_contour_features, ball_init):
    contour_df = pd.DataFrame(ball_contour_features, index=frame_nos,
                              columns=['width_80', 'width_max', 'leading_edge_x', 'leading_edge_y',
                                       'direction', 'length'])
    contour_df['since_impact'] = contour_df.index - contour_df.index.min()
    contour_df = add_na_rows(contour_df, ball_init)

    wide = pd.concat([contour_df.width_80, contour_df.width_max, contour_df.leading_edge_x,
                      contour_df.leading_edge_y, contour_df.direction, contour_df.length]).values
    col_names = []
    for col in contour_df.columns[:-1]:
        for i in range(contour_df.shape[0]):
            col_names.append(col + '_' + str(i))
    wide_df = pd.DataFrame(wide.reshape(1, -1), columns=col_names)

    return wide_df


def make_predictions(wide_df, ball_init, azimuth_regr, back_spin_regr, ball_speed_regr, launch_angle_regr):
    back_spin_x = wide_df.leading_edge_y_3.values / ball_init[2]

    azimuth = azimuth_regr.predict(wide_df.leading_edge_x_3.values.reshape(1, -1))
    back_spin = back_spin_regr.predict(back_spin_x.reshape(1, -1))
    ball_speed = ball_speed_regr.predict(wide_df.width_max_3.values.reshape(1, -1))
    launch_angle = launch_angle_regr.predict(back_spin_x.reshape(1, -1))

    return {'azimuth': azimuth[0], 'back_spin': back_spin[0],
            'ball_speed': ball_speed[0], 'launch_angle': launch_angle[0]}


def get_shot_factors(video_path, ball_init, predict_impact, classify_bbox,
                     azimuth_regr, back_spin_regr, ball_speed_regr, launch_angle_regr):
    frames = get_frames(video_path)
    frame_nos, ball_contour_features, ball_boxes = step_1(video_path, ball_init,
                                                          frames, predict_impact, classify_bbox)
    feature_df = make_df(frame_nos, ball_contour_features, ball_init)
    shot_factors = make_predictions(feature_df, ball_init, azimuth_regr, back_spin_regr,
                                    ball_speed_regr, launch_angle_regr)
    shot_factors['side_spin'] = -409

    return shot_factors

