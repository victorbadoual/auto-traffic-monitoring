import argparse

import cv2


def get_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    count = 0
    while success:
        cv2.imwrite("frames/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = video_capture.read()
        print('Read a new frame: ', success)
        count += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    get_frames(args.video_path)
