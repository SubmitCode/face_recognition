"""
the goal of this script is to annotate videos properly.
"""
import face_recognition
import cv2
import argparse
import os
import pickle
from tqdm import tqdm


def get_input_movie(path: str):
    return cv2.VideoCapture(path)


def get_video_writer(path: str, frame_width: int, frame_height: int):
    """
    function gives back the video
    param:
        - path: is the path of the video
    returns:
        - cv2.VideoWriter
    """
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        24,
        (frame_width, frame_height)
    )


def load_known_faces(path: str, path_save_pickle=None):
    """
    this function iterates over the know faces folder.
    This function assumse the the name of the file is the
    persons name.

    param:
        - path to the folder

    returns:
        - [names, encodings]
    """
    names = []
    encodings = []
    for file in tqdm(os.listdir(path)):
        if file.split(".")[1] == "jpg":
            path_img = os.path.join(path, file)
            image = face_recognition.load_image_file(path_img)
            encoding = face_recognition.face_encodings(image)[0]
            names.append((file.split(".")[0]).split("_")[0])
            encodings.append(encoding)

    if path_save_pickle is not None:
        with open(path_save_pickle, 'wb') as f:
            pickle.dump([names, encodings], f)

    return [names, encodings]


def process_video_loop(
        input_movie: cv2.VideoCapture,
        output_movie: cv2.VideoWriter,
        names: list,
        encodings: list):
    """
    this is the main loop to process videos.

    params:
        - input_movie: cv2.VideoCapture,
        - output_movie: cv2.VideoWriter,
        - names: list,
        - encodings: list
    """

    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0

    for _ in tqdm(range(length)):
        # Grab a single frame of video
        ret, frame = input_movie.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses)
        # to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the
        # current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(
                    encodings,
                    face_encoding,
                    tolerance=0.50)

            # If you had more than 2 faces, you could make this
            # logic a lot prettier but I kept it simple for the demo
            name = None
            if match[0]:
                name = "Lin-Manuel Miranda"
            elif match[1]:
                name = "Alex Lacamoire"

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Write the resulting image to the output video file
        print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(frame)

    # All done!
    input_movie.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process avi movies')

    parser.add_argument('--path_input_movie',
                        default='outpy.avi',
                        type=str,
                        help='path to the video you want to annotate'
                        )

    parser.add_argument('--path_output_movie',
                        default='outpy_annotated.avi',
                        type=str,
                        help='path to the video you want to annotate'
                        )

    parser.add_argument('--path_known_faces_folder',
                        default='known_faces',
                        type=str,
                        help='path to the known faces folder'
                        )

    parser.add_argument('--use_saved_encodings',
                        default=False,
                        type=str,
                        help='wether to use already saved encodings')

    parser.add_argument('--path_save_encodings',
                        default='known_encodings.pkl',
                        type=str,
                        help='path to the known faces folder'
                        )

    parser.add_argument('--encode_only',
                        default=False,
                        type=bool,
                        help='wether to do only encoding for faster loading'    )

    args = parser.parse_args()

    # create encodings
    if os.path.exists(args.path_save_encodings):
        if args.use_saved_encodings:
            with open(args.path_save_encodings, 'rb') as f:
                [names, encodings] = pickle.load(f)
        else:
            [names, encodings] = load_known_faces(
                args.path_known_faces_folder,
                args.path_save_encodings)
    else:
        [names, encodings] = load_known_faces(
                args.path_known_faces_folder,
                args.path_save_encodings)

    # stop here if we only encode
    if args.encode_only:
        quit()

    input_movie = get_input_movie(args.path_input_movie)
    frame_width = int(input_movie.get(3))
    frame_height = int(input_movie.get(4))
    output_movie = get_video_writer(
        args.path_output_movie,
        frame_width,
        frame_height
        )
    process_video_loop(
            input_movie,
            output_movie,
            names,
            encodings)
