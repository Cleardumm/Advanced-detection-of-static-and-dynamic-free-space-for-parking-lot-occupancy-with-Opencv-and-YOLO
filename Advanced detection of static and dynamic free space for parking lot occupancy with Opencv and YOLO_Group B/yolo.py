import numpy as np
import argparse
import cv2
import subprocess

import yolo_utils
import lib


#VIDEO_SOURCE = "https://media.dcaiti.tu-berlin.de/tccams/2r/axis-cgi/mjpg/video.cgi?camera=1&resolution=1280x720&rotation=0&audio=0&mirror=0&fps=0&compression=0"


def run_yolo_(VIDEO_SOURCE):
    VIDEO_SOURCE = VIDEO_SOURCE
    FLAGS = []
    minDistanceBetweenCars = 2.5  # meters
    maxIoU = 0.1
    parser = argparse.ArgumentParser()
    #add video analysis parameters
    parser.add_argument('-m', '--model-path',
                        type=str,
                        default='./yolov3-coco/',
                        help='The directory where the model weights and configuration files are.')

    parser.add_argument('-w', '--weights',
                        type=str,
                        default='./yolov3-coco/yolov3_5l_5000.weights',
                        help='Path to the file which contains the weights for YOLOv3.')

    parser.add_argument('-cfg', '--config',
                        type=str,
                        default='./yolov3-coco/yolov3_5l.cfg',
                        help='Path to the configuration file for the YOLOv3ok model.')

    parser.add_argument('-i', '--image-path',
                        type=str,
                        # default='/Users/jaehyunlee/Desktop/dcaiti cam images/center/02-1c-2-1.jpg',
                        help='The path to the image file')

    parser.add_argument('-v', '--video-path',
                        type=str,
                        help='The path to the video file')

    parser.add_argument('-vo', '--video-output-path',
                        type=str,
                        default='./output.avi',
                        help='The path of the output video file')

    parser.add_argument('-l', '--labels',
                        type=str,
                        # default='./yolov3-coco/coco-labels',
                        default='./yolov3-coco/obj.names',
                        help='Path to the file having the labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
                        type=float,
                        # default=0.5,
                        default=0.3,
                        help='The model will reject boundaries which has a probabiity less than the confidence value. '
                             'default: 0.5')

    parser.add_argument('-th', '--threshold',
                        type=float,
                        default=0.3,
                        help='The threshold to use when applying the Non-Max Suppresion')

    parser.add_argument('--download-model',
                        type=bool,
                        default=False,
                        help='Set to True, if the model weights and configurations are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
                        type=bool,
                        default=False,
                        help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()

    # Download the YOLOv3 models if needed
    if FLAGS.download_model:
        subprocess.call(['./yolov3-coco/get_model.sh'])

    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # If both image and video files are given then raise error
    if FLAGS.image_path is None and FLAGS.video_path is None:
        print('Neither path to an image or path to video provided')
        print('Starting Inference on Webcam')

    # Do inference with given image
    if FLAGS.image_path:
        # Read the image
        try:
            img = cv2.imread(FLAGS.image_path)
            height, width = img.shape[:2]
        except:
            raise Exception('Image cannot be loaded!\n\
                               Please check the path provided!')

        finally:
            img, boxes, _, _, _ = yolo_utils.infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
            carBoxes = lib.get_car_boxes(boxes)
            #yolo_utils.show_image(img)

    elif FLAGS.video_path:
        # Read the video
        try:
            vid = cv2.VideoCapture(FLAGS.video_path)
            height, width = None, None
            writer = None
        except:
            raise Exception('Video cannot be loaded!\n\
                               Please check the path provided!')

        finally:
            while True:
                grabbed, frame = vid.read()

                # Checking if the complete video is read
                if not grabbed:
                    break

                if width is None or height is None:
                    height, width = frame.shape[:2]

                frame, _, _, _, _ = yolo_utils.infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

                if writer is None:
                    # Initialize the video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                             (frame.shape[1], frame.shape[0]), True)

                writer.write(frame)

            print("[INFO] Cleaning up...")
            writer.release()
            vid.release()

    else:
        # Infer real-time on webcam
        count = 0

        vid = VIDEO_SOURCE #cv2.VideoCapture(

        #while True:
        cap, frame = vid.read()
        height, width = frame.shape[:2]
        frame_n = np.zeros_like(frame)
        #get only the middle area as input source
        frame_n[260:330, 0:1279] = frame[260:330, 0:1279]

        #if count == 0:
            #frame, boxes, confidences, classids, idxs = yolo_utils.infer_image(net, layer_names, height , width , frame_n, colors,
                                                                    #labels, FLAGS)
            #count += 1
        #else:
        #get frame size, bounding box coordinate, and final box of the objects
        frame, boxes, confidences, classids, idxs,colors,labels = yolo_utils.infer_image(net, layer_names, height , width , frame_n, colors, labels, FLAGS) #boxes, confidences, classids, idxs, infer=False)


            #count = (count + 1) % 6
        img = frame
        #carBoxes = lib.get_car_boxes(boxes) #detect the coordinate of the car in the middle
        # where are the spotted cars - which parking

        # number of parkings with discovered cars

        # all parkings from this camera view


        #cv2.imshow('webcam', frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

        #vid.release()
        #cv2.destroyAllWindows()


    return boxes, confidences, classids, idxs,colors,labels
