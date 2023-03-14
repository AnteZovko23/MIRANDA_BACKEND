import cv2, dlib, argparse
from Face_Recognition_Library import Face_Recognition_Library as frl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align faces in image')
    parser.add_argument('input', type=str, help='')
    parser.add_argument('output', type=str, help='')
    parser.add_argument('--scale', metavar='S', type=int, default=1, help='an integer for the accumulator')
    args = parser.parse_args()

    lib = frl()

    input_image = args.input
    output_image = args.output
    scale = args.scale

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    x1, y1 = 384, 185
    x2, y2 = 384, 618
    x3, y3 = 622, 185
    x4, y4 = 622, 618
    
    x, y = min(x1, x2, x3, x4), min(y1, y2, y3, y4)
    w,h = max(x1, x2, x3, x4) - x, max(y1, y2, y3, y4) - y
    
    x, y, w, h = [int(i) for i in [x, y, w, h]]
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # quit(0)
    height, width = img.shape[:2]
    s_height, s_width = height // scale, width // scale
    img = cv2.resize(img, (s_width, s_height))
    # Resize x, y, w, h
    x, y, w, h = [int(i / scale) for i in [x, y, w, h]]
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # dets = detector(img, 1)
    # print(type(dets))
    dets = dlib.rectangles([dlib.rectangle(x, y, x + w, y + h)])
    # print(type(dets))
    # quit()
    # landmarks = predictor(img, dlib.rectangle(x, y, x + w, y + h))
    # quit(0)
    for i, det in enumerate(dets):
        shape = predictor(img, det)
        left_eye = lib.extract_left_eye_center(shape)
        right_eye = lib.extract_right_eye_center(shape)

        M = lib.get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

        cropped = lib.crop_image(rotated, det)

        if output_image.endswith('.jpg'):
            output_image_path = output_image.replace('.jpg', '_%i.jpg' % i)
        elif output_image.endswith('.png'):
            output_image_path = output_image.replace('.png', '_%i.jpg' % i)
        else:
            output_image_path = output_image + ('_%i.jpg' % i)
        cv2.imwrite(output_image_path, cropped)