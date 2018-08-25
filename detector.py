import dlib
import cv2
import sys
from tqdm import tqdm


class FishDetector:
    def __init__(self, filepath=None):
        self.__option = dlib.simple_object_detector_training_options()
        if filepath:
            self.__detector = dlib.simple_object_detector(filepath)

    def __prep_image(self, paths):
        images = [cv2.imread("_LABELED-FISHES-IN-THE-WILD/Training_and_validation/Positive_fish/" + path)
                  for path in tqdm(paths[:300])]
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in tqdm(images)]
        return images

    def __prep_annotations(self, annotations):
        annots = [[dlib.rectangle(left=left, top=top, right=right, bottom=bottom)]
                  for (left, top, right, bottom) in tqdm(annotations[:300])]
        return annots

    def fit(self,
            paths,
            annotations,
            target_path,
            vis=True,
            n_thread=1,
            C=5,
            verbose=True,
            add_left_right_image_flips=True):
        self.__option.num_threads = n_thread
        self.__option.C = C
        self.__option.be_verbose = verbose
        self.__option.add_left_right_image_flips = add_left_right_image_flips

        images = self.__prep_image(paths)
        print('[IMAGE SIZE] {:.2f} MB'.format(sys.getsizeof(images) / 1024*1024))
        annotations = self.__prep_annotations(annotations)
        print('[ANNOT SIZE] {:.2f} MB'.format(sys.getsizeof(annotations) / 1024*1024))

        self.__detector = dlib.train_simple_object_detector(images, annotations, self.__option)

        if vis:
            win = dlib.image_window()
            win.set_image(self.__detector)
            dlib.hit_enter_to_continue()

        self.__detector.save(target_path)

    def predict(self, image):
        rects = self.__detector(image)
        res = [(left, top, right, bottom) for (left, top, right, bottom) in rects]
        return res
