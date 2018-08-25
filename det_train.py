import argparse
import logging
import numpy as np
from detector import FishDetector

ap = argparse.ArgumentParser()
ap.add_argument("--img", default="train_paths.npy",
                help="Path to image .npy file")
ap.add_argument("--annot", default="train_annots.npy",
                help="Path to annotation .npy file")
ap.add_argument("--C", type=int, default=5,
                help="C params for SVM")
ap.add_argument("--n_thread", type=int, default=2,
                help="Num thread for training")


def main():
    args = ap.parse_args()
    IMG = args.img
    ANNOT = args.annot
    C = args.C
    N_THREAD = args.n_thread

    logger.info('Load image and annotation')
    paths = np.load(IMG)
    annots = np.load(ANNOT)

    detector = FishDetector()
    detector.fit(paths, annots, "fishdetector.svm", n_thread=N_THREAD, C=C)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
