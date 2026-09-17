"""Regression checks for the COCO evaluation postprocessing contract."""

import inspect
import unittest

from utils import coco_evaluation


class CocoEvaluationDefaultsTest(unittest.TestCase):
    def test_uses_low_confidence_and_yolo_nms_defaults(self):
        signature = inspect.signature(coco_evaluation.evaluate_coco)

        self.assertEqual(coco_evaluation.COCO_CONF_THRESH, 0.001)
        self.assertEqual(coco_evaluation.COCO_NMS_IOU_THRESH, 0.6)
        self.assertEqual(signature.parameters["conf_thres"].default, 0.001)
        self.assertEqual(signature.parameters["nms_thresh"].default, 0.6)


if __name__ == "__main__":
    unittest.main()
