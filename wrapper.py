import sys
sys.path.append("./yolov5-master")

import detect

opt = detect.parse_opt()
detect.main(opt)
