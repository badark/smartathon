import numpy as np
np.__version__
def get_iou(boxGround, boxPred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(boxGround[0], boxPred[0])
    iy1 = np.maximum(boxGround[1], boxPred[1])
    ix2 = np.minimum(boxGround[2], boxPred[2])
    iy2 = np.minimum(boxGround[3], boxPred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 , np.array(0.))
    i_width = np.maximum(ix2 - ix1 , np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = boxGround[3] - boxGround[1] 
    gt_width = boxGround[2] - boxGround[0] 
     
    # Prediction dimensions.
    pd_height = boxPred[3] - boxPred[1] 
    pd_width = boxPred[2] - boxPred[0] 
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    print("Area Ground")
    print(gt_height * gt_width)
    print("Area Predict")
    print(pd_height * pd_width)
    print("Area Union")
    print(area_of_union)
    print("Area Intersect")
    print(area_of_intersection)
    iou = area_of_intersection / area_of_union
     
    return iou

if __name__ == '__main__':
    # Pointing out a wrong IoU implementation in https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    boxTruth = [1202, 123, 1650, 868]
    boxPrediction = [1,1,1,1]

    correct = get_iou(boxTruth, boxPrediction)

# Load CSV
# Process CSV file and convert to dictionary (key by image path) and will gather bounding boxes
# Pair n boxes to k boxes, calculate centroid of each box and measure distance, closest ones to each other will be paired
# for extra, label as "bad"; for fewer label 'recall'


    print('Correct solution - also analytical: {}\n'
          .format(correct))