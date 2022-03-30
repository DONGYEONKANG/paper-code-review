import sys
import torch
from collections import Counter

sys.path.append("G:/내 드라이브/Github/paper-code-review/basic_object_detection")
from IoU.IoU import intersection_over_union


# 단순히 0.5를 기준으로 진행한것임.
def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format=" corners", num_classes=20
):

    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:  # [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
            if detection[1] == c:  # class_prediction == c(class)
                detections.append(detection)

        for true_box in true_boxes:  # same
            if true_box[1] == c:
                ground_truths.append(true_box)

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])


        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val) # amounnt_bboxes = {0: torch.tenser([0,0,0], 1: torch.tensor([0,0,0,0,0])}



        #class c에 해당하는 이미지별 bbox수만큼 tensor을 생성한다.
        detections.sort(key=lambda x: x[2], reverse=True) # sort: prob_score
        TP = torch.zeros(len(detections)) #
        FP = torch.zeros((len(detections))) #
        total_true_bboxes = len(ground_truths) #

        for detection_idx, detection in enumerate(detections):
            ground_truths_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truths_img) # num of target bbox(?)
            best_iou = 0

            for idx, gt in enumerate(ground_truths_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # prediction is correct!!!
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon)) # element wise
        precisions = torch.cat((torch.tensor([1]), precisions)) # 1부터 시작한다..!
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls)) # y,x

    return sum(average_precisions) / len(average_precisions)





