import time
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
    
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  
  A = anc.size()[0]
  B, H_prime, W_prime, _, = grid.size()

  anchors = torch.zeros((B, A, H_prime, W_prime, 4), device=grid.device, dtype=grid.dtype) # (B, A, H', W', 4)
  assert not anchors.requires_grad # FIXME:

  grid = grid.unsqueeze(1).repeat(1, A, 1, 1, 1) # (B, A, H', W', 2)

  anc = anc.unsqueeze(0).unsqueeze(2).unsqueeze(2).repeat(B, 1, H_prime, W_prime, 1) # (B, A, H', W', 2)

  anchors[:, :, :, :, :2] = grid - anc / 2
  anchors[:, :, :, :, 2:] = grid + anc / 2
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  
  temp = torch.zeros_like(anchors)

  # Anchor entry = (x_tl, y_tl, x_br, y_br)
  
  # Transform from corner into (x, y, w, h) representation
  
  # w = x_br - x_tl and h = y_br - y_tl
  temp[:, :, :, :, 2:] = anchors[:, :, :, :, 2:] - anchors[:, :, :, :, :2]
  # x_c = x_tl + 0.5 * w and y_c = y_tl + 0.5 * y
  temp[:, :, :, :, :2] = anchors[:, :, :, :, :2] + 0.5 * temp[:, :, :, :, 2:]
  
  # Apply the transformation
  
  if method == 'YOLO':
    # x_proposal = x + t_x and y_proposal = y + t_y
    temp[:, :, :, :, :2] += offsets[:, :, :, :, :2]
  else:
    assert method == 'FasterRCNN'
    # x_proposal = x + t_x * w and y_proposal = y + t_y * h
    # FIXME: Split w_h and x_y up ==> autograd struggles with this!
    temp[:, :, :, :, :2] += offsets[:, :, :, :, :2] * temp[:, :, :, :, 2:].clone()

  # w_proposal = w * exp(t_w) and h_proposal = h * exp(t_h)
  temp[:, :, :, :, 2:] *= torch.exp(offsets[:, :, :, :, 2:])

  # Transform back into corner coordinates
  
  proposals = torch.zeros_like(temp)
  # x_tl = x - w / 2 and y_tl = y - h / 2
  proposals[:, :, :, :, :2] = temp[:, :, :, :, :2] - temp[:, :, :, :, 2:] / 2
  # x_br = x + w / 2 and y_br = y + h / 2
  proposals[:, :, :, :, 2:] = temp[:, :, :, :, :2] + temp[:, :, :, :, 2:] / 2
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return proposals


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # However, you need to make sure to compute the IoU correctly (it should be  #
  # 0 in those cases).                                                         # 
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where                          #
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################

  B, A, H_prime, W_prime, _ = proposals.size()
  N = bboxes.size()[1]

  # Proposal = (x_tl, y_tl, x_br, y_br)
  area_of_proposals = torch.prod(proposals[:, :, :, :, 2:] - proposals[:, :, :, :, :2], dim=-1) # (B, A, H', W')
  area_of_proposals = area_of_proposals.reshape(B, -1).unsqueeze(-1) # (B, A*H'*W', 1)

  # Bbox = (x_tl, y_tl, x_br, y_br, class)
  area_of_bboxes = torch.prod(bboxes[:, :, 2:4] - bboxes[:, :, :2], dim=-1) # (B, N)
  area_of_bboxes = area_of_bboxes.unsqueeze(1) # (B, 1, N)

  proposals_ = proposals.reshape(B, A*H_prime*W_prime, -1).unsqueeze(-2) # (B, A*H'*W', 1, 4)
  bboxes_ = bboxes.unsqueeze(1) # (B, 1, N, 5)
  
  inter_tl = torch.max(proposals_[:, :, :, :2], bboxes_[:, :, :, :2]) # (B, A*H'*W', N, 2)
  inter_br = torch.min(proposals_[:, :, :, 2:], bboxes_[:, :, :, 2:4]) # (B, A*H'*W', N, 2)

  area_of_intersection = torch.prod(inter_br - inter_tl, dim=-1) * (inter_tl <= inter_br).all(dim=-1) # (B, A*H'*W', N)

  area_of_union = area_of_bboxes + area_of_proposals - area_of_intersection # (B, A*H'*W', N)

  iou_mat = area_of_intersection / area_of_union # (B, A*H'*W', N)

  assert iou_mat.size() == (B, A*H_prime*W_prime, N)
  
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    
    self.pred_layer = nn.Sequential(
      nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1), 
      nn.Dropout(p=drop_ratio), 
      nn.LeakyReLU(), 
      nn.Conv2d(hidden_dim, 5 * self.num_anchors + self.num_classes, kernel_size=1, stride=1)
    )

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      of all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    
    B, _, H, W = features.size()
    
    assert (pos_anchor_idx != None and neg_anchor_idx != None) or (pos_anchor_idx == None and neg_anchor_idx == None)
    
    temp = self.pred_layer(features) # (B, 5A+C, H, W)
    
    anchor_data = temp[:, :5*self.num_anchors, :, :].reshape(B, -1, 5, H, W) # (B, A, 5, H, W)
    
    # Squash the conf scores s.t. they are between 0 and 1
    conf_scores = torch.sigmoid(anchor_data[:, :, :1, :, :]) # (B, A, 1, H, W)
    
    offsets = anchor_data[:, :, 1:, :, :] # (B, A, 4, H, W)
    # The first two elements of the offsets should be between -0.5 and 0.5
    offsets[:, :, :2, :, :] = torch.sigmoid(offsets[:, :, :2, :, :]) - 0.5

    class_scores = temp[:, 5*self.num_anchors:, :, :] # (B, C, H, W)

    if pos_anchor_idx != None and neg_anchor_idx != None:
      # Training pass
      conf_scores = torch.cat((self._extract_anchor_data(conf_scores, pos_anchor_idx), self._extract_anchor_data(conf_scores, neg_anchor_idx)), dim=0) # (2*M, 1)
      offsets = self._extract_anchor_data(offsets, pos_anchor_idx) # (M, 4)
      class_scores = self._extract_class_scores(class_scores, pos_anchor_idx) # (M, C)
    else:
      conf_scores = conf_scores.squeeze(2) # (B, A, H, W)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4  (N, 4)
  - scores: scores for each one of the boxes, of shape N (N, )
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)
 
  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################

  device = boxes.device

  scores = scores.clone()
  
  keep_indices = []
  remaining_indices = [i for i in range(scores.size()[0])]

  while len(remaining_indices) > 0 and (topk is None or len(keep_indices) != topk):        
    # Select the highest-scoring box
    highest_scoring_idx = torch.argmax(scores).item()
    keep_indices.append(highest_scoring_idx)
    scores[highest_scoring_idx] = -float('inf')

    # Compute the IoU with the remaining boxes
    remaining_indices = list(filter(lambda idx: idx != highest_scoring_idx, remaining_indices))
    
    proposal = boxes[highest_scoring_idx] # (4,)
    proposal = proposal.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1, 1, 1, 1, 4)

    bboxes = boxes[remaining_indices] # (R, 4)
    bboxes = bboxes.unsqueeze(0) # (1, R, 4)

    iou_matrix = IoU(proposal, bboxes)
    iou_matrix = iou_matrix.squeeze(0).squeeze(0)

    # filter out boxes which have an IoU > threshold with the current highest-scoring box
    idx_filter = set()
    for i, idx in enumerate(remaining_indices):
      if iou_matrix[i] > iou_threshold:
        idx_filter.add(idx)
        scores[idx] = -float('inf')
    
    remaining_indices = list(filter(lambda idx: idx not in idx_filter, remaining_indices))

  keep = torch.tensor(keep_indices, device=device, dtype=torch.long)

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep


def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss (scalar)
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss (scalar)
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss


class SingleStageDetector(nn.Module):
  def __init__(self, use_cuda=True):
    super().__init__()

    self.anchor_list = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    if use_cuda:
      self.anchor_list = self.anchor_list.cuda()
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, reg_loss by       #
    #    BboxRegression, and cls_loss by ObjectClassification.                   #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    
    # Extract the features from the images
    features = self.feat_extractor(images) # (B, 1280, H', W')

    # Grid and anchor generation
    B = features.size()[0]
    
    grid = GenerateGrid(B, device=images.device) # (B, H', W', 2)
    anchors = GenerateAnchor(self.anchor_list, grid) # (B, A, H', W', 4)

    # Compute IoU between anchors and GT boxes 
    iou_mat = IoU(anchors, bboxes) # (B, A*H'*W', N) 

    # Determine activated/negative anchors and GT_conf_scores, GT_offsets, GT_class
    activated_anc_ind, negative_anc_id, GT_conf_scores, GT_offsets, GT_class, _, _ =  \
        ReferenceOnActivatedAnchors(anchors, bboxes, grid, iou_mat, neg_thresh=0.2, method='YOLO')

    # Compute conf_scores, offsets, and class_prob using the prediction network
    conf_scores, offsets, class_scores = \
        self.pred_network(features, pos_anchor_idx=activated_anc_ind, neg_anchor_idx=negative_anc_id)

    # Compute the total loss
    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss = BboxRegression(offsets, GT_offsets)
    anc_per_image = iou_mat.size()[1]
    cls_loss = ObjectClassification(class_scores, GT_class, B, anc_per_image, activated_anc_ind)

    total_loss = w_conf * conf_loss + \
                 w_reg * reg_loss + \
                 w_cls * cls_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Kept proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    
    # Extract the features from the images
    features = self.feat_extractor(images) # (B, 1280, H', W')

    # Grid and anchor generation
    B = features.size()[0]
    
    grid = GenerateGrid(B, device=images.device) # (B, H', W', 2)
    anchors = GenerateAnchor(self.anchor_list, grid) # (B, A, H', W', 4)

    A = anchors.size()[1]

    # Compute conf_scores, offsets, and class_prob using the prediction network
    conf_scores, offsets, class_scores = self.pred_network(features) # (B, A, H', W'), (B, A, 4, H', W'), (B, C, H', W')

    # Generate proposals using the anchors and the predicted offsets
    proposals = GenerateProposal(anchors, offsets.permute(0, 1, 3, 4, 2)) # (B, A, H', W', 4)

    C = class_scores.size()[1]

    # N := H'*W'*A
    proposals = proposals.reshape(B, -1, 4) # (B, N, 4)
    conf_scores = conf_scores.reshape(B, -1) # (B, N)
    class_scores = class_scores.unsqueeze(2).repeat(1, 1, A, 1, 1).reshape(B, C, -1).permute(0, 2, 1) # (B, N, C)

    # Process each item in the batch
    for sample_idx in range(B):
      sample_proposals = proposals[sample_idx] # (N, 4)
      sample_conf_scores = conf_scores[sample_idx] # (N)
      sample_class_scores = class_scores[sample_idx] # (N, C)

      # Apply the threshold `thresh` on the conf_score
      mask = torch.nonzero(sample_conf_scores >= thresh).view(-1)

      if len(mask) == 0:
        final_proposals.append(torch.empty((0, 4), device=images.device))
        final_conf_scores.append(torch.empty((0, 1), device=images.device))
        final_class.append(torch.empty((0, 1), device=images.device))
        continue

      sample_proposals = sample_proposals[mask]
      sample_conf_scores = sample_conf_scores[mask]
      sample_class_scores = sample_class_scores[mask]

      # Apply NMS (torchvision.ops.nms) on the filtered proposals
      kept_idxs = torchvision.ops.nms(sample_proposals, sample_conf_scores, nms_thresh)

      final_sample_proposals = sample_proposals[kept_idxs]
      final_sample_conf_scores = sample_conf_scores[kept_idxs]
      final_sample_class_scores = sample_class_scores[kept_idxs]

      # Determin the class index
      final_sample_class = final_sample_class_scores.argmax(dim=-1)

      # Append to the final output
      final_proposals.append(final_sample_proposals)
      final_conf_scores.append(final_sample_conf_scores.unsqueeze(1))
      final_class.append(final_sample_class.unsqueeze(1))
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_scores, final_class
