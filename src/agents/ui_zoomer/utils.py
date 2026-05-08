import numpy as np
from pydantic import BaseModel
from src.agents.parsers import _xy as get_xy_from_text

class UIZoomerInput(BaseModel):
    # ---- sample parameteres
    model_outputs: list
    w: int
    h: int
    # --- llm model parameter
    ignored_tokens: set
    # ---- hyperparameters of algorithm
    point_var_ration: float = 0.02 # click has 20 pixel radius for 1000 side
    confidence_threshold: float = 0.65
    min_side: int = 256
    scale: float = 2.5

class UIZoomerOutput(BaseModel):
    tokens_confidences: list[float]
    spatial_confidence: float
    confidence: float
    # output of stage-1
    voted_point: tuple[float, float]
    use_voted_point: bool
    # output of stage-2
    bbox: list[int] 
    
    
def run_ui_zoomer_flow(input: UIZoomerInput):
    w, h = input.w, input.h
    points = get_points_from(input.model_outputs, w, h)
    # === Stage 1
    tokens_confidences = get_text_confidence(input.model_outputs, input.ignored_tokens)
    point_radius = max(w,h)*input.point_var_ration
    spatial_confidence, spatial_votes = get_spatial_confidence_and_votes(points, point_radius)
    
    confidence = np.mean(tokens_confidences) + spatial_confidence
    voted_point = get_best_sample_output(points, spatial_votes, tokens_confidences)
    # === stage 2
    bbox = []
    if confidence < input.confidence_threshold:
        points = remove_outlier_points(points) # filtered_inputs [K, 2]
        bbox = apative_cropping(points, input.scale, input.min_side, input.w, input.h)

    return UIZoomerOutput(
        tokens_confidences = list(tokens_confidences),
        spatial_confidence = spatial_confidence,
        confidence = confidence,
        voted_point = (voted_point[0], voted_point[1]),
        use_voted_point = len(bbox) == 0, # no generated bbox
        bbox = bbox
    )
    
def get_points_from(outputs: list, w, h):
    """
    @input outputs: directly from vllm output
    @return points (np.ndarray) of shape N,2 wher N=number of points [x, y]
    """ 
    points = np.array([get_xy_from_text(raw_out.text) for raw_out in outputs[0].outputs])
    points[:,0] = points[:,0]*w 
    points[:,1] = points[:,1]*h 
    
    return points

def get_text_confidence(outputs: list, ignored_tokens: set):
    """Stage 1 - Step 1 - get mean of logprops
    @input outputs: directly from vllm output
    @input ignored_tokens: set of tokens ids to ignore when calculating confidences: s
        @example: for aguvis (tokenizer.encode("pyautogui.click(x=, y=)") + tokenizer.encode("pyautogui.doubleClick(x=, y=)") + [151658]) to ignore
    """
    tokens_confidences = []
    for sample_output in outputs[0].outputs:
        props = [list(i.items())[0] for i in sample_output.logprobs]
        # t = [token_details.decoded_token for token_id, token_details in props if token_id not in ignored_tokens]
        props = [token_details.logprob for token_id, token_details in props if token_id not in ignored_tokens]
        # ci = np.exp(np.mean(props))
        tokens_confidences.append(np.exp(np.mean(props)))
    # return np.mean(tokens_confidences)
    return tokens_confidences

def get_spatial_confidence_and_votes(points: np.ndarray, point_radius: float):
    """Stage 1 - Step 2,3
        - step 2: get spatial confidences = 1 when points are same
        - step 3.1: get spatial votes- gets the votes of spatial clossness for speed up
    @input points: np.ndarray [N, 2]
    @input point_radius: hyperparameter descripe the variance or width of guassian distribution from point
    @return spatial confidence: float
        in ui-zoomer            = avg[IoU(bbox_i, bbox_j)] 
        in our implementation   = avg[P(x,y|px,py; std=point_radius)] 
    @return spatial votes: np.ndarray (N,)
        in ui-zoomer            = II[IoU(bbox_i, bbox_j) > 0.5] 
        in our implementation   = II[dist(pi,pj) < point_radius] 
    """
    N = points.shape[0]
    # N x N
    diff = points[:, None, :] - points[None, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    # ignore distances of i.i    
    mask = ~np.eye(N, dtype=bool) 
    sq_dists_flatten = sq_dists[mask] # N * (N-1)
    # === Step 2 ===
    # E(exp(-1/2 dist(pipj)^2 / \var^2)
    spatial_confidence = np.mean(np.exp(-0.5 * sq_dists_flatten / (point_radius**2)))
    # === Prepare Step 3 ===
    agreement_mask = sq_dists < (point_radius**2)
    np.fill_diagonal(agreement_mask, False) # Ignore self-agreement (j != i)
    votes = np.sum(agreement_mask, axis=1) # Shape: [N,]

    return spatial_confidence, votes

def get_best_sample_output(points: np.ndarray, spatial_votes: np.ndarray, tokens_confidences: list[float]):
    """
    @input points: np.ndarray [N, 2]
    @input spatial_votes: np.ndarray (N,)
    @input tokens_confidences: list of float (N,)
    ---
    @return [x, y] point np.ndarray
    """
    best_idx = max(range(len(points)), key=lambda i: (spatial_votes[i], tokens_confidences[i]))
    return points[best_idx]

def remove_outlier_points(points: np.ndarray):
    """Stage 2 - Step 1 - remove outliers
    @input points: np.ndarray [N, 2]
    @return filtered_points: np.ndarray [K, 2]
    """
    N = points.shape[0]
    K = max(int(N * 0.75), 1)
    
    p_mean = np.mean(points, axis=0) #[2,]
    dist_from_mean = np.linalg.norm(points - p_mean, axis=1) # [N,]
    filtered_points = points[np.argsort(dist_from_mean)[:K]] # [K, 2] - closest K points to mean
    
    return filtered_points

def apative_cropping(points: np.ndarray, scale: float, min_side: int, w: int, h: int):
    """Stage 2 - Step 2 - crop
    @input points: filtered points [K, 2]
    @input scale: gamma usually 2.5-3.5
    @input min_side: m usually 512 
    @input w: width
    @input h: height
    @return bbox list of 4 integers [int(x0), int(y0), int(x1), int(y1)] -- used later in mapping
    ---
    @NOTE: x_crop_pred * width_cropped + x0 = x_crop_pred (x1 - x0) + x0
    """
    # 1 calculate std_inter
    std_inter = np.std(points, axis=0)
    # 2. calculate std_intra -- which is variance of widths and heighs of generated bboxs == in our cases = 0
    # 3. calculate s - s:side of cropped square
    min_side = min(min_side, w, h) # avoid very hard edge case
    s = max(2*scale*std_inter[0], 2*scale*std_inter[1], min_side)
    half_s = s / 2
    # 4. calculate crop bbox
    # --- 4.1 mean point
    p_mean = np.mean(points, axis=0) #[2,]
    xm, ym = p_mean[0], p_mean[1] # x_mean, y_mean
    # --- 4.2 bbox inital
    x0, y0, x1, y1 = xm-half_s, ym-half_s, xm+half_s, ym+half_s
    # --- 4.3 better handling of cropping outside image by shifting instead of clipping (ref: from ui-zoomed repo)
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= (x1 - w); x1 = w
    if y1 > h: y0 -= (y1 - h); y1 = h
    # 4.4 --- crop original image
    bbox = [int(x0), int(y0), int(x1), int(y1)]

    return bbox

    
        