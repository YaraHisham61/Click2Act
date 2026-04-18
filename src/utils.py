import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
from .constants import C2A_PALETTE


def to_numpy(x):
    """ convert (list|torch|numpy) to numpy float array """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)

# ------------ co-ordinates -----------
def get_xywh(input):
    """from [x1,y1][x2, y2] to [x1, y1, w, h] """
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    """from [x1,y1][x2, y2] to [x1, y1, x2, y2] """
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp


def int_box_area(box, w, h):
    """ return area using normalized x,y,x,y """
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area


def visualize_boxes(image_np, boxes, confidences=None, labels=None,
                    conf_threshold=0.0, high_conf=0.9, mid_conf=0.5,
                    title="Detections", figsize=(18, 10), ax=None):
    """Visualize xyxy bounding boxes with optional confidence scores on an image.

    Args:
        image_np      : numpy array (H, W, 3)
        boxes         : xyxy boxes — Tensor, ndarray, or list of shape (N, 4).
                        Auto-detected: if all values <= 1.0 the coords are treated
                        as normalised and scaled by (w, h) of the image.
        confidences   : per-box scores — Tensor, ndarray, or list of shape (N,).
                        Pass None to draw all boxes without score colouring.
        labels        : per-box text labels — list/array of shape (N,), or None.
        conf_threshold: discard boxes whose confidence is below this value.
        high_conf     : lower bound for the "high confidence" colour (success green).
        mid_conf      : lower bound for the "mid confidence" colour (amber);
                        below this threshold -> warning red.
        title         : axes title string.
        figsize       : matplotlib figure size tuple.
        ax            : existing Axes to draw on, or None for a standalone figure.
    """
    # --- normalise inputs to float32 numpy arrays ---
    boxes_np = to_numpy(boxes)
    confs_np = to_numpy(confidences)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.patch.set_facecolor(C2A_PALETTE["bg"])

    ax.set_facecolor(C2A_PALETTE["bg"])
    ax.imshow(image_np)

    if boxes_np is None or boxes_np.size == 0:
        ax.set_title(f"{title} (no boxes)", color=C2A_PALETTE["text"], fontsize=11)
        ax.axis("off")
        if standalone:
            plt.tight_layout()
            plt.show()
        return

    boxes_np = boxes_np.reshape(-1, 4)
    h_img, w_img = image_np.shape[:2]

    if boxes_np.max() <= 1.0:
        boxes_np = boxes_np * np.array([w_img, h_img, w_img, h_img], dtype=np.float32)

    n = len(boxes_np)
    if confs_np is None:
        confs_np = np.ones(n, dtype=np.float32)

    for i, (box, conf) in enumerate(zip(boxes_np, confs_np)):
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        if conf >= high_conf:
            color = C2A_PALETTE["success"]
        elif conf >= mid_conf:
            color = C2A_PALETTE["tertiary"]
        else:
            color = C2A_PALETTE["warning"]

        ax.add_patch(Rectangle((x1, y1), bw, bh,
                                linewidth=1.2, edgecolor=color,
                                facecolor=color + "33"))

        label_text = labels[i] if labels is not None else ""
        if label_text:
            ax.text(x1, y1 - 3, label_text,
                    fontsize=5.5, color=C2A_PALETTE["text"],
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                            ec=color, lw=0.8, alpha=0.85))

    ax.set_title(title, color=C2A_PALETTE["text"], fontsize=11)
    ax.axis("off")

    legend_elements = [
        Line2D([0], [0], color=C2A_PALETTE["success"],  lw=2, label=f"conf >= {high_conf:.2f}"),
        Line2D([0], [0], color=C2A_PALETTE["tertiary"], lw=2,
                label=f"{mid_conf:.2f} <= conf < {high_conf:.2f}"),
        Line2D([0], [0], color=C2A_PALETTE["warning"],  lw=2, label=f"conf < {mid_conf:.2f}"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
            fontsize=7, framealpha=0.85,
            labelcolor=C2A_PALETTE["text"])

    if standalone:
        plt.tight_layout()
        plt.show()


def draw_click(image, x: int, y: int, radius: int = 20, opacity: float = 0.6,
               color: str = None, figsize: tuple = (12, 7), ax=None):
    """Overlay a semi-transparent circle on an image to mark a click point.

    Args:
        image   : PIL.Image or numpy array (H, W, 3).
        x, y    : pixel coordinates of the circle centre.
        radius  : circle radius in pixels.
        opacity : alpha in [0, 1] — 0 = invisible, 1 = fully opaque.
        color   : hex colour string (default: C2A_PALETTE["warning"]).
        figsize : matplotlib figure size (used only when ax is None).
        ax      : existing Axes to draw on, or None for a standalone figure.
    """
    if color is None:
        color = C2A_PALETTE["warning"]

    # --- normalise to PIL RGB ---
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    # --- parse hex colour to RGB tuple ---
    hex_color = color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # --- draw circle on a transparent overlay ---
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = int(opacity * 255)
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=(*rgb, alpha), outline=(*rgb, 255), width=2)

    # --- composite onto the original ---
    result = Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")
    result_np = np.array(result)

    # --- render ---
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.patch.set_facecolor(C2A_PALETTE["bg"])

    ax.set_facecolor(C2A_PALETTE["bg"])
    ax.imshow(result_np)
    ax.plot(x, y, "+", color=C2A_PALETTE["text"], markersize=8, markeredgewidth=1.2)
    ax.set_title(f"click @ ({x}, {y})", color=C2A_PALETTE["text"], fontsize=11)
    ax.axis("off")

    if standalone:
        plt.tight_layout()
        plt.show()

    return result_np
