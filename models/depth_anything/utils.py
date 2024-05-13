
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import torch

def get_depth_model(model_name, pretrained_resource, **kwargs):
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    overwrite['img_size'] = [640, 640]
    overwrite['input_height'] = 640
    overwrite['input_width'] = 640

    config = get_config(model_name, "eval", 'kitti', **overwrite)
    model = build_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model

# @time_tracker
@torch.no_grad()
def infer_depth_anything(model, images, **kwargs):
    """Inference with flip augmentation"""
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred

