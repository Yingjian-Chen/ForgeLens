import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast
from tqdm import tqdm
import pandas as pd
import random
from options.options import Options
from util import Logger, translate_duplicate

from models.network.net_stage1 import net_stage1
from models.network.net_stage2 import net_stage2


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_torch(3407)
    options = Options()
    opt = options.parse()
    log_dir = os.path.join('./check_points', opt.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    Logger(os.path.join(log_dir, 'inference_log.log'))

    input_dir = opt.input_dir

    # model loading
    if opt.eval_stage == 1:
        model = net_stage1()
    else:
        model = net_stage2(opt, train=False)
    model_load = torch.load(opt.weights)
    model.load_state_dict(model_load['model_state_dict'])
    model.cuda()
    model.eval()
    print("Model loaded successfully.")

    # preprocessing
    transform = transforms.Compose([
            transforms.Lambda(translate_duplicate),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    image_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    results = []

    # inference
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Inferencing"):
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).cuda()

            with autocast():
                pred = model(img_tensor)
                prob = torch.sigmoid(pred).item()

            label = "AI" if prob >= 0.5 else "Nature"
            results.append({
                "Image": os.path.basename(img_path),
                "Score": round(prob, 4),
                "Label": label
            })

    # save
    df = pd.DataFrame(results)
    output = os.path.join(log_dir, "inference_result.csv")
    df.to_csv(output, index=False)
    print(f"\n Inference completed. Results saved to: {output}")