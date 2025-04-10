# visualize.py
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from HiFiCAFM import Encoder  # 确保你已经训练好 encoder 模型

# 图像反归一化
def denormalize(tensor):
    return tensor * 0.5 + 0.5  # [-1,1] → [0,1]

# 可视化函数
def visualize_comparison(cover, stego, save_path=None):
    cover_np = denormalize(cover).cpu().numpy().transpose(1, 2, 0)
    stego_np = denormalize(stego).cpu().detach().numpy().transpose(1, 2, 0)
    diff_np = np.abs(cover_np - stego_np)

    psnr = 10 * np.log10(1.0 / np.mean((cover_np - stego_np) ** 2))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(cover_np)
    axs[0].set_title("Original (Cover)")
    axs[0].axis('off')

    axs[1].imshow(stego_np)
    axs[1].set_title(f"Stego Image\n(PSNR: {psnr:.2f} dB)")
    axs[1].axis('off')

    axs[2].imshow(diff_np * 5.0)  # 放大差异
    axs[2].set_title("Residual (|Cover - Stego|)")
    axs[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 加载图像
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)

# 主函数
@torch.no_grad()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载模型
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("checkpoints/HiFiCAFM-encoder.pth", map_location=device))
    encoder.eval()

    # 加载图像
    image_path = "mirflickr/train/000001.jpg"  # 替换为你自己的图像路径
    cover = load_image(image_path).unsqueeze(0).to(device)

    # 随机生成水印消息
    msg = torch.randint(0, 2, (1, 64)).float().to(device)

    # 嵌入
    stego = encoder(cover, msg).clamp(-1, 1)

    # 可视化
    visualize_comparison(cover[0], stego[0])

if __name__ == '__main__':
    main()
