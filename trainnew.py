import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import csv
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import save_image
from HiFiCAFM import Encoder, Decoder, Discriminator, MessageLoss
from skimage.metrics import structural_similarity as compare_ssim
import lpips

class DataSet(Dataset):
    def __init__(self, image_folder):
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            m.bias.data.zero_()

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def train():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'mirflickr/train/'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.epochs = 40
    args.lr = 0.0005
    args.batch_size = 16
    args.msg_size = 64
    args.log_step = 100
    args.save_step = 5000

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("visuals", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 初始化模型
    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    discriminator = Discriminator().to(args.device)

    optimizer_coder = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    optimizer_discr = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    start_epoch = 0
    step = 0

    

    # 自动加载 latest.pth
    latest_path = 'checkpoints/latest.pth'
    if os.path.exists(latest_path):
        print("Loading checkpoint from latest.pth...")
        checkpoint = torch.load(latest_path, map_location=args.device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_coder.load_state_dict(checkpoint['optimizer_coder'])
        optimizer_discr.load_state_dict(checkpoint['optimizer_discr'])
        start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        print(f"Resumed from epoch {start_epoch}, step {step}")
    else:
        encoder.apply(weight_init)
        decoder.apply(weight_init)
        discriminator.apply(weight_init)

    dataloader = torch.utils.data.DataLoader(DataSet(args.dataset), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    label_real = torch.ones(args.batch_size, 1).to(args.device)
    label_fake = torch.zeros(args.batch_size, 1).to(args.device)

    msgloss = MessageLoss()
    imgloss = nn.MSELoss()
    advloss = nn.BCEWithLogitsLoss()

    lpips_fn = lpips.LPIPS(net='alex').to(args.device)
    lpips_fn.eval()
    
    lambda1, lambda2, lambda3 = 1, 1, 0.0001

    # 初始化 CSV 日志文件
    log_path = 'logs/train_log.csv'
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Epoch', 'Acc', 'PSNR', 'SSIM', 'LPIPS', 'BER', 'MsgLoss', 'ImgLoss'])

    print("Start training...")
    for epoch in range(start_epoch, args.epochs):
        for data in dataloader:
            cover = data.to(args.device)
            origin_msg = torch.randint(0, 2, (args.batch_size, args.msg_size)).float().to(args.device)

            stego = encoder(cover, origin_msg).clamp(-1, 1)
            noised = stego + torch.randn_like(stego) * 0.05 * 2
            extract_msg = decoder(noised)

            # 训练判别器
            optimizer_discr.zero_grad()
            d_real = discriminator(cover).mean(dim=1, keepdim=True)
            d_fake = discriminator(stego.detach()).mean(dim=1, keepdim=True)
            d_loss = advloss(d_real, label_real) + advloss(d_fake, label_fake)
            d_loss.backward()
            optimizer_discr.step()

            # 训练编码器和解码器
            msg_loss = msgloss(extract_msg, origin_msg)
            img_loss = imgloss(stego, cover)
            adv_loss = advloss(discriminator(stego).mean(dim=1, keepdim=True), label_real)

            lambda1_decay = 10 ** (np.clip((step - 1000) / (10000 - 1000), 0, 1) * 3)
            loss = lambda1 / lambda1_decay * msg_loss + lambda2 * img_loss + lambda3 * adv_loss

            optimizer_coder.zero_grad()
            loss.backward()
            optimizer_coder.step()

            # 日志输出 & CSV 记录
            if step % args.log_step == 0:
                with torch.no_grad():
                    # 消息准确率
                    pred = (extract_msg.sigmoid() >= 0.5).float()
                    accu = (pred.eq(origin_msg).sum().float() / origin_msg.numel()).item()

                    # PSNR
                    psnr = 10 * torch.log10(4 / torch.mean((cover - stego) ** 2)).item()

                    # SSIM（只计算第一个样本）
                    c_img = denormalize(cover[0]).permute(1, 2, 0).cpu().numpy()  # HWC
                    s_img = denormalize(stego[0]).permute(1, 2, 0).cpu().numpy()
                    ssim_val = compare_ssim(c_img, s_img, channel_axis=-1, data_range=1.0)

                    # LPIPS（整 batch 平均）
                    lpips_val = lpips_fn(cover, stego).mean().item()

                    # BER（比特错误率）
                    total_bits = origin_msg.numel()
                    bit_errors = (pred != origin_msg).sum().item()
                    ber = bit_errors / total_bits

                    # 打印全部指标
                    print(f"Step {step} | Epoch {epoch+1} | Acc: {accu:.4f} | PSNR: {psnr:.2f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val:.4f} | BER: {ber:.4f} | MsgLoss: {msg_loss.item():.4f} | ImgLoss: {img_loss.item():.4f}")

                    # 写入 CSV（可选：你也可以更新 CSV 写入列）
                    with open(log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([step, epoch+1, accu, psnr, ssim_val, lpips_val, ber, msg_loss.item(), img_loss.item()])

                # 保存模型（定期）
            if step % args.save_step == 0:
                torch.save(encoder.state_dict(), 'checkpoints/HiFiCAFM-encoder.pth')
                torch.save(decoder.state_dict(), 'checkpoints/HiFiCAFM-decoder.pth')
                torch.save(discriminator.state_dict(), 'checkpoints/HiFiCAFM-discriminator.pth')

            step += 1

        # 每轮保存模型
        torch.save(encoder.state_dict(), f'checkpoints/encoder_epoch{epoch+1:02d}.pth')
        torch.save(decoder.state_dict(), f'checkpoints/decoder_epoch{epoch+1:02d}.pth')
        torch.save(discriminator.state_dict(), f'checkpoints/discr_epoch{epoch+1:02d}.pth')

        # 保存 latest.pth
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_coder': optimizer_coder.state_dict(),
            'optimizer_discr': optimizer_discr.state_dict(),
            'epoch': epoch,
            'step': step
        }, 'checkpoints/latest.pth')

        # 保存可视化图像（仅保存第一个样本）
        # 保存可视化图像（仅保存第一个样本）
        with torch.no_grad():
            c = denormalize(cover[0].detach().cpu())
            s = denormalize(stego[0].detach().cpu())
            r = (s - c + 0.5).clamp(0, 1)  # residual 可视化

            save_image(c, f'visuals/epoch{epoch+1:02d}_cover.png')
            save_image(s, f'visuals/epoch{epoch+1:02d}_stego.png')
            save_image(r, f'visuals/epoch{epoch+1:02d}_residual.png')

            # === 保存水印可视化 ===
            def msg_to_image(msg_tensor):
                """
                将 64-bit 消息（1D）转换为 8x8 黑白图像
                """
                msg_img = msg_tensor[:64].view(8, 8).cpu().numpy()  # shape: (8, 8)
                msg_img = (msg_img * 255).astype(np.uint8)  # 0 or 255
                return Image.fromarray(msg_img, mode='L')  # 'L' = 灰度图

            # 原始水印（0 or 1）
            origin_img = msg_to_image(origin_msg[0])
            origin_img.save(f'visuals/epoch{epoch+1:02d}_origin_msg.png')

            # 提取水印（经过 sigmoid 和二值化）
            pred_msg = (extract_msg[0].sigmoid() >= 0.5).float()
            extract_img = msg_to_image(pred_msg)
            extract_img.save(f'visuals/epoch{epoch+1:02d}_extract_msg.png')


        # 学习率调整
        if epoch == 25:
            for g in optimizer_coder.param_groups:
                g['lr'] = 0.0002
            for g in optimizer_discr.param_groups:
                g['lr'] = 0.0002

        print(f"Epoch {epoch+1}/{args.epochs} completed.")

if __name__ == '__main__':
    train()
