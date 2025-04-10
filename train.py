# train.py
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from HiFiCAFM import Encoder, Decoder, Discriminator, MessageLoss

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

    dataloader = torch.utils.data.DataLoader(DataSet(args.dataset), batch_size=args.batch_size, shuffle=True, drop_last=True,num_workers=4)

    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    discriminator = Discriminator().to(args.device)

    encoder.apply(weight_init)
    decoder.apply(weight_init)
    discriminator.apply(weight_init)

    label_real = torch.ones(args.batch_size, 1).to(args.device)
    label_fake = torch.zeros(args.batch_size, 1).to(args.device)

    msgloss = MessageLoss()
    imgloss = nn.MSELoss()
    advloss = nn.BCEWithLogitsLoss()

    optimizer_coder = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    optimizer_discr = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    lambda1, lambda2, lambda3 = 1, 1, 0.0001
    step = 0

    print("Start training...")
    for epoch in range(args.epochs):
        for data in dataloader:
            cover = data.to(args.device)
            origin_msg = torch.randint(0, 2, (args.batch_size, args.msg_size)).float().to(args.device)

            stego = encoder(cover, origin_msg).clamp(-1, 1)
            noised = stego + torch.randn(*cover.shape).to(args.device)*0.05*2
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

            # 日志
            if step % args.log_step == 0:
                with torch.no_grad():
                    pred = (extract_msg.sigmoid() >= 0.5).float()
                    accu = (pred.eq(origin_msg).sum().float() / origin_msg.numel()).item()
                    psnr = 10 * torch.log10(4 / torch.mean((cover - stego) ** 2)).item()
                    print(f"Step {step} | Acc: {accu:.4f} | PSNR: {psnr:.2f} | MsgLoss: {msg_loss.item():.4f} | ImgLoss: {img_loss.item():.4f}")

            # 保存模型
            if step % args.save_step == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(encoder.state_dict(), 'checkpoints/HiFiCAFM-encoder.pth')
                torch.save(decoder.state_dict(), 'checkpoints/HiFiCAFM-decoder.pth')
                torch.save(discriminator.state_dict(), 'checkpoints/HiFiCAFM-discriminator.pth')

            step += 1

        if epoch == 25:
            for g in optimizer_coder.param_groups:
                g['lr'] = 0.0002
            for g in optimizer_discr.param_groups:
                g['lr'] = 0.0002

        print(f"Epoch {epoch+1}/{args.epochs} completed.")

if __name__ == '__main__':
    train()
