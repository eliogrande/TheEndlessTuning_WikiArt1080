import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, fit_callback):
        
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        fit_step = 1
        for i in tqdm(range(N), desc='Generating filters'):
            fit_callback(fit_step)
            fit_step += 1
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        self.N = N
        self.p1 = p1
        self.masks = torch.from_numpy(self.masks).float().to(device)
        self.N = self.masks.shape[0]

    def forward(self, x, exp_callback):
        with torch.no_grad():
            N = self.N
            _, _, H, W = x.size()
            
            # Apply array of filters to the image
            p = []
            explain_step = self.gpu_batch
            import time
            for i in range(0, N, self.gpu_batch):
                batch_masks = self.masks[i:i+self.gpu_batch]
                batch_stack = torch.mul(batch_masks, x.data)
                exp_callback(explain_step)
                explain_step += self.gpu_batch
                p.append(self.model(batch_stack))
            
            p = torch.cat(p)
            
            # Number of classes
            CL = p.size(1)
            sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
            del p
            sal = sal.view((CL, H, W))
            sal = sal / N / self.p1

        return sal
    
    
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal

# To process in batches
# def explain_all_batch(data_loader, explainer):
#     n_batch = len(data_loader)
#     b_size = data_loader.batch_size
#     total = n_batch * b_size
#     # Get all predicted labels first
#     target = np.empty(total, 'int64')
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
#         p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
#         target[i * b_size:(i + 1) * b_size] = c
#     image_size = imgs.shape[-2:]
#
#     # Get saliency maps for all images in val loader
#     explanations = np.empty((total, *image_size))
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
#         saliency_maps = explainer(imgs.cuda())
#         explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
#             range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
#     return explanations
