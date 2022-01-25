
import dataclasses
import typing

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

import vqgan_clip_zquantize

@dataclasses.dataclass
class VQGANCLIP:
    args: typing.Any

    prompts: list
    size: list
    
    image_prompts: list = dataclasses.field(default_factory=list)
    noise_prompt_seeds: list = dataclasses.field(default_factory=list)
    noise_prompt_weights: list = dataclasses.field(default_factory=list)
    
    init_image: str = None
    
    # training params
    init_weight: float = 0.
    step_size: float = 0.05
    cutn: int = 64
    cut_pow: float = 1.
    seed: int = 0

    # model data paths
    clip_model: str = 'ViT-B/32'
    vqgan_config: str = 'vqgan_clip_zquantize/vqgan_imagenet_f16_1024.yaml'
    vqgan_checkpoint: str = 'vqgan_clip_zquantize/vqgan_imagenet_f16_1024.ckpt'

    #for later use
    i: int = 0
    prev_losses: typing.Any = dataclasses.field(default_factory=list)

    def __post_init__(self):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        self.model = vqgan_clip_zquantize.load_vqgan_model(self.vqgan_config, self.vqgan_checkpoint).to(device)
        self.perceptor = vqgan_clip_zquantize.clip.load(self.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

        cut_size = self.perceptor.visual.input_resolution
        e_dim = self.model.quantize.e_dim
        f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = vqgan_clip_zquantize.MakeCutouts(cut_size, self.cutn, cut_pow=self.cut_pow)
        n_toks = self.model.quantize.n_e
        toksX, toksY = self.size[0] // f, self.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if self.seed is not None:
            torch.manual_seed(self.seed)

        if self.init_image:
            pil_image = Image.open(vqgan_clip_zquantize.fetch(self.init_image)).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            self.z, *_ = self.model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = optim.Adam([self.z], lr=self.step_size)

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        self.prompts = []

        # add text prompts
        for prompt in self.prompts:
            txt, weight, stop = vqgan_clip_zquantize.parse_prompt(prompt)
            embed = self.perceptor.encode_text(vqgan_clip_zquantize.clip.tokenize(txt).to(device)).float()
            self.prompts.append(vqgan_clip_zquantize.Prompt(embed, weight, stop).to(device))

        # add image prompts
        for prompt in self.image_prompts:
            path, weight, stop = vqgan_clip_zquantize.parse_prompt(prompt)
            img = vqgan_clip_zquantize.resize_image(Image.open(vqgan_clip_zquantize.fetch(path)).convert('RGB'), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = self.perceptor.encode_image(normalize(batch)).float()
            self.prompts.append(vqgan_clip_zquantize.Prompt(embed, weight, stop).to(device))

        # add noise prompts
        for seed, weight in zip(self.noise_prompt_seeds, self.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            self.prompts.append(vqgan_clip_zquantize.Prompt(embed, weight).to(device))
        

    def epoch(self):
        '''Compute one epoch of training.
        '''
        # compute and store the loss
        self.opt.zero_grad()
        self.lossAll = self.ascend_txt()
        loss = sum(self.lossAll)
        self.prev_losses.append(loss.item())
        
        # backpropogate
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

        # increment epoch counter
        self.i += 1
    
    def is_converged(self, thresh=0.01, quit_after=10):
        '''Check if loss is less than thresh for quit_after iterations.
        '''
        ls = self.prev_losses
        if len(self.prev_losses) > 5:
            if len([i for i in range(len(ls)-1) if (ls[i+1]-ls[i])>thresh]) > quit_after:
                #if self.prev_losses[-1] - self.prev_losses[-2] > 0.01:
                #self.save_current_image(final_results_folder)
                return True
        False

    @torch.no_grad()
    def save_current_image(self, fname):
        out = self.synth(self.z)
        TF.to_pil_image(out[0].cpu()).save(fname)

    def ascend_txt(self):
        out = self.synth(self.z)
        iii = self.perceptor.encode_image(normalize(self.make_cutouts(out))).float()

        result = []

        if self.init_weight:
            result.append(F.mse_loss(self.z, self.z_orig) * self.init_weight / 2)

        for prompt in self.prompts:
            result.append(prompt(iii))

        return result
    
    def synth(self, z):
        z_q = vqgan_clip_zquantize.vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return vqgan_clip_zquantize.clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)







