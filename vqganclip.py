
import dataclasses
import typing

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

import vqgan_clip_zquantize

@dataclasses.dataclass
class VQGANCLIP:
    args: typing.Any
    opt: typing.Any
    z: typing.Any
    z_min: float
    z_max: float
    i: int = 0
    prev_losses: typing.Any = dataclasses.field(default_factory=list)
    display_freq: int = 1
    save_freq: int = None

    def epoch(self):
        '''Compute one epoch of training.
        '''
        # compute and store the loss
        self.opt.zero_grad()
        self.lossAll = self.ascend_txt()
        loss = sum(self.lossAll)
        self.prev_losses.append(loss.item())
        
        # print iteration output to screen

        
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
        out = self.synth(z)
        TF.to_pil_image(out[0].cpu()).save(fname)

    def ascend_txt(self):
        out = self.synth(self.z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

        result = []

        if args.init_weight:
            result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

        for prompt in pMs:
            result.append(prompt(iii))

        return result
    
    @staticmethod
    def synth(z):
        z_q = vqgan_clip_zquantize.vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
        return vqgan_clip_zquantize.clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)







