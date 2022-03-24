import typing
from typing import List, Dict
import tqdm
import pathlib
from itertools import product
import dataclasses

import gifmaker


if __name__ == '__main__':

    imfp = pathlib.Path('resource_images')
    kwargs = dict(save_freq=2)

    texts = [
        ['transcendance'],
        #['peace'],
        #['love'],
        ['peace and love'],
        #['culture'],
        #['culture and algorithms'],
        #['beautiful sunset'],
        #['sunset clouds in the mountains'],
        #['mountain sunset'],
        #['green forest'],
        #['lush garden'],
        #['jungle'],
        #['jungle forest'],
        #['forest at sunset'],
        #['blue ocean'],
        #['reflection in the water'],
        #['Christian Christianity'],
        #['Star Wars'],
        #['women, fire, and dangerous things'],
    ]
    suffixes = ['', ' in the style of Studio Ghibli', ' Artstation']

    texts = [[]] + [[f'{t}{pre}' for t in ts] for ts, pre in product(texts, suffixes)]

    img_prompts = [[]] + [[fp] for fp in imfp.glob('famous_paintings/*.jpg')]

    init_imgs = list(imfp.glob('devin/ig2/*.png'))
    
    all_params = list()
    for ii, ips, txts in product(init_imgs, img_prompts, texts):
        #print(f'{ii}, {ips}, {txts}\n')
        all_params.append(gifmaker.AnimateParams(
            init_image = ii,
            img_prompts = {0:ips},
            txt_prompts = {0:txts},
            max_iter = 300,
            init_image_as_prompt = True,
            **kwargs,
        ))
    
    # count params
    print(f'running {len(all_params)} param configurations')

    #exit()

    # start the outer loop
    for params in tqdm.tqdm(all_params):
        gifmaker.make_gif(
            'ig05', 
            pathlib.Path(f'output/instagram'), 
            params,
            display_freq = None
        )


