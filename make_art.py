import typing
from typing import List, Dict
import tqdm
import pathlib
from itertools import product
import dataclasses

import gifmaker


if __name__ == '__main__':

    imfp = pathlib.Path('resource_images')
    
    #kwargs = dict(
    #    #init_image = imfp.joinpath('angela/angela_garden1.jpg'),
    #)
    #jhilams = imfp.joinpath('jhilam').glob('*.png')
    #stocks = imfp.joinpath('stock_photos').glob('*.png')
#
    #all_params = list()
    #for jfp, sfp in product(jhilams, stocks):
    #    all_params.append(gifmaker.AnimateParams(
    #        #size = size,
    #        init_image = sfp,
    #        img_prompts = {0: [jfp]},
    #        txt_prompts = {},
    #        **kwargs,
    #    ))


    texts = [
        'Angels Landing Artstation',
        'Angels Landing Ghibli',
        'Grand Canyon Artstation',
        'Grand Canyon Ghibli',
        'building at sunset Artstation',
        'building at sunset Deviantart',
        'building at sunset Ghibli',
        'sunset in the city Artstation',
        'sunset in the city Deviantart',
        'sunset in the city Ghibli',
    ]
    kwargs = dict(
        size = [600, 400],
        max_iter = 300,
    )
    all_params = list()
    for seed, text in product(range(5), texts):
        all_params.append(gifmaker.AnimateParams(
            #size = size,
            init_image = None,
            img_prompts = {},
            txt_prompts = {0: [text]},
            seed = seed,
            **kwargs,
        ))
    
    # count params
    print(f'running {len(all_params)} param configurations')

    # start the outer loop
    for params in tqdm.tqdm(all_params):
        gifmaker.make_gif(
            'stylized03', 
            pathlib.Path(f'output/blog01'), 
            params,
            display_freq = None
        )


