import typing
from typing import List, Dict
import tqdm
import pathlib
from itertools import product
import dataclasses

import gifmaker


if __name__ == '__main__':

    imfp = pathlib.Path('resource_images')

    all_params = [
        gifmaker.AnimateParams(
            size = [600, 400],
            init_image = imfp.joinpath('stock_photos/nyc.png'),
            img_prompts = {0: [imfp.joinpath('jhilam/jhilam01.png')]},
            txt_prompts = {},
        ),
        gifmaker.AnimateParams(
            size = [600, 400],
            init_image = imfp.joinpath('stock_photos/nyc.png'),
            img_prompts = {0: [imfp.joinpath('jhilam/jhilam02_cropped.png')]},
            txt_prompts = {},
        ),
        gifmaker.AnimateParams(
            size = [600, 400],
            init_image = imfp.joinpath('stock_photos/nyc.png'),
            img_prompts = {0: [imfp.joinpath('jhilam/jhilam03.png')]},
            txt_prompts = {},
        ),
        gifmaker.AnimateParams(
            size = [600, 400],
            init_image = imfp.joinpath('stock_photos/nyc.png'),
            img_prompts = {0: [imfp.joinpath('jhilam/jhilam04.png')]},
            txt_prompts = {},
        ),
        gifmaker.AnimateParams(
            size = [600, 400],
            init_image = imfp.joinpath('stock_photos/nyc.png'),
            img_prompts = {0: [imfp.joinpath('jhilam/jhilam04.png')]},
            txt_prompts = {0: ['beautiful sunset in the city']},
        ),
        gifmaker.AnimateParams(
            size = [600, 400],
            init_image = imfp.joinpath('stock_photos/nyc.png'),
            img_prompts = {0: [imfp.joinpath('jhilam/jhilam04.png')]},
            txt_prompts = {0: ['colors of the city']},
        ),
        gifmaker.AnimateParams(
            size = [600, 400],
            init_image = imfp.joinpath('stock_photos/nyc.png'),
            img_prompts = {0: [imfp.joinpath('jhilam/jhilam04.png')]},
            txt_prompts = {0: ['beautiful colors in the city']},
        ),
        gifmaker.AnimateParams(
            size = [600, 400],
            init_image = imfp.joinpath('stock_photos/nyc.png'),
            img_prompts = {0: [imfp.joinpath('jhilam/jhilam05.png')]},
            txt_prompts = {},
        ),
    ]
    
    # count params
    print(f'running {len(all_params)} param configurations')

    # start the outer loop
    for params in tqdm.tqdm(all_params):
        gifmaker.make_gif(
            'jhilam01_cityscape', 
            pathlib.Path(f'output/group04'), 
            params,
            seed = 0, 
            display_freq = None
        )


