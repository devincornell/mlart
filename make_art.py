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
            size = [500, 600],
            init_image = imfp.joinpath('kiersten/kiersten_dog.png'),
            img_prompts = {
                0: [
                    imfp.joinpath('kiersten/kiersten_owl_eyes.png')
                ]
            },
            txt_prompts = {},
        ),
        gifmaker.AnimateParams(
            size = [500, 600],
            init_image = imfp.joinpath('kiersten/kiersten_dog.png'),
            img_prompts = {
                0: [
                    imfp.joinpath('stock_photos/cosmos_stock.png')
                ]
            },
            txt_prompts = {},
        ),
        gifmaker.AnimateParams(
            size = [500, 600],
            init_image = imfp.joinpath('kiersten/kiersten_dog.png'),
            img_prompts = {
                0: [imfp.joinpath('stock_photos/flag_horizontal.png')]
            },
            txt_prompts = {},
        ),
        gifmaker.AnimateParams(
            size = [500, 600],
            init_image = imfp.joinpath('kiersten/kiersten_dog.png'),
            img_prompts = {
                0: [imfp.joinpath('stock_photos/forest_bridge.png')]
            },
            txt_prompts = {},
        ),
        gifmaker.AnimateParams(
            size = [500, 600],
            init_image = imfp.joinpath('kiersten/kiersten_dog.png'),
            img_prompts = {
                0: [imfp.joinpath('stock_photos/pink_blue_windows.png')]
            },
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


