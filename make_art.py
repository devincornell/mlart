
import tqdm
import pathlib
from itertools import product

#sys.path.append('./taming-transformers')

import gifmaker

if __name__ == '__main__':

    #text_prompts_post = [
        #('', ''),
        #('ghibli', ' in the style of Studio Ghibli'),
        #('deviantart', ' from Deviantart'),
        #('artstation', ' from Artstation'),
        #('vray', ' from vray'),
        #('flickr', ' from Flickr'),
        #('unreal', ' rendered by Unreal Engine'),
    #]

    size = [600, 400]
    init_image = pathlib.Path('images/stock_images/angela_garden1.jpg')
    all_image_prompts = [
        {
            0: [
                init_image,
                #pathlib.Path('images/stock_images/dantes_inferno1.png'),
            ],
            200: [
                init_image,
                pathlib.Path('images/stock_images/sunset_forest.png'),
            ],
        },
        #[
        #    init_image,
        #    pathlib.Path('images/stock_images/cosmos_stock.png'),
        #],
        #[
        #    init_image,
        #    pathlib.Path('images/stock_images/starry_night.png'),
        #],
        #[
        #    init_image,
        #    pathlib.Path('images/stock_images/gia_fractal_deviantart.png'),
        #],
        #[
        #    init_image,
        #    pathlib.Path('images/stock_images/many_red_flowers.png'),
        #],
    ]

    if not all(p.exists() for ipt in all_image_prompts for ps in ipt.values() for p in ps):
        s = '\n'.join([f'{p}: {p.exists()}' for ipt in all_image_prompts for ps in ipt.values() for p in ps])
        raise ValueError(f'not all images were found! {s}')

    all_text_prompts = [
        {0: ['blue flowers'], 100: ['red flowers'], 200: ['sunset in the forest']},
        #['lightwave',],
        #['light wave',],
        #['sparkling colorful lights',],
        #['lightsabers',],
    ]
    
    # count params
    params = list(product(all_image_prompts, all_text_prompts))
    print(f'running {len(params)} param combinations')

    # start the outer loop
    for image_prompt_times, text_prompt_times in tqdm.tqdm(params):
        gifmaker.make_gif(
            'test06', 
            pathlib.Path(f'images'), 
            init_image = init_image, 
            image_prompt_times = image_prompt_times, 
            text_prompt_times = text_prompt_times, 
            size = size, 
            save_freq = 2, 
            step_size = 0.05, 
            max_iter = 300, 
            seed = 0, 
            still_frames = 0, 
            display_freq = None
        )


