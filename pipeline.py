import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = 512 // 8
LATENT_HEIGHT = 512 // 8

def generate(prompt: str, uncond_prompt: str, input_image=None, strength= 0.8, 
             do_cfg=True, 
             cfg_scale=7.5, sampler_name= 'ddpm', n_inference_steps=50,
             models={}, seed= None,
             device=None, idle_device=None,
             tokenizer= None):
    with torch.no_grad():

        if not (0< strength <=1):
            raise ValueError("strength must be btw 0 and 1")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
            
        clip = models['clip']
        clip.to(device)
        
        if do_cfg:
           # convert into a list of length seq_len = 77
           cond_tokens = tokenizer.batch_encode_plus(
               [prompt], padding = 'max_length', max_length=77
            ).input_ids
           
            #(bs, seq_len)
           cond_tokens = torch.tensor(cond_tokens, dtype = torch.long, device=device)
           # (bs, seq_len) -> (bs, seq_len, embedding_dim)
           cond_context = clip(cond_tokens)
           uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
           uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
           uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
           context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)
        
        
        
        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")
        
        latents_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)
        
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            #(h, w, channels)
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype = torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,-1))
            
            #( height, width, channels) -> (batch_size, height, width, channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            
            # (ba, 4, latent_h, latent_w)
            encoder_noise = torch.randn(latents_shape, generator=generator, device = device )
            latents = encoder(input_image_tensor, encoder_noise)
            
            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            to_idle(encoder)
        else:
            # random noise to start with 
            latents = torch.randn(latents_shape, generator = generator, device=device)
        
        
        diffusion = models["diffusion"]
        diffusion.to(device)
        
        timesteps = tqdm(sampler.timesteps)
        for i , timestep in enumerate(timesteps):
            # (1, 320)
            time_ebedding = get_time_embedding(timestep).to(device)
            
            model_input = latents
            if do_cfg:
            # (bs, 4, latent_h, laent_width) -> (2*batch , latent_h, latent_w)
                model_input = model_input.repeat(2,1, 1, 1)
            
            model_output = diffusion(model_input, context, time_ebedding)
            
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale* (output_cond-output_uncond) + output_uncond
            
            # (bs, 4, latent_h, latent_w) -> (bs, 4, latent_h, latent_w)
            latents = sampler.step(timestep, latents, model_output)
            
        to_idle(diffusion)
        
        decoder = models["deocer"]
        decoder.to(device)
        #(bs, 4, latent_h, latent_w) -> (bs, 3, h, w)
        
        images = decoder(latents)
        to_idle(decoder)
        
        images = rescale(images, (-1, -1), (0, 255) , clamp = True)
        #(bs , channel, height, width ) - > (bs, height, width, channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        
        return images[0]
    
def rescale(x, old_range, new_range, clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max-new_min)/(old_max - old_min)
    x += new_min
    
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    
    
        
        
        
        
