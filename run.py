import threading
import time
import torch
from diffusers import DiffusionPipeline, DDPMScheduler, StableCascadePriorPipeline, StableCascadeDecoderPipeline, StableCascadeCombinedPipeline
from PIL import Image
import pygame_textinput
import pygame
from pygame.locals import *

import tkinter
import tkinter.simpledialog
import uuid, random


WIDTH = 1280
HEIGHT = 900
IMG_WIDTH = 400
IMG_HEIGHT = 400

PROMPT = ""
NEGATIVE_PROMPT = "bad anatomy, lowres, normal quality, worstquality, watermark, bad proportions, out of focus, long neck, deformed, mutated, mutation, disfigured, poorly drawn face, skin blemishes, skin spots, acnes, missing limb, malformed limbs, floating limbs, disconnected limbs, extra limb, extra arms, mutated hands, poorly drawn hands, malformed hands, mutated hands and fingers, bad hands, missing fingers, fused fingers, too many fingers, extra legs, bad feet, cross-eyed, (distorted, :1.3) , (:1.4) , low quality, camera, BadDream, UnrealisticDream, bad-hands-5, BadNegAnatomyV1-neg, EasyNegative, FastNegativeV2, bad-picture-chill-75v"

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Stable Diffusion')


def drawText(surface, text, color, rect, font, aa=False, bkg=None):
    rect = Rect(rect)
    y = rect.top
    lineSpacing = -2

    # get the height of the font
    fontHeight = font.size("Tg")[1]

    while text:
        i = 1

        # determine if the row of text will be outside our area
        if y + fontHeight > rect.bottom:
            break

        # determine maximum width of line
        while font.size(text[:i])[0] < rect.width and i < len(text):
            i += 1

        # if we've wrapped the text, then adjust the wrap to the last word
        if i < len(text):
            i = text.rfind(" ", 0, i) + 1

        # render the line and blit it to the surface
        if bkg:
            image = font.render(text[:i], 1, color, bkg)
            image.set_colorkey(bkg)
        else:
            image = font.render(text[:i], aa, color)

        surface.blit(image, (rect.left, y))
        y += fontHeight + lineSpacing

        # remove the text we just blitted
        text = text[i:]

    return text

def generate(prompt, negative_prompt, seed):
    global py_image
    global py_image_rect
    global working
    global image_surfaces
    global cursor
    working = True
    generator = torch.Generator(device="cuda").manual_seed(seed)

    #pipe_id = "runwayml/stable-diffusion-v1-5"
    pipe_id = "prompthero/openjourney"
    pipe = DiffusionPipeline.from_pretrained(
        pipe_id,
        torch_dtype=torch.float16,
        #variant="fp16",
        use_safetensors=True,
        num_inference_steps=steps,
        guidance_scale=8.0,
        sag_scale=0.75
    )
    #pipe.enable_model_cpu_offload()
    pipe.to("cuda")


    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=8.0,
        sag_scale=0.7,
        generator=generator,
        width=IMG_WIDTH,
        height=IMG_HEIGHT,
    ).images[0]

    mode = image.mode
    size = image.size
    data = image.tobytes()

    py_image = pygame.image.fromstring(data, size, mode)
    image_surfaces.append(py_image)
    cursor = len(image_surfaces)-1

    working = False


gen_thread = None
font = pygame.font.SysFont('Hack', 16)
font2 = pygame.font.SysFont('Hack', 14)
text_surfaces = [None for n in range(8)]
image_surfaces = []
cursor = 0
cursor_x = 0
clock = pygame.time.Clock()
seed = 0
steps = 39


working = False
running = True
while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == K_RETURN:
                if working is False:
                    gen_thread = threading.Thread(target=generate, args=(PROMPT, NEGATIVE_PROMPT, seed))
                    gen_thread.start()
            if event.key == K_PAGEUP:
                steps += 1
            if event.key == K_PAGEDOWN:
                steps -= 1
            if event.key == K_UP:
                seed = random.randint(seed, seed+1000)
                seed += 1000
            if event.key == K_DOWN:
                seed = random.randint(seed-1000, seed)
                seed -= 1000
            if event.key == K_LEFT:
                if cursor > 0:
                    cursor -= 1
            if event.key == K_RIGHT:
                if (cursor+1) < len(image_surfaces):
                    cursor += 1
            if event.key == K_r:
                PROMPT = str(tkinter.simpledialog.askstring("Titre", "PROMPT:", initialvalue=PROMPT))
            if event.key == K_n:
                NEGATIVE_PROMPT = str(tkinter.simpledialog.askstring("Titre", "NEGATIVE PROMPT:", initialvalue=NEGATIVE_PROMPT))
            if event.key == K_s:
                if (len(image_surfaces) > 0):
                    pygame.image.save(image_surfaces[cursor], str(uuid.uuid4())+".png")
            if event.key == K_ESCAPE:
                running = False
        if event.type == pygame.QUIT:
            running = False


    screen.fill((0, 0, 0))

    if len(image_surfaces) > 0:
        x = WIDTH//2 - ((len(image_surfaces)+cursor-1) * 20)
        for i in image_surfaces:
            if image_surfaces.index(i) == cursor:
                cursor_x = x
                x += 40
                continue
            rect = i.get_rect()
            rect.center = (x, HEIGHT//2)
            screen.blit(i, rect)
            pygame.draw.rect(screen, (255, 255, 255), rect, 2)
            x += 40

        rect = image_surfaces[cursor].get_rect()
        rect.center = (cursor_x, HEIGHT//2)
        screen.blit(image_surfaces[cursor], rect)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)


    y = 5
    for s in text_surfaces:
        if s is not None:
            screen.blit(s, (5,y))
            y += 20


    # HUD
    text_surfaces[0] = font.render("[R] Edit prompt | [N] Edit negative prompt | [S] Save picture", True, (255, 255, 255))
    text_surfaces[1] = font.render("[▲] Increase seed | [▼] Decrease seed", True, (255, 255, 255))
    text_surfaces[2] = font.render("[⇞] Increase steps | [⇟] Decrease steps", True, (255, 255, 255))
    text_surfaces[3] = font.render("[Enter] Generate | [Esc] Exit", True, (255, 255, 255))
    text_surfaces[4] = font.render("Seed: " + str(seed), True, (255, 255, 255))
    text_surfaces[5] = font.render("Steps: " + str(steps), True, (255, 255, 255))

    # PROMPT
    rect = Rect(0, HEIGHT//1.23, WIDTH, HEIGHT-(HEIGHT//1.23))
    drawText(screen, "PROMPT: " + PROMPT, (203, 255, 140), rect, font2, aa=True, bkg=None)

    # NEGATIVE PROMPT
    rect = Rect(0, HEIGHT//1.12, WIDTH, HEIGHT-(HEIGHT//1.12))
    drawText(screen, "NEGATIVE PROMPT: " + NEGATIVE_PROMPT, (191,26,47), rect, font2, aa=True, bkg=None)


    if working is True:
        text = font.render("Generating...", True, (236, 157, 237))
        screen.blit(text, (WIDTH//2 - text.get_rect().width//2,HEIGHT//2))


    clock.tick(30)
    pygame.display.flip()





