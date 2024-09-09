"""
Script Name: app.py
Description: Gradio app to perform mask generation, object removal, inpainting and image processing
Author: Mandar Avhad
Date: 2024-09-10
"""


import os
import sys

# debugging
# Get the current working directory
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

import gradio as gr
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch
import tempfile
from lama_inpaint import inpaint_img_with_lama, build_lama_model, inpaint_img_with_builded_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "segment-anything"))
from segment_anything import SamPredictor, sam_model_registry
import argparse
import cv2

# from models.clipseg import CLIPDensePredT
from clipseg.models.clipseg import CLIPDensePredT
from torchvision import transforms
from io import BytesIO
from torch import autocast
import requests
import PIL
from diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline


def setup_args(parser):
    parser.add_argument(
        "--lama_config", type=str,
        default="./third_party/lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default="pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )
    parser.add_argument(
        "--sam_ckpt", type=str,
        default="./pretrained_models/sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )
def mkstemp(suffix, dir=None):
    fd, path = tempfile.mkstemp(suffix=f"{suffix}", dir=dir)
    os.close(fd)
    return Path(path)


def get_sam_feat(img):
    model['sam'].set_image(img)
    features = model['sam'].features 
    orig_h = model['sam'].orig_h 
    orig_w = model['sam'].orig_w 
    input_h = model['sam'].input_h 
    input_w = model['sam'].input_w 
    model['sam'].reset_image()
    print("===SAM features Done====")
    return features, orig_h, orig_w, input_h, input_w

 
def get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size):
    point_coords = [w, h]
    point_labels = [1]

    model['sam'].is_image_set = True
    model['sam'].features = features
    model['sam'].orig_h = orig_h
    model['sam'].orig_w = orig_w
    model['sam'].input_h = input_h
    model['sam'].input_w = input_w

    masks, _, _ = model['sam'].predict(
        point_coords=np.array([point_coords]),
        point_labels=np.array(point_labels),
        multimask_output=True,
    )

    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]
    else:
        masks = [mask for mask in masks]

    figs = []
    for idx, mask in enumerate(masks):
        # save the pointed and masked image
        tmp_p = mkstemp(".png")
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [point_coords], point_labels,
                    size=(width*0.04)**2)
        show_mask(plt.gca(), mask, random_color=False)
        plt.tight_layout()
        plt.savefig(tmp_p, bbox_inches='tight', pad_inches=0)
        figs.append(fig)
        plt.close()
    print("====Get Masked Image Done====")
    return *figs, *masks


def get_inpainted_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size):

    img_with_mask_0, img_with_mask_1, img_with_mask_2, mask0, mask1, mask2 = get_masked_img(img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size)

    lama_config = args.lama_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for mask in [mask0, mask1, mask2]:
        if len(mask.shape)==3:
            mask = mask[:,:,0]
        img_inpainted = inpaint_img_with_builded_lama(
            model['lama'], img, mask, lama_config, device=device)
        out.append(img_inpainted)
    print("====Get Inpainted Image Done====")
    return *out, mask0, mask1, mask2


def get_walls_painted_cv2(img_rm_with_mask_1, img_rm_with_mask_2, selected_color_output, color_options):
    # For blending
    alpha=0.6 

    selected_color_output = handle_selection(color_options)
    # Reversed tuple, rgb to bgr input
    selected_color_output = selected_color_output[::-1]

    img_list = [img_rm_with_mask_1, img_rm_with_mask_2]
    
    output_img_list = []
    
    for i in range(2):
        image = img_list[i]
        # mask  = mask_list[i]

        ###############################
        # building's mask generation logic here, prompt based
        # debugging
        # cv2.imwrite("img_1.jpg", image)

        image, mask = generate_building_mask(image)
        display_mask = mask
        ####

        # For ensuring if mask is binary
        mask = mask.astype(np.uint8)
        mask[mask > 0] = 1

        # Create a color layer matching the image size, filled with the desired color
        color_layer = np.zeros_like(image, dtype=np.uint8)
        color_layer[:] = selected_color_output

        # Apply the color layer only to the masked region
        colored_object = cv2.bitwise_and(color_layer, color_layer, mask=mask)

        # Extract the original image object (in the masked region)
        original_object = cv2.bitwise_and(image, image, mask=mask)

        # Blend the colored object with the original object using alpha blending
        blended_object = cv2.addWeighted(original_object, 1 - alpha, colored_object, alpha, 0)

        # Combine the blended object with the original background
        background = cv2.bitwise_and(image, image, mask=1 - mask)
        painted_image = cv2.add(blended_object, background)

        # Convert from BGR to RGB
        painted_image = cv2.cvtColor(painted_image, cv2.COLOR_BGR2RGB)

        # debugging
        # cv2.imwrite("mask_1.jpg", mask)
        # cv2.imwrite("blended_1.jpg", painted_image)
        
        output_img_list.append(painted_image)
    
    print("====Get Walls Painted CV2 Image Done====")
    return display_mask, output_img_list[0], output_img_list[1]


# Function to handle radio button selection of colors
def handle_selection(selected_color):
    # returning RGB values of selected colors
    if selected_color == "White":
        return (255,255,255)
    elif selected_color == "Sleigh Bells (Grey)":
        return (134,134,134)
    elif selected_color == "Garlic Pod (Cream)":
        return (243,229,203)
    elif selected_color == "Sunderbans (Brown)":
        return (187,145,97)

############################
# prompt based building mask generation
def generate_building_mask(input_image_arr):
    # Get the current working directory
    current_directory = os.getcwd()
    # Print the current working directory
    print("2 Current Working Directory:", current_directory)

    # load model
    model_clip = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model_clip.eval();
    
    # non-strict, because we only stored decoder weights (not CLIP weights)
    model_clip.load_state_dict(torch.load('./clipseg/weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False);

    input_image_pil = Image.fromarray(input_image_arr)
    # mask_image_pil = Image.fromarray(mask_image_arr)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ])
    img = transform(input_image_pil).unsqueeze(0)

    input_image_pil.convert("RGB").resize((512, 512)).save("init_image.png", "PNG")

    # prompt/name of the object whose mask is to be generated
    prompts = ['buildings']
    
    # predict
    with torch.no_grad():
        preds = model_clip(img.repeat(len(prompts),1,1,1), prompts)[0]

    filename = f"mask.png"
    plt.imsave(filename,torch.sigmoid(preds[0][0]))

    # mask image
    img2 = cv2.imread(filename)

    gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    # For debugging
    cv2.imwrite(filename,bw_image)
    # fix color format
    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
    
    image = cv2.imread('init_image.png')
    
    return image, bw_image
########################


# get args 
parser = argparse.ArgumentParser()
setup_args(parser)
args = parser.parse_args(sys.argv[1:])
# build models
model = {}
# build the sam model
model_type="vit_h"
ckpt_p=args.sam_ckpt
model_sam = sam_model_registry[model_type](checkpoint=ckpt_p)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_sam.to(device=device)
model['sam'] = SamPredictor(model_sam)

# build the lama model
lama_config = args.lama_config
lama_ckpt = args.lama_ckpt
device = "cuda" if torch.cuda.is_available() else "cpu"
model['lama'] = build_lama_model(lama_config, lama_ckpt, device=device)

button_size = (100,50)
with gr.Blocks() as demo:
    
    features = gr.State(None)
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Input")
            with gr.Row():
                img = gr.Image(type="numpy", label="Input Image")
        with gr.Column(variant="panel"):
            # with gr.Row():
            #     gr.Markdown("## Pointed Image")
            # with gr.Row():
            #     img_pointed = gr.Plot(label='Pointed Image')
            with gr.Row():
                gr.Markdown("## Additional Inputs")
            with gr.Row():
                w = gr.Number(label="Point Coordinate W")
                h = gr.Number(label="Point Coordinate H")
            dilate_kernel_size = gr.Slider(label="Dilate Kernel Size", minimum=0, maximum=100, step=1, value=15)

            # Add Radio Buttons
            color_options = gr.Radio(choices=["White", "Sleigh Bells (Grey)", "Garlic Pod (Cream)", "Sunderbans (Brown)"], label="Choose a color to paint the walls")
            # Text output to display the result
            selected_color_output = gr.Textbox(label="RGB Value")
            # Action: Call the handle_selection function when the radio button is selected
            color_options.change(fn=handle_selection, inputs=color_options, outputs=selected_color_output)

        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Get Results")
            # with gr.Row():
            #     w = gr.Number(label="Point Coordinate W")
            #     h = gr.Number(label="Point Coordinate H")
            # dilate_kernel_size = gr.Slider(label="Dilate Kernel Size", minimum=0, maximum=100, step=1, value=15)
            # dilate_kernel_size = 15

            # # Add Radio Buttons
            # color_options = gr.Radio(choices=["Grey", "Cream"], label="Choose a color to paint the walls")
            # # Text output to display the result
            # selected_color_output = gr.Textbox(label="RGB Value")
            # # Action: Call the handle_selection function when the radio button is selected
            # color_options.change(fn=handle_selection, inputs=color_options, outputs=selected_color_output)

            lama = gr.Button("Remove Object & Inpaint", variant="primary")
            wall_paint_cv2 = gr.Button("Get Final Output", variant="primary")
            clear_button_image = gr.Button(value="Reset", variant="secondary")
    
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Segmentation Mask", visible=False)
            with gr.Row():
                mask_0 = gr.Image(type="numpy", label="Segmentation Mask 0", visible=False)
                mask_1 = gr.Image(type="numpy", label="Segmentation Mask 1", visible=False)
                mask_2 = gr.Image(type="numpy", label="Segmentation Mask 2", visible=False)

    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image with Mask", visible=False)
            with gr.Row():
                img_with_mask_0 = gr.Plot(label="Image with Segmentation Mask 0", visible=False)
                img_with_mask_1 = gr.Plot(label="Image with Segmentation Mask 1", visible=False)
                img_with_mask_2 = gr.Plot(label="Image with Segmentation Mask 2", visible=False)

    # object removed output
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Image with Object Removed")
            with gr.Row():
                img_rm_with_mask_0 = gr.Image(
                    type="numpy", label="Image with Object Removed 0", visible=False)
                img_rm_with_mask_1 = gr.Image(
                    type="numpy", label="Image with Object Removed 1")
                img_rm_with_mask_2 = gr.Image(
                    type="numpy", label="Image with Object Removed 2")
    
    # final cv2 wall paint output
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                gr.Markdown("## Final Output with Walls Painted")
            with gr.Row():
                final_mask_1 = gr.Image(
                    type="numpy", label="Building Mask")
                final_img_1 = gr.Image(
                    type="numpy", label="Result 1")
                final_img_2 = gr.Image(
                    type="numpy", label="Result 2")

    ###########

    def get_select_coords(img, evt: gr.SelectData):

        color = (0, 255, 0)  # Green color marker
        marker_type = cv2.MARKER_TRIANGLE_UP # cv2.MARKER_CROSS
        marker_size = 20
        thickness = 5
        x = evt.index[0]
        y = evt.index[1]
        marked_img = cv2.drawMarker(img, (x, y), color, markerType=marker_type, 
                    markerSize=marker_size, thickness=thickness)

        return evt.index[0], evt.index[1], marked_img

    img.select(get_select_coords, [img], [w, h, img]) # , img
    img.upload(get_sam_feat, [img], [features, orig_h, orig_w, input_h, input_w])


    # to get image with object removed
    lama.click(
        get_inpainted_img,
        [img, w, h, features, orig_h, orig_w, input_h, input_w, dilate_kernel_size],
        [img_rm_with_mask_0, img_rm_with_mask_1, img_rm_with_mask_2, mask_0, mask_1, mask_2]
    )

    # to get the image with walls painted, cv2
    wall_paint_cv2.click(
        get_walls_painted_cv2,
        [img_rm_with_mask_1, img_rm_with_mask_2, selected_color_output, color_options],
        [final_mask_1, final_img_1, final_img_2]
    )
 

    def reset(*args):
        return [None for _ in args]

    clear_button_image.click(
        reset,
        [img, features, w, h, mask_0, mask_1, mask_2, img_with_mask_0, img_with_mask_1, img_with_mask_2, img_rm_with_mask_0, img_rm_with_mask_1, img_rm_with_mask_2, final_mask_1, final_img_1, final_img_2],
        [img, features, w, h, mask_0, mask_1, mask_2, img_with_mask_0, img_with_mask_1, img_with_mask_2, img_rm_with_mask_0, img_rm_with_mask_1, img_rm_with_mask_2, final_mask_1, final_img_1, final_img_2]
    )

if __name__ == "__main__":
    # demo.launch()
    demo.launch(max_threads=400, server_name="0.0.0.0", server_port=7000, share=True)
    
    