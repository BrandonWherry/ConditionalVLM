import torch
from cse.trainer import *
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, transforms, Normalize, InterpolationMode
import torchvision
from cse.re_alogirthms_unsafe import *
from cse.bass_functions import *
from PIL import Image
from torchvision import models
import glob
import os
from skimage.color import rgb2gray
from skimage.filters import sobel
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
    
def main(attr_map: str = 'full_grad',
         seg_map: str = 'bass',
         output_class: int = 0,
         img_dir: str='data/train',
         unsafe_directory=None,
         bass_output_folder=None,
         sam_output_folder=None,
         model_path=None):
    
    metric_folder_path = os.path.join(unsafe_directory, 'masked_trained_resnet')
    results_folder_path = os.path.join(unsafe_directory, 'masked_trained_resnet')
    
    if not os.path.exists(metric_folder_path):
        os.makedirs(metric_folder_path)
        
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)
    
    metric_folder_path = os.path.join(metric_folder_path, 'metric_' + seg_map + '_' + attr_map)
    results_folder_path = os.path.join(results_folder_path, 'results_' + seg_map + '_' + attr_map)
    
    if not os.path.exists(metric_folder_path):
        os.makedirs(metric_folder_path)
        
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    top_n_start = 1
    top_n_stop = 10
    threshold = 0.50
    pruning_heuristic = 1
    batch_sz = 16
    model_dict = torch.load(model_path)
    
    model = models.resnet50(pretrained=False)
    last_linear_in_features = model.fc.in_features
    model.fc = nn.Linear(last_linear_in_features, 2) 
    
    model.load_state_dict(model_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()

    images = Image.open(img_dir)
    img_dir, img_name = (img_dir.split('/')[:-2], img_dir.split('/')[-1])

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGBA') if 'P' in img.getbands() else img), # Convert Palette images to RGBA
        transforms.Lambda(lambda img: img.convert('RGB')), # Convert RGBA to RGB
        #transforms.Resize(232, interpolation=InterpolationMode.BILINEAR), # Resize to 232
        #transforms.CenterCrop(224), # Center crop to 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize
    ])

    transform_viz = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGBA') if 'P' in img.getbands() else img), # Convert Palette images to RGBA
        transforms.Lambda(lambda img: img.convert('RGB')), # Convert RGBA to RGB
        #transforms.Resize(232, interpolation=InterpolationMode.BILINEAR), # Resize to 232
        #transforms.CenterCrop(224), # Center crop to 224
        transforms.ToTensor(),
         # Normalize
    ])
    
    def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(2)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(2)
        return tensor.cpu() * std + mean
    
    inv_img = transform(images).unsqueeze(0)
    img_np = inv_img.detach().cpu().squeeze().numpy()
    
    inv_img_viz = transform_viz(images).unsqueeze(0)
    img_np_viz = inv_img_viz.detach().cpu().squeeze().numpy()
    #plt.imshow(img_np)
    # compactness=50

    input_tensor = transform(images).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(output_class)]
    target_layers = [model.layer4[-1]]

    cam_constructors = {
        'grad_cam': lambda: GradCAM(model=model, target_layers=target_layers),
        'grad_cam++': lambda: GradCAMPlusPlus(model=model, target_layers=target_layers),
        'full_grad': lambda: FullGrad(model=model, target_layers=target_layers),
        'x_grad_cam': lambda: XGradCAM(model=model, target_layers=target_layers),
        'ablation_cam': lambda: AblationCAM(model=model, target_layers=target_layers)
    }

    gradient = sobel(rgb2gray(img_np.transpose((1, 2, 0))))
    bass_output_folder = str(Path(bass_output_folder + img_name).with_suffix('.csv'))
    sam_output_folder = str(Path(sam_output_folder + img_name).with_suffix('.csv'))
    segment_methods = {
        'slic': lambda: slic(img_np, n_segments=25, compactness=15, start_label=1, channel_axis=0),
        'bass': lambda: csv_mask_to_numpy(bass_output_folder),
        'felzen': lambda: felzenszwalb(img_np, scale=500, sigma=0.5, min_size=200, channel_axis=0),
        'watershed': lambda: watershed(gradient, markers=25, compactness=0.001),
        'sam': lambda: csv_mask_to_numpy_SAM(sam_output_folder),
    }

    cam_instance = cam_constructors.get(attr_map, lambda: None)()
    
    segments = segment_methods.get(seg_map, lambda: None)()
    print('NUMBER OF REGIONS:', len(np.unique(segments)))

    grayscale_cam = cam_instance(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    working_example = region_explainability(image = input_tensor, segment_mask = segments, top_n_start = top_n_start, 
                                            model = model, SMU_class_index = output_class, 
                                            threshold = threshold, top_n_stop = top_n_stop,
                                            MAX_BATCH_SZ = batch_sz,
                                            PRUNE_HEURISTIC = pruning_heuristic)

    torch.save(working_example, os.path.join(metric_folder_path, img_name + '.pt'))
    
    if working_example == -1:
        return -1

    ori_prediction = working_example[4][0]
    ori_confidence = working_example[3][0]
    cf_prediction = working_example[4][1]
    cf_confidence = working_example[3][1]

    print("Regions Analyzed", working_example[-3])
    print("Original Version Predicted Class:", ori_prediction, 
          "     With Confidence:", ori_confidence)
    print("Modified Version Predicted Class:", cf_prediction, 
          "     With Confidence:", cf_confidence)

    # Assuming input_tensor is a PyTorch tensor
    plot_images = [
        inv_img_viz.detach().cpu().squeeze().numpy().transpose((1, 2, 0)),
        grayscale_cam,
        segmentation.mark_boundaries(img_np_viz.transpose((1, 2, 0)), segments),
        denormalize(working_example[0].squeeze()).detach().cpu().squeeze().numpy().transpose((1, 2, 0))
    ]

    figure_name = plt.figure(figsize=(14, 14))
    for i, img in enumerate(plot_images):
        plt.subplot(1, 4, i+1)
        plt.axis('off')

        # Check the shape of the image to determine if it's grayscale or RGB
        if len(img.shape) == 2:  # Grayscale
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.margins(x=0)


    img_dir = os.path.join(results_folder_path, img_name)
    print('Image name:', img_name)
    plt.tight_layout()
    figure_name.savefig(img_dir, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    ATTR_LIST = ['grad_cam', 'grad_cam++', 'full_grad', 'x_grad_cam', 'ablation_cam']
    SEG_LIST = ['slic', 'bass', 'felzen', 'watershed', 'sam']

    unsafe_directories = [
        '/workspace/adv_robustness/CSE/labelme/self_harm/',
        '/workspace/adv_robustness/CSE/labelme/cyberbullying/',
        '/workspace/adv_robustness/CSE/labelme/nsfw/',
    ]

    bass_out_folders = [
        '/workspace/adv_robustness/CSE/labelme/self_harm/BASS_output/',
        '/workspace/adv_robustness/CSE/labelme/cyberbullying/BASS_output/',
        '/workspace/adv_robustness/CSE/labelme/nsfw/BASS_output/',
    ]
    
    sam_out_folders = [
        '/workspace/adv_robustness/CSE/labelme/self_harm/SAM_output/',
        '/workspace/adv_robustness/CSE/labelme/cyberbullying/SAM_output/',
        '/workspace/adv_robustness/CSE/labelme/nsfw/SAM_output/',
    ]

    model_paths = [
        '/workspace/adv_robustness/CSE/unsafe_models_SH/best_model_masked.pth',
        '/workspace/adv_robustness/CSE/unsafe_models_CB/best_model.pth',
        '/workspace/adv_robustness/CSE/unsafe_models_NSFW/best_model_masked.pth',
    ] 

    image_directories = [
        '/workspace/adv_robustness/CSE/labelme/self_harm/test_images_transformed/',
        '/workspace/adv_robustness/CSE/labelme/cyberbullying/test_images_transformed/',
        '/workspace/adv_robustness/CSE/labelme/nsfw/test_images_transformed/',
    ]
    
    #for i in range(3):    
    for i in range(3):
        if i == 1:
            continue
        unsafe_directory = unsafe_directories[i]
        bass_out_folder = bass_out_folders[i]
        sam_out_folder = sam_out_folders[i]
        model_path = model_paths[i]
        image_directory = image_directories[i]
        images = [img for img in glob.glob(os.path.join(image_directory, '*')) if not os.path.basename(img).startswith('.')]
        
        seg_method = SEG_LIST[4]
        for attr in ATTR_LIST:
            for img in images:
                print('#'*100)
                print('SEG_METHOD:', seg_method)
                print('ATTR_METHOD:', attr)
                print()
                main(attr_map=attr,
                     seg_map=seg_method,
                     output_class=1,
                     img_dir=img,
                     unsafe_directory=unsafe_directory,
                     bass_output_folder=bass_out_folder,
                     sam_output_folder=sam_out_folder,
                     model_path=model_path)
                print()
                print('#'*100)
            print('Ratio of success')
            
        seg_method = SEG_LIST[1]
        for attr in ATTR_LIST:
            for img in images:
                print('#'*100)
                print('SEG_METHOD:', seg_method)
                print('ATTR_METHOD:', attr)
                print()
                main(attr_map=attr,
                     seg_map=seg_method,
                     output_class=1,
                     img_dir=img,
                     unsafe_directory=unsafe_directory,
                     bass_output_folder=bass_out_folder,
                     sam_output_folder=sam_out_folder,
                     model_path=model_path)
                print()
                print('#'*100)
            print('Ratio of success')
            
        attr = ATTR_LIST[2]
        for seg_method in SEG_LIST[::-1]:
            if seg_method == SEG_LIST[1] or seg_method == SEG_LIST[4]:
                continue
            for img in images:
                print('#'*100)
                print('SEG_METHOD:', seg_method)
                print('ATTR_METHOD:', attr)
                print()
                main(attr_map=attr,
                     seg_map=seg_method,
                     output_class=1,
                     img_dir=img,
                     unsafe_directory=unsafe_directory,
                     bass_output_folder=bass_out_folder,
                     sam_output_folder=sam_out_folder,
                     model_path=model_path)
                print()
                print('#'*100)
            print('Ratio of success')
            
