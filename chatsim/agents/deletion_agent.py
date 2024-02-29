import openai 
import numpy as np
from termcolor import colored
import traceback
import openai
import os
import cv2
import imageio.v2 as imageio
import copy
import sys
from chatsim.agents.utils import check_and_mkdirs, transform_nerf2opencv_convention, generate_vertices, get_outlines

class DeletionAgent:
    def __init__(self, config):
        self.config = config
        self.inpaint_dir = config['inpaint_dir'] # image inpaint
        self.video_inpaint_dir = config['video_inpaint_dir']

    
    def llm_finding_deletion(self, scene, message, scene_object_description):
        try:
            q0 = "I will provide you with an operation statement and a dictionary containing information about cars in a scene. " + \
                 " You need to determine which car or cars should be deleted from the dictionary. " 

            q1 = "The dictionary is " + str(scene_object_description)

            q2 = "The keys of the dictionary are the car IDs, and the value is also a dictionary containing car detail, " + \
                 "including its image coordinate (u,v) in an image frame, depth, color in RGB."

            q2 = "My statement may include information about the car's color or position. You should find out from my statement which cars should be deleted and return their car IDs"
            
            q3 = "Note: (1) The definitions of u and v conform to the image coordinate system, u=0, v=0 represents the upper left corner. " + \
                 "And the larger the 'u', the more to the right; And the larger the 'v', the more to the down. " + \
                 "(2) You can judge the distance by the 'depth'. The greater the depth, the farther the distance, the smaller the depth, the closer the distance." + \
                 "(3) The description of the color may not be absolutely accurate, choose the car with the closest color."
            
            q4 = "You should return a JSON dictionary, with a key: 'removed_cars'." + \
                " 'removed_cars' contains IDs of all the cars that meet the requirements. "
            
            q5 = "Note that there is no need to return any code or explanations; only provide a JSON dictionary."

            q6 = "The requirement is :" + message
            
            prompt_list = [q0,q1,q2,q3,q4,q5,q6]

            result = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an assistant helping me to assess and maintain information in a dictionary."}] + \
                         [{"role": "user", "content": q} for q in prompt_list]
            )   

            answer = result['choices'][0]['message']['content']
            print(f"{colored('[Deletion Agent LLM] finding the car to delete', color='magenta', attrs=['bold'])} \
                    \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}")

            start = answer.index("{")
            answer = answer[start:]
            end = answer.rfind("}")
            answer = answer[:end+1]
            deletion_car_ids = eval(answer)['removed_cars']

            print(f"{colored('[Extracted Response>>>]', attrs=['bold'])} {deletion_car_ids} \n")

        
        except Exception as e:
            print(e)
            traceback.print_exc()
            print("[Deletion Agent LLM] finding the car to delete fails")
            return []

        return deletion_car_ids
    
    def llm_putting_back_deletion(self, scene, message, scene_object_description):
        try: 
            deleted_object_dict = {k: v for (k,v) in scene_object_description.items() if k in scene.removed_cars}
            
            q0 = "I will provide you with a dictionary in which each key is a vehicle id, and each value is the description of the vehicle in the image."

            q1 = "Specifically, description of the vehicle is also a dictionary. It has keys: (1) vehicle's u in image coordinate (2) vehicle's v in image coordinate (3) vehicle color in RGB. (4) vehicle's depth from viewpoint"

            q2 = "The definitions of u and v conform to the image coordinate system, u=0, v=0 represents the upper left corner. " + \
                  "The larger the 'u', the more to the right; And the larger the 'v', the more to the down. "

            q3 = "I will get you a requirement, and I want you can follow this requirement and take out all the relavant vehicle ids from the dictionary."
            
            q4 = f"Now the dictionary is {deleted_object_dict}, and my requirement is {message}. My requirement may contain extraneous verb descriptions or the wrong singular and plural expression, please ignore."

            q5 = "Note that you should return a JSON dictionary, the key is 'selected_vehicle', the value includes the vehicle ids. DO NOT return anything else. I'm not asking you to write code."

            prompt_list = [q0, q1, q2, q3, q4, q5]

            result = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an assistant helping me maintain and return dictionaries."}] + \
                         [{"role": "user", "content": q} for q in prompt_list]
            )

            answer = result['choices'][0]['message']['content']

            print(f"{colored('[Deletion Agent LLM] finding the car to be put back', color='magenta', attrs=['bold'])}  \
                    \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}")

            start = answer.index("{")
            answer = answer[start:]
            end = answer.rfind("}")
            answer = answer[:end+1]
            put_back_car_ids = eval(answer)['selected_vehicle'] # for example, ['0', '1']

            print(f"{colored('[Extracted Response>>>]', attrs=['bold'])} {put_back_car_ids} \n")

        except Exception as e:
            print(e)
            traceback.print_exc()
            print("[Deletion Agent LLM] finding the car to be put back fails")

        return put_back_car_ids
    
    
    def func_inpaint_scene(self, scene):
        """
        Call inpainting, store results in scene.current_inpainted_images

        if no scene.removed_cars
            just return

        """
        # if no need to inpainting, assign current images as current inpainted images
        if len(scene.removed_cars) == 0:
            print(f"{colored('[Inpaint]', 'green', attrs=['bold'])} No inpainting.")
            scene.current_inpainted_images = scene.current_images
            return

        current_dir = os.getcwd()
        inpaint_input_path = os.path.join(current_dir, scene.cache_dir, "inpaint_input")
        inpaint_output_path = os.path.join(current_dir, scene.cache_dir, "inpaint_output")

        check_and_mkdirs(inpaint_input_path)
        check_and_mkdirs(inpaint_output_path)

        if scene.is_ego_motion is False: # inpaint image
            print(f"{colored('[Inpaint]', 'green', attrs=['bold'])} is_ego_motion is False, inpainting one frame.")

            all_mask = self.func_get_mask(scene)
            # current images are all the same. 
            # can be one or multiple frames.
            img = scene.current_images[0]
            masked_img = copy.deepcopy(img)
            if scene.is_wide_angle:
                masked_img = cv2.resize(masked_img, (1152,256))
            else:
                masked_img = cv2.resize(masked_img, (512, 384))

            imageio.imwrite(os.path.join(inpaint_input_path, "img.png"), masked_img.astype(np.uint8))
            imageio.imwrite(os.path.join(inpaint_input_path, "img_mask.png"), all_mask.astype(np.uint8))

            current_dir = os.getcwd()
            os.chdir(self.inpaint_dir)
            os.system(
                f"python scripts/inpaint.py --indir {inpaint_input_path} --outdir {inpaint_output_path}"
            )
            os.chdir(current_dir)

            new_img = imageio.imread(os.path.join(inpaint_output_path, "img.png"))
            new_img = cv2.resize(new_img, (scene.width, scene.height)) 

            #combine the inpainted image and the original image
            all_mask_in_ori_resolution = (cv2.resize(all_mask, (scene.width, scene.height))).reshape(scene.height,scene.width,1).repeat(3,axis=2)
            new_img = np.where(all_mask_in_ori_resolution==0, scene.current_images[0], new_img)

            scene.current_inpainted_images = [new_img] * scene.frames

        else:  # inpaint video
            print(f"{colored('[Inpaint]', 'green', attrs=['bold'])} is_ego_motion is True, inpainting multiple frame (as video).")

            mask_list = []
            for i in range(scene.frames):
                current_frame_mask = np.zeros((scene.height, scene.width))

                for car_id in scene.bbox_data.keys():
                    if scene.bbox_car_id_to_name[car_id] in scene.removed_cars:
                        corners = generate_vertices(scene.bbox_data[car_id])
                        mask, mask_corners = get_outlines(corners, 
                                                      transform_nerf2opencv_convention(scene.current_extrinsics[i]), 
                                                      scene.intrinsics, 
                                                      scene.height, 
                                                      scene.width
                        )
                        current_frame_mask[mask == 1] = 1
                mask_list.append(current_frame_mask)

            np.save(f'{self.video_inpaint_dir}/chatsim/masks.npy', mask_list)
            np.save(f'{self.video_inpaint_dir}/chatsim/current_images.npy', scene.current_images)

            current_dir = os.getcwd()
            os.chdir(self.video_inpaint_dir)
            os.system(f'python remove_anything_video_npy.py \
                        --dilate_kernel_size 15 \
                        --lama_config lama/configs/prediction/default.yaml \
                        --lama_ckpt ./pretrained_models/big-lama \
                        --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
                        --vi_ckpt ./pretrained_models/sttn.pth \
                        --mask_idx 2 \
                        --fps 25')      
            os.chdir(current_dir)

            print(f"{colored('[Inpaint]', 'green', attrs=['bold'])} Video Inpainting Done!")
            inpainted_images = np.load(f'{self.video_inpaint_dir}/chatsim/inpainted_imgs.npy', allow_pickle = True)
            scene.current_inpainted_images = [np.array(image) for image in inpainted_images]

            
    def func_get_mask(self, scene):
        masks = []
        extrinsic_for_project = transform_nerf2opencv_convention(
            scene.current_extrinsics[0]
        )
        for car_name in scene.removed_cars:
            car_id = scene.name_to_bbox_car_id[car_name]
            corners = generate_vertices(scene.bbox_data[car_id])
            mask, _ = get_outlines(
                corners,
                extrinsic_for_project,
                scene.intrinsics,
                scene.height,
                scene.width,
            )
            mask *= 255
            masks.append(mask)
        mask = np.max(np.stack(masks), axis=0)
        if scene.is_wide_angle:
            mask = cv2.resize(mask, (1152, 256))
        else:
            mask = cv2.resize(mask, (512, 384))
        
        return mask