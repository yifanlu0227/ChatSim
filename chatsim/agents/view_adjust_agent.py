import openai 
import numpy as np
from termcolor import colored
import traceback
import openai
from chatsim.background.mcnerf.scripts.inter_poses import inter_poses

class ViewAdjustAgent:
    def __init__(self, config):
        self.config = config

    def llm_reasoning_ego_motion(self, scene, message):
        try: 
            q0 = "I will give you a description about view adjustment, I need you to help me judge if the description is related to static view adjust or ego is dynamic(with motion)."
            q1 = "Given my description, return a dictionary in JSON format, with key 'if_view_motion'"
            q2 = "If the description is just a view adjust operation, the 'if_view_motion' should be 0. If the description is related to view motion, the 'if_view_motion' should be 1."
            
            # Few-shot examples
            q3 = "I will give you some examples. <user>: Rotate the viewpoint 30 degrees to the left, you should return {'if_view_motion':0}. " + \
                 "<user>: viewpoint moves ahead slowly, you should return {'if_view_motion':1}. "

            result = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an assistant helping me to provide information and ultimately return a JSON dictionary."},
                    {"role": "user", "content": q0},
                    {"role": "user", "content": q1},
                    {"role": "user", "content": q2},
                    {"role": "user", "content": q3},
                    {"role": "user", "content": message}
                    ]
            )

            answer = result['choices'][0]['message']['content']

            print(f"{colored('[View Adjust Agent LLM] reasoning the view motion', color='magenta', attrs=['bold'])}  \
                    \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}")

            start = answer.index("{")
            answer = answer[start:]
            end = answer.rfind("}")
            answer = answer[:end+1]
            if_view_motion = eval(answer)
            print(f"{colored('[Extracted Response>>>]', attrs=['bold'])} {if_view_motion} \n")

        except Exception as e:
            print(e)
            traceback.print_exc()
            return "[View Adjust Agent LLM] fails, can not recongnize instruction"

        if if_view_motion['if_view_motion'] == 0:
            return False
        else:
            return True
        

    def llm_view_motion_gen(self, scene, message):
        try: 
            q0 = "I will give you a description about ego motion, you should tell me the speed of ego."
            q1 = "Given my description, return a dictionary in JSON format, with key 'speed'."
            q2 = "If the ego motion is fast, 'speed' should be 'fast'; if the ego motion is slow, 'speed' should be 'slow'; if the description doesnot mention speed, 'speed' is default as 'fast'."
            
            # Few-shot examples
            q3 = "I will give you some examples. <user>: ego vehicle moves forward, you should return {'speed':'fast'}. " + \
                 "<user>: ego vehicle drives ahead slowly, you should return {'speed':'slow'}. "

            result = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an assistant helping me to provide information and ultimately return a JSON dictionary."},
                    {"role": "user", "content": q0},
                    {"role": "user", "content": q1},
                    {"role": "user", "content": q2},
                    {"role": "user", "content": q3},
                    {"role": "user", "content": message}
                    ]
            )

            answer = result['choices'][0]['message']['content']

            print(f"{colored('[View Adjust Agent LLM] generating the ego motion', color='magenta', attrs=['bold'])}  \
                    \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}")

            start = answer.index("{")
            answer = answer[start:]
            end = answer.rfind("}")
            answer = answer[:end+1]
            ego_motion_speed = eval(answer)
            print(f"{colored('[Extracted Response>>>]', attrs=['bold'])} {ego_motion_speed} \n")

        except Exception as e:
            print(e)
            traceback.print_exc()
            return "[View Adjust Agent LLM] fails, can not recongnize instruction"

        if ego_motion_speed['speed'] == 'fast':
            return 0, scene.nerf_motion_extrinsics.shape[0]
        else:
            return 0, (scene.nerf_motion_extrinsics.shape[0] // 3)


    def llm_view_adjust(self, scene, message):
        try:
            q0 = "I will give you a transformation operation for my viewpoint, which may include translation in 'x', 'y', 'z' or a rotation 'theta' around z-axis. " # General task description
            q1 = "For translation, positive 'x' represents forward, positve 'y' represents left, and 'z' represents up. It follows a left-hand coordinate system." + \
                 "For rotation, postive 'theta' is counterclockwise. So from own perspective, my viewpoint turns to the left. 'theta' is in degree." # Interpretation of details
            q2 = "Given my operation, return a dictionary in JSON format, with keys 'x', 'y', 'z', 'theta'." # Return format
            
            # Few-shot examples
            q3 = "I will give you some examples: <user>: Rotate the viewpoint 30 degrees to the left " + \
                 "<assistant>: {\n  'x': 0,\n  'y': 0,\n  'z': 0,\n  'theta': 30,\n } \n" + \
                 "<user>: move the viewpoint forward by 1 " + \
                 "<assistant>: {\n  'x': 1,\n  'y': 0,\n  'z': 0,\n  'theta': 0,\n }  \n" + \
                 "<user>: move the viewpoint to the right by 1" + \
                 "<assistant>: {\n  'x': 0,\n  'y': -1,\n  'z': 0,\n  'theta': 0,\n} "
            
            result = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an assistant helping me to provide information and ultimately return a JSON dictionary."},
                    {"role": "user", "content": q0},
                    {"role": "user", "content": q1},
                    {"role": "user", "content": q2},
                    {"role": "user", "content": q3},
                    {"role": "user", "content": message}
                    ]
            )

            answer = result['choices'][0]['message']['content']

            print(f"{colored('[View Adjust Agent LLM] analyzing view change', color='magenta', attrs=['bold'])}  \
                    \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}")

            start = answer.index("{")
            answer = answer[start:]
            end = answer.rfind("}")
            answer = answer[:end+1]
            delta_extrinsic = eval(answer)
            print(f"{colored('[Extracted Response>>>]', attrs=['bold'])} {delta_extrinsic} \n")
            
        except Exception as e:
            print(e)
            traceback.print_exc()
            return "[View Adjust Agent LLM] fails, can not recongnize instruction"
        
        return delta_extrinsic


    def func_update_extrinsic(self, scene, delta_extrinsic):
        scene.current_extrinsics[:,0,3] += delta_extrinsic['x']
        scene.current_extrinsics[:,1,3] += delta_extrinsic['y']
        scene.current_extrinsics[:,2,3] += delta_extrinsic['z']

        theta = delta_extrinsic['theta']  # positive indicates turn left 
        theta = theta/ 180 * np.pi
        T_theta = np.array([
                            [np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]
                        ])
        scene.current_extrinsics = np.matmul(T_theta, scene.current_extrinsics)

    def func_generate_extrinsic(self, scene, start_frame_idx, end_frame_idx):
        scene.current_extrinsics = inter_poses(scene.nerf_motion_extrinsics[start_frame_idx:end_frame_idx:3], scene.frames)