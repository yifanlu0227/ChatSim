import openai 
import numpy as np
from termcolor import colored
import traceback
import openai

class ViewAdjustAgent:
    def __init__(self, config):
        self.config = config

    def llm_view_adjust(self, scene, message):
        try:
            # General task description
            q0 = "I will give you a transformation operation for my viewpoint, which may include translation in 'x', 'y', 'z' or a rotation 'theta' around z-axis. "
            # Interpretation of details
            q1 = "For translation, positive 'x' represents forward, positve 'y' represents left, and 'z' represents up. It follows a left-hand coordinate system." + \
                 "For rotation, postive 'theta' is counterclockwise. So from own perspective, my viewpoint turns to the left. 'theta' is in degree."
            # Return format
            q2 = "Given my operation, return a dictionary in JSON format, with keys 'x', 'y', 'z', 'theta'." + \
                 " Note that there is no need to return any code or explanations; only provide a JSON dictionary."
            # Few-shot examples
            q3 = "I will give you some examples:<user>: rotate the viewpoint 30 degrees to the left; <assistant>: {\n  'x': 0,\n  'y': 0,\n  'z': 0,\n  'theta': 30,\n } \
                                                <user>: move the viewpoint forward by 1; <assistant>: {\n  'x': 1,\n  'y': 0,\n  'z': 0,\n  'theta': 0,\n }  \
                                                <user>: move the viewpoint to the right by 1; <assistant>: {\n  'x': 0,\n  'y': -1,\n  'z': 0,\n  'theta': 0,\n} "
            
            
            q0 = "I will give you a transformation operation for my viewpoint, which may include translation in 'x', 'y', 'z' or a rotation 'theta' around z-axis. " # General task description
            q1 = "For translation, positive 'x' represents forward, positve 'y' represents left, and 'z' represents up. It follows a left-hand coordinate system." + \
                 "For rotation, postive 'theta' is counterclockwise. So from own perspective, my viewpoint turns to the left. 'theta' is in degree." # Interpretation of details
            q2 = "Given my operation, return a dictionary in JSON format, with keys 'x', 'y', 'z', 'theta'." # Return format
            q3 = "I will give you some examples:<user>: rotate the viewpoint 30 degrees to the left; <assistant>: {\n  'x': 0,\n  'y': 0,\n  'z': 0,\n  'theta': 30,\n } \
                                                <user>: move the viewpoint to the right by 1; <assistant>: {\n  'x': 0,\n  'y': -1,\n  'z': 0,\n  'theta': 0,\n} " # Few-shot examples

            result = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an assistant helping me to provide information and ultimately return a JSON dictionary."},
                    {"role": "user", "content": q0},
                    {"role": "user", "content": q1},
                    {"role": "user", "content": q2},
                    {"role": "user", "content": q3},
                    {"role": "user", "content": "<user>: Rotate the viewpoint 30 degrees to the left\
                                                <assistant>: {\n  'x': 0,\n  'y': 0,\n  'z': 0,\n  'theta': 30,\n } \n \
                                                <user>: move the viewpoint forward by 1 \
                                                <assistant>: {\n  'x': 1,\n  'y': 0,\n  'z': 0,\n  'theta': 0,\n }  \n \
                                                <user>: move the viewpoint to the right by 1 \
                                                <assistant>: {\n  'x': 0,\n  'y': -1,\n  'z': 0,\n  'theta': 0,\n}  \n \ "
                                                },
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