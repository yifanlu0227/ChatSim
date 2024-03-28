import openai 
from termcolor import colored
import traceback
import openai
import random
import os 

class AssetSelectAgent:
    def __init__(self, config):
        self.config = config
        self.asset_bank = {
            'audi' : "Audi_Q3_2023.blend",
            'benz_g' : "Benz_G.blend",
            'benz_s' : "Benz_S.blend",
            'mini' : "BMW_mini.blend",
            'cadillac' : "Cadillac_CT6.blend",
            'chevrolet' : "Chevrolet.blend",
            'dodge' : "Dodge_SRT_Hellcat.blend",
            'ferriari' : "Ferriari_f150.blend",
            'lamborghini' : "Lamborghini.blend",
            'rover' : "Land_Rover_range_rover.blend",
            'tank' : "M1A2_tank.blend",
            'police_car' : "Police_car.blend",
            'porsche' : "Porsche-911-4s-final.blend",
            'tesla_cybertruck' : "Tesla_cybertruck.blend",
            'tesla_roadster' : "Tesla_roadster.blend",
            'loader_truck' : "obstacles/Loader_truck.blend",
            'bulldozer' : "obstacles/Bulldozer.blend",
            'cement' : "obstacles/Cement_isolation_pier.blend",
            'excavator' : "obstacles/Excavator.blend",
            'sign_fence' : "obstacles/Sign_fence.blend",
            'cone' : "obstacles/Traffic_cone.blend"
        }
        self.assets_dir = config['assets_dir']

    def llm_selecting_asset(self, scene, message):
        try:
            q0 = "I will provide you with an operation statement to add and place a vehicle, and I need you to determine the car's color and type. "  

            q1 = "You need to return a JSON dictionary with 2 keys, including "

            q2 = "(1) 'color', representing in RGB with range from 0 to 255. If the color is not mentioned, the value is just 'default'."

            q3 = "(2) 'type', one of [audi, benz_g, benz_s, mini, cadillac, chevrolet, dodge, ferriari, lamborghini, rover, tank, police_car, porsche, tesla_cybertruck, tesla_roadster, cone, loader_truck, bulldozer, cement, excavator, sign_fence, random]. If the type is not mentioned or not in the type list, it defaults to random."

            q4 = "An example: Given operation statement 'add a black Rover at the front', you should return: {'color':[0,0,0], 'type':'Rover'}"

            q5 = "Note that you should not return any code or explanations, only provide a JSON dictionary."

            q6 = "The operation statement is:" + message

            prompt_list = [q0,q1,q2,q3,q4,q5,q6]

            result = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an assistant helping me to determine a car's color and type."}] + \
                    [{"role": "user", "content": q} for q in prompt_list]
            )
            answer = result['choices'][0]['message']['content']

            print(f"{colored('[Asset Agent LLM] deciding asset type and color', color='magenta', attrs=['bold'])} \
                    \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}")

            start = answer.index("{")
            answer = answer[start:]
            end = answer.rfind("}")
            answer = answer[:end+1]
            color_and_type = eval(answer)
            color_and_type['type'] = color_and_type['type'] if color_and_type['type'] != 'random' else random.choice(list(self.asset_bank.keys()))
            print(f"{colored('[Extracted Response>>>]', attrs=['bold'])} {color_and_type} \n")

        except Exception as e:
            print(e)
            traceback.print_exc()
            return "[Asset Agent LLM] deciding asset type and color fails."

        return color_and_type

    def llm_revise_added_cars(self, scene, message, added_car_dict):
        """ This function is a little go beyond asset_select_agent's role. It also consider the motion of the car

        It determine how to modify the dictionary about already added cars
        """
        try:
            q0 = "I will provide you with a dictionary in which each key is a vehicle id, and each value is the status description of the vehicle in the scene."

            q1 = "Specifically, description of the vehicle is also a dictionary. It has keys as follows:"
            
            q2 = "(1) 'x', vehicle's x position in meter. positive x is heading forward (2) 'y', vehicle's y position in meter. positive y is heading left " + \
                    "(3) 'color', vehicle's color in RGB. 'color' would be 'default' or a list represent the RGB values. If the color is not mentioned, the value is just 'default'."

            q3 = "(4) 'type', one of [audi, benz_g, benz_s, mini, cadillac, chevrolet, dodge, ferriari, lamborghini, rover, tank, police_car, porsche, tesla_cybertruck, tesla_roadster, cone, loader_truck, bulldozer, cement, excavator, sign_fence]. "

            q4 = "(5) 'action', vehicle's driving action, one of ['random', 'straight', 'turn left', 'turn right', 'change lane left', 'change lane right', 'static', 'back']"

            q5 = "(6) 'speed', vehicle's driving speed, one of ['random', 'fast', 'slow']"

            q6 = "(7) 'direction', one of ['away', 'close']. In ego view, moving forward is 'away' while moving towards is 'close'."

            q7 =  "I will get you a requirement. To follow my requirement, you should first find out which car I am describing, and then modify its status description dictionary according to my requirement. \
                For unmentioned properties, keep them unchanged."
            
            q8 = f"Now the dictionary is {added_car_dict}, and my requirement is {message}. "

            q9 = "Note that you should return a JSON dictionary, which only containing the specfic car in requirement with its modified status. \
                Just return the JSON dictionary, I'm not asking you to write code."

            prompt_list = [q0,q1,q2,q3,q4,q5,q6,q7,q8,q9]
            
            result = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an assistant helping me modify and return dictionaries."}] + \
                            [{"role": "user", "content": q} for q in prompt_list]
            )

            answer = result['choices'][0]['message']['content']

            print(f"{colored('[Asset Select Agent LLM] revising added cars', color='magenta', attrs=['bold'])}  \
                    \n{colored('[Raw Response>>>]', attrs=['bold'])} {answer}")

            start = answer.index("{")
            answer = answer[start:]
            end = answer.rfind('}')
            answer = answer[:end+1]
            modified_car_dict = eval(answer)

            print(f"{colored('[Extracted Response>>>]', attrs=['bold'])} {modified_car_dict} (number={len(modified_car_dict)})\n")

        except Exception as e:
            print(e)
            traceback.print_exc()
            return "[Asset Select Agent LLM] revising added cars fails."

        return modified_car_dict
    
    def func_retrieve_blender_file(self, scene):
        """Retrieve the path of the asset file given the asset type.
        """
        for car_name, car_info in scene.added_cars_dict.items():
            car_blender_file = self.asset_bank[car_info["type"].casefold()]
            car_info['blender_file'] = os.path.join(self.assets_dir, car_blender_file)
